"""
资产自动发现与扫描器 (Asset Discoverer)
从目录中自动梳理 WSI 切片与任意倍率已有的截图证据 (支持 .tif, .png, .jpg 等任意格式及染色名自动推导)
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re

from crossstainwsi.inventory.assets import AssetInventory, ROIEvidence, SampleAssets, SlideAsset


class AssetDiscoverer:
    """
    负责扫描数据目录并生成完整的 AssetInventory
    支持任意倍率截图与文件名模式识别
    """
    def __init__(
        self,
        base_dir: Path,
        tiff_dir: Optional[Path] = None,
        mirrored_sample_ids: Optional[Set[str]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.tiff_dir = Path(tiff_dir) if tiff_dir else None
        self.mirrored_sample_ids = mirrored_sample_ids or set()

    def discover(self) -> AssetInventory:
        samples_dict: Dict[str, SampleAssets] = {}

        if not self.base_dir.exists():
            return AssetInventory()

        # 1. 扫描所有 WSI 切片文件 (支持 kfb, svs, ndpi, mrxs, ome.tiff 等)
        wsi_extensions = {".kfb", ".svs", ".ndpi", ".mrxs", ".tiff", ".tif"}
        for p in self.base_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in wsi_extensions:
                continue

            # 防护：如果该文件位于 tiff_dir 下，或者是截图命名模式 (如 -4x, -20x)，跳过 WSI 扫描
            if self.tiff_dir and (self.tiff_dir in p.parents or p.parent == self.tiff_dir):
                continue
            if re.search(r"[-_](\d+(?:\.\d+)?)[xX]", p.stem):
                continue

            stem = p.stem
            if stem.lower().endswith(".ome"):
                stem = stem[:-4]
            parts = stem.split("-")
            if len(parts) >= 2:
                stain = parts[-1]
                sample_id = "-".join(parts[:-1])
            else:
                stain = "Unknown"
                sample_id = stem

            if sample_id not in samples_dict:
                samples_dict[sample_id] = SampleAssets(sample_id=sample_id)

            samples_dict[sample_id].slides[stain] = SlideAsset(
                stain=stain,
                path=p,
                format=p.suffix.lstrip(".").lower(),
            )

        # 2. 如果提供了 tiff_dir，扫描已有的截图证据 (基于已发现的 sample_id 列表进行最长前缀匹配)
        img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        known_sids = sorted(samples_dict.keys(), key=len, reverse=True)

        if self.tiff_dir and self.tiff_dir.exists():
            for p in self.tiff_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in img_extensions:
                    continue

                stem = p.stem
                matched_sid = None

                # 优先匹配已有的 WSI 样本号
                for sid in known_sids:
                    if stem.lower().startswith(sid.lower()):
                        matched_sid = sid
                        break

                # 若未能在已知 WSI 中匹配，尝试通用正则推断样本号
                if not matched_sid:
                    m_mag_pos = re.search(r"[-_](\d+(?:\.\d+)?)[xX]", stem)
                    if m_mag_pos:
                        raw_prefix = stem[:m_mag_pos.start()]
                        matched_sid = self._match_sample_id(raw_prefix, known_sids) or raw_prefix

                if matched_sid:
                    if matched_sid not in samples_dict:
                        samples_dict[matched_sid] = SampleAssets(sample_id=matched_sid)

                    ev = samples_dict[matched_sid].roi_evidence
                    ev.is_mirrored = matched_sid in self.mirrored_sample_ids

                    # 解析倍率与染色
                    rem = stem[len(matched_sid):].lstrip("-_") if stem.lower().startswith(matched_sid.lower()) else stem
                    m_mag = re.search(r"(\d+(?:\.\d+)?)[xX]", rem)
                    mag_val = float(m_mag.group(1)) if m_mag else 4.0

                    stain_cand = re.sub(r"[-_]?\d+(?:\.\d+)?[xX].*$", "", rem).strip("-_")
                    if stain_cand:
                        ev.inferred_reference_stain = stain_cand

                    ev.add_evidence_path(p, nominal_mag=mag_val)

        return AssetInventory(samples=samples_dict)

    @staticmethod
    def _match_sample_id(query: str, available_ids: List[str]) -> Optional[str]:
        q_clean = query.strip().lower()
        for sid in available_ids:
            if sid.lower() == q_clean:
                return sid
        # 兼容无连字符情况
        q_no_dash = q_clean.replace("-", "").replace("_", "")
        for sid in available_ids:
            if sid.lower().replace("-", "").replace("_", "") == q_no_dash:
                return sid
        return None
