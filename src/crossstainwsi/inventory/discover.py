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
            if p.is_file() and p.suffix.lower() in wsi_extensions:
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

        # 2. 如果提供了 tiff_dir，扫描已有的截图证据 (支持任意倍率如 4x, 10x, 20x, 40x)
        img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        if self.tiff_dir and self.tiff_dir.exists():
            for p in self.tiff_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in img_extensions:
                    continue

                stem = p.stem
                # 模式 1: 包含染色名与倍率，例如 3-4w-2-masson-4x, 3-4w-2_HE_10X, 3-4w-2-HE-40x
                # 模式 2: 仅包含倍率，例如 3-4w-2-4x, 3-4w-2_20X, 3-4w-2-10x
                match_named = re.match(r"^(.+?)[-_]([a-zA-Z]+)[-_](\d+(?:\.\d+)?)[xX].*$", stem)
                match_simple = re.match(r"^(.+?)[-_](\d+(?:\.\d+)?)[xX].*$", stem)

                if match_named:
                    sid_raw, stain_raw, mag_str = match_named.group(1), match_named.group(2), match_named.group(3)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids
                        ev.inferred_reference_stain = stain_raw
                        ev.add_evidence_path(p, nominal_mag=float(mag_str))
                elif match_simple:
                    sid_raw, mag_str = match_simple.group(1), match_simple.group(2)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids
                        ev.add_evidence_path(p, nominal_mag=float(mag_str))

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
