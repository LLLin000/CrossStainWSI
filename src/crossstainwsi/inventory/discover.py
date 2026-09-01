"""
资产自动发现与扫描器 (Asset Discoverer)
从目录中自动梳理 WSI 切片与已有的截图证据 (支持 .tif, .png, .jpg 等任意格式及染色名自动推导)
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re

from crossstainwsi.inventory.assets import AssetInventory, ROIEvidence, SampleAssets, SlideAsset


class AssetDiscoverer:
    """
    负责扫描数据目录并生成完整的 AssetInventory
    支持灵活的文件名模式与染色识别
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

        # 1. 扫描所有 WSI 切片文件 (支持 kfb, svs, ndpi, mrxs 等)
        wsi_extensions = {".kfb", ".svs", ".ndpi", ".mrxs"}
        for p in self.base_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in wsi_extensions:
                stem = p.stem
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

        # 2. 如果提供了 tiff_dir，扫描已有的截图证据 (支持 .tif, .tiff, .png, .jpg, .jpeg)
        img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        if self.tiff_dir and self.tiff_dir.exists():
            for p in self.tiff_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in img_extensions:
                    continue

                stem = p.stem
                # 模式 1: 包含染色名，例如 3-4w-2-masson-4x, 3-4w-2_HE_20X
                # 模式 2: 仅包含倍率，例如 3-4w-2-4x, 3-4w-2_20X
                match_named_4x = re.match(r"^(.+?)[-_]([a-zA-Z]+)[-_]4x.*$", stem, re.IGNORECASE)
                match_named_20x = re.match(r"^(.+?)[-_]([a-zA-Z]+)[-_]20x.*$", stem, re.IGNORECASE)
                match_simple_4x = re.match(r"^(.+?)[-_]4x.*$", stem, re.IGNORECASE)
                match_simple_20x = re.match(r"^(.+?)[-_]20x.*$", stem, re.IGNORECASE)

                if match_named_4x:
                    sid_raw, stain_raw = match_named_4x.group(1), match_named_4x.group(2)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.crop_4x_path = p
                        ev.inferred_reference_stain = stain_raw
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids
                elif match_named_20x:
                    sid_raw, stain_raw = match_named_20x.group(1), match_named_20x.group(2)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.crop_20x_path = p
                        ev.inferred_reference_stain = stain_raw
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids
                elif match_simple_4x:
                    sid_raw = match_simple_4x.group(1)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.crop_4x_path = p
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids
                elif match_simple_20x:
                    sid_raw = match_simple_20x.group(1)
                    matched_id = self._match_sample_id(sid_raw, list(samples_dict.keys()))
                    if matched_id:
                        ev = samples_dict[matched_id].roi_evidence
                        ev.crop_20x_path = p
                        ev.is_mirrored = matched_id in self.mirrored_sample_ids

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
