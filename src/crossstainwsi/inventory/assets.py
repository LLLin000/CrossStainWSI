"""
样本资产清单 (Asset Inventory) 模型
清晰表达用户已有的材料 (WSI 切片、历史截图证据、坐标标注等)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SlideAsset:
    """单个染色 WSI 切片资产"""
    stain: str
    path: Path
    format: str
    dimensions: Optional[Tuple[int, int]] = None
    mpp: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ROIEvidence:
    """已有的 ROI 证据材料 (可选)"""
    crop_4x_path: Optional[Path] = None
    crop_20x_path: Optional[Path] = None
    native_center_l0: Optional[Tuple[float, float]] = None
    native_size_l0: Optional[Tuple[int, int]] = None
    inferred_reference_stain: Optional[str] = None
    is_mirrored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_4x(self) -> bool:
        return self.crop_4x_path is not None

    @property
    def has_20x(self) -> bool:
        return self.crop_20x_path is not None

    @property
    def has_native_roi(self) -> bool:
        return self.native_center_l0 is not None

    @property
    def has_any_evidence(self) -> bool:
        return self.has_4x or self.has_20x or self.has_native_roi


@dataclass
class SampleAssets:
    """单个样本的所有可用资产"""
    sample_id: str
    slides: Dict[str, SlideAsset] = field(default_factory=dict)
    roi_evidence: ROIEvidence = field(default_factory=ROIEvidence)

    def has_stain(self, stain: str) -> bool:
        return stain.lower() in [s.lower() for s in self.slides.keys()]

    def get_slide(self, stain: str) -> Optional[SlideAsset]:
        for k, v in self.slides.items():
            if k.lower() == stain.lower():
                return v
        return None

    def describe(self) -> str:
        slide_list = ", ".join(self.slides.keys()) if self.slides else "None"
        ev_items = []
        if self.roi_evidence.has_4x:
            ev_items.append("4x crop")
        if self.roi_evidence.has_20x:
            ev_items.append("20x crop")
        if self.roi_evidence.has_native_roi:
            ev_items.append("Native ROI coords")
        ev_str = ", ".join(ev_items) if ev_items else "No existing ROI evidence"
        return f"Sample [{self.sample_id}]: Slides=[{slide_list}], Evidence=[{ev_str}]"


@dataclass
class AssetInventory:
    """全项目/批次资产清单"""
    samples: Dict[str, SampleAssets] = field(default_factory=dict)

    def get_sample(self, sample_id: str) -> Optional[SampleAssets]:
        if sample_id in self.samples:
            return self.samples[sample_id]
        q_clean = sample_id.strip().lower()
        for sid, sample in self.samples.items():
            if sid.lower() == q_clean:
                return sample
        q_no_dash = q_clean.replace("-", "").replace("_", "")
        for sid, sample in self.samples.items():
            if sid.lower().replace("-", "").replace("_", "") == q_no_dash:
                return sample
        return None
    def summary(self) -> Dict[str, Any]:
        return {
            "total_samples": len(self.samples),
            "sample_ids": list(self.samples.keys()),
            "samples_with_crops": [s_id for s_id, s in self.samples.items() if s.roi_evidence.has_any_evidence],
            "samples_wsi_only": [s_id for s_id, s in self.samples.items() if not s.roi_evidence.has_any_evidence],
        }
