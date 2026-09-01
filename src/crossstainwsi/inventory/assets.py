"""
样本资产清单 (Asset Inventory) 模型
完全支持通用 EvidenceView 证据集合与原生 WSI ROI 标注
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crossstainwsi.domain import EvidenceView


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
    """已有的 ROI 证据材料集合 (支持任意多尺度截图与原生坐标)"""
    evidence_views: List[EvidenceView] = field(default_factory=list)
    native_center_l0: Optional[Tuple[float, float]] = None
    native_size_l0: Optional[Tuple[int, int]] = None  # 严格为 Level 0 像素尺寸 (w0, h0)
    inferred_reference_stain: Optional[str] = None
    is_mirrored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 构造函数支持旧版关键字参数 (crop_4x_path, crop_20x_path)
    def __init__(
        self,
        evidence_views: Optional[List[EvidenceView]] = None,
        crop_4x_path: Optional[Path] = None,
        crop_20x_path: Optional[Path] = None,
        native_center_l0: Optional[Tuple[float, float]] = None,
        native_size_l0: Optional[Tuple[int, int]] = None,
        inferred_reference_stain: Optional[str] = None,
        is_mirrored: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.evidence_views = list(evidence_views) if evidence_views else []
        self.native_center_l0 = native_center_l0
        self.native_size_l0 = native_size_l0
        self.inferred_reference_stain = inferred_reference_stain
        self.is_mirrored = is_mirrored
        self.metadata = dict(metadata) if metadata else {}

        if crop_4x_path is not None:
            self.add_evidence_path(crop_4x_path, nominal_mag=4.0)
        if crop_20x_path is not None:
            self.add_evidence_path(crop_20x_path, nominal_mag=20.0)

    @property
    def crop_4x_path(self) -> Optional[Path]:
        for v in self.evidence_views:
            if abs(v.nominal_magnification - 4.0) < 0.5 and v.source_path:
                return v.source_path
        return None

    @crop_4x_path.setter
    def crop_4x_path(self, path: Optional[Path]) -> None:
        if path is not None:
            self.add_evidence_path(path, nominal_mag=4.0)

    @property
    def crop_20x_path(self) -> Optional[Path]:
        for v in self.evidence_views:
            if abs(v.nominal_magnification - 20.0) < 1.0 and v.source_path:
                return v.source_path
        return None

    @crop_20x_path.setter
    def crop_20x_path(self, path: Optional[Path]) -> None:
        if path is not None:
            self.add_evidence_path(path, nominal_mag=20.0)

    @property
    def first_evidence_path(self) -> Optional[Path]:
        if self.evidence_views and self.evidence_views[0].source_path:
            return self.evidence_views[0].source_path
        return None

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
        return bool(self.evidence_views) or self.has_native_roi

    def add_evidence_path(
        self,
        path: Path,
        nominal_mag: float = 4.0,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
        mpp_xy: Optional[Tuple[float, float]] = None,
    ) -> EvidenceView:
        for v in self.evidence_views:
            if v.source_path == path:
                return v

        # 尝试快速读取图像文件头部的真实尺寸
        real_w, real_h = width_px, height_px
        if (real_w is None or real_h is None) and path.exists():
            try:
                from PIL import Image
                with Image.open(path) as img:
                    real_w, real_h = img.size
            except Exception:
                real_w, real_h = 2257, 1310
        else:
            real_w = real_w or 2257
            real_h = real_h or 1310

        view = EvidenceView(
            id=f"evidence_{path.stem}",
            width_px=real_w,
            height_px=real_h,
            nominal_magnification=nominal_mag,
            mpp_xy=mpp_xy,
            source_path=path,
            is_mirrored=self.is_mirrored,
        )
        self.evidence_views.append(view)
        self.evidence_views.sort(key=lambda item: item.nominal_magnification)
        return view

    def get_primary_anchor_evidence(self) -> Optional[EvidenceView]:
        return self.evidence_views[0] if self.evidence_views else None

    def get_secondary_verification_evidence(self) -> Optional[EvidenceView]:
        return self.evidence_views[1] if len(self.evidence_views) > 1 else None


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
        ev_items = [f"{v.nominal_magnification}x ({v.source_path.name if v.source_path else v.id})" for v in self.roi_evidence.evidence_views]
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
