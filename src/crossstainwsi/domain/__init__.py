from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CoordinateSpace(str, Enum):
    EVIDENCE_VIEW = "evidence_view"
    WSI_LEVEL_0 = "wsi_level_0"
    WSI_LEVEL_2 = "wsi_level_2"
    WSI_LEVEL_4 = "wsi_level_4"
    TISSUE_ISLAND = "tissue_island"


class TransformType(str, Enum):
    RIGID = "rigid"               # Rotation + Translation
    SIMILARITY = "similarity"     # Rotation + Translation + Isotropic Scale
    AFFINE = "affine"             # General Affine
    IDENTITY = "identity"


class RegistrationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class FailureCode(str, Enum):
    """细粒度失败分类体系 (Failure Taxonomy)"""
    NONE = "NONE"
    REFERENCE_ANCHOR_FAIL = "REFERENCE_ANCHOR_FAIL"
    MIRROR_AMBIGUOUS = "MIRROR_AMBIGUOUS"
    ROTATION_AMBIGUOUS = "ROTATION_AMBIGUOUS"
    LOW_INFORMATION = "LOW_INFORMATION"
    LOW_OVERLAP = "LOW_OVERLAP"
    FEATURE_MATCH_WEAK = "FEATURE_MATCH_WEAK"
    STRUCTURE_CONFLICT = "STRUCTURE_CONFLICT"
    CROSS_SCALE_CONFLICT = "CROSS_SCALE_CONFLICT"
    MODALITY_ADAPTER_FAIL = "MODALITY_ADAPTER_FAIL"
    LOCAL_REFINEMENT_FAIL = "LOCAL_REFINEMENT_FAIL"
    SECTION_CORRESPONDENCE_WEAK = "SECTION_CORRESPONDENCE_WEAK" # 替代确定性的生物漂移断言


@dataclass(frozen=True)
class PyramidLevel:
    level: int
    dimensions: Tuple[int, int]   # (width, height)
    downsample: float


@dataclass(frozen=True)
class SlideSpec:
    id: str
    sample_id: str
    stain: str
    path: Path
    format: str                   # "kfb", "svs", "ndpi", "ome.tif", etc.
    dimensions: Tuple[int, int]   # Level 0 (width, height)
    levels: List[PyramidLevel]
    mpp_x: float = 0.44243
    mpp_y: float = 0.44243
    mpp_source: str = "configured_override"
    properties: Dict[str, Any] = field(default_factory=dict)

    def get_level_downsample(self, level: int) -> float:
        for lvl in self.levels:
            if lvl.level == level:
                return lvl.downsample
        return float(2 ** level)

    def get_level_dimensions(self, level: int) -> Tuple[int, int]:
        for lvl in self.levels:
            if lvl.level == level:
                return lvl.dimensions
        ds = self.get_level_downsample(level)
        return (int(round(self.dimensions[0] / ds)), int(round(self.dimensions[1] / ds)))


@dataclass
class PhysicalFOV:
    width_um: float
    height_um: float


@dataclass
class EvidenceView:
    """
    通用证据视场 (彻底替代写死的 crop4 / crop20)
    表示输入证据截图或请求输出视场的几何与物理属性
    """
    id: str
    width_px: int
    height_px: int
    nominal_magnification: float = 4.0   # 名义放大倍率 (如 4.0, 10.0, 20.0, 40.0)
    mpp_xy: Optional[Tuple[float, float]] = None
    source_path: Optional[Path] = None
    center_relation: str = "concentric"  # 与主参考视场的空间关系 (concentric / offset)
    is_mirrored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ROI:
    id: str
    source_slide_id: str
    coordinate_space: CoordinateSpace
    center_lvl0: Tuple[float, float]
    size_pixels: Tuple[int, int]
    physical_fov: Optional[PhysicalFOV] = None
    is_mirrored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceRecord:
    """
    匹配器返回的通用结构化证据记录 (支持可插拔 Feature / Structure / NMI 匹配器)
    """
    backend: str                        # "LoFTR", "SIFT", "DISK", "PhaseCorrelation", "ContourDistance", "NMI"
    support_score: float = 0.0          # 综合支持度评分 (0.0 ~ 1.0)
    inliers: int = 0
    inlier_ratio: float = 0.0
    spatial_coverage: float = 0.0
    median_reproj_error: float = 999.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    residual_dispersion_px: float = 0.0 # 局部块位移场方差 (学习 PALOM consensus)
    is_independent_evidence: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCMetrics:
    """统一向后兼容的质控指标包"""
    inliers: int = 0
    matches: int = 0
    inlier_ratio: float = 0.0
    spatial_coverage: float = 0.0
    median_reproj_error: float = 999.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    ncc_score: Optional[float] = None
    mask_iou: Optional[float] = None
    background_agreement: Optional[float] = None
    edge_corr: Optional[float] = None
    method: str = "unknown"
    failure_code: FailureCode = FailureCode.NONE
    evidence_records: List[EvidenceRecord] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    sample_id: str
    moving_stain: str
    reference_stain: str
    status: RegistrationStatus
    reason: str
    failure_code: FailureCode = FailureCode.NONE
    transform_matrix_3x3: Optional[List[List[float]]] = None
    metrics: QCMetrics = field(default_factory=QCMetrics)
    qc_details: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
