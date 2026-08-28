from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CoordinateSpace(str, Enum):
    CROP_4X = "crop_4x"
    CROP_20X = "crop_20x"
    WSI_LEVEL_0 = "wsi_level_0"
    WSI_LEVEL_2 = "wsi_level_2"
    WSI_LEVEL_4 = "wsi_level_4"
    ISLAND = "island"


class TransformType(str, Enum):
    RIGID = "rigid"               # Rotation + Translation
    SIMILARITY = "similarity"     # Rotation + Translation + Isotropic Scale
    AFFINE = "affine"             # General Affine (includes shear/anisotropic scale)
    IDENTITY = "identity"


class RegistrationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"
    MANUAL_REVIEW = "MANUAL_REVIEW"


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
    format: str                   # "kfb", "svs", "tif", etc.
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
        # Fallback approximation for standard KFB pyramids
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
class ROI:
    id: str
    source_slide_id: str
    coordinate_space: CoordinateSpace
    center_lvl0: Tuple[float, float]
    size_pixels: Tuple[int, int]   # (width, height)
    physical_fov: Optional[PhysicalFOV] = None
    is_mirrored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCMetrics:
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
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    sample_id: str
    moving_stain: str
    reference_stain: str
    status: RegistrationStatus
    reason: str
    transform_matrix_3x3: Optional[List[List[float]]] = None
    metrics: QCMetrics = field(default_factory=QCMetrics)
    qc_details: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
