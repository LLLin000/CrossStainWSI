"""
CrossStainWSI: Auditable Cross-Stain Whole-Slide Image Registration Toolkit
"""

__version__ = "0.1.0"

from crossstainwsi.domain import (
    CoordinateSpace,
    PyramidLevel,
    QCMetrics,
    RegistrationResult,
    RegistrationStatus,
    ROI,
    SlideSpec,
    TransformType,
)
from crossstainwsi.io import ImageCropReader, KFBReader, SlideReader
from crossstainwsi.matching import (
    ImageMatcher,
    LoFTRMatcher,
    MatchResult,
    PhaseCorrelationMatcher,
    SiftMatcher,
    TemplateMatcher,
)
from crossstainwsi.pipeline import BatchRunner, PipelineConfig, SampleRunner
from crossstainwsi.qc import QCRuleConfig, QCRuleEngine, compute_same_image_metrics
from crossstainwsi.registration import (
    AnchorResult,
    GlobalAlignmentResult,
    GlobalRegistrar,
    LocalRefineResult,
    LocalRefiner,
    ReferenceAnchorLocator,
)
from crossstainwsi.reporting import ContactSheetGenerator, ReportGenerator
from crossstainwsi.sampling import WSISampler
from crossstainwsi.tissue import TissueIsland, TissueSegmenter
from crossstainwsi.transforms import TransformGraph

__all__ = [
    "__version__",
    "CoordinateSpace",
    "PyramidLevel",
    "QCMetrics",
    "RegistrationResult",
    "RegistrationStatus",
    "ROI",
    "SlideSpec",
    "TransformType",
    "SlideReader",
    "KFBReader",
    "ImageCropReader",
    "ImageMatcher",
    "MatchResult",
    "SiftMatcher",
    "TemplateMatcher",
    "LoFTRMatcher",
    "PhaseCorrelationMatcher",
    "ReferenceAnchorLocator",
    "AnchorResult",
    "GlobalRegistrar",
    "GlobalAlignmentResult",
    "LocalRefiner",
    "LocalRefineResult",
    "WSISampler",
    "TissueIsland",
    "TissueSegmenter",
    "TransformGraph",
    "QCRuleConfig",
    "QCRuleEngine",
    "compute_same_image_metrics",
    "ReportGenerator",
    "ContactSheetGenerator",
    "PipelineConfig",
    "SampleRunner",
    "BatchRunner",
]
