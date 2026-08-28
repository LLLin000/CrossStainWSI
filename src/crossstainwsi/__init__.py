"""
CrossStainWSI: Auditable Cross-Stain Whole-Slide Image Registration & Workflow Toolkit
"""

__version__ = "0.2.0"

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
from crossstainwsi.inventory import (
    AssetDiscoverer,
    AssetInventory,
    ROIEvidence,
    SampleAssets,
    SlideAsset,
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
from crossstainwsi.planning import (
    AcquisitionProfile,
    ExecutionPlan,
    StainRequirement,
    TaskType,
    UserGoal,
    ViewSpec,
    WorkflowPlanner,
)
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
from crossstainwsi.review import (
    ArtifactTier,
    ConfidenceTier,
    RunVerdict,
    resolve_artifact_dir,
)
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
    "AssetInventory",
    "SampleAssets",
    "SlideAsset",
    "ROIEvidence",
    "AssetDiscoverer",
    "UserGoal",
    "ViewSpec",
    "StainRequirement",
    "AcquisitionProfile",
    "ExecutionPlan",
    "TaskType",
    "WorkflowPlanner",
    "ArtifactTier",
    "ConfidenceTier",
    "RunVerdict",
    "resolve_artifact_dir",
]
