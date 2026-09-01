"""
CrossStainWSI: Auditable Cross-Stain Whole-Slide Image Registration & Multi-Scale Workflow Toolkit
"""

__version__ = "0.3.0"

from crossstainwsi.benchmark import (
    BenchmarkEvaluator,
    BenchmarkHarness,
    BenchmarkSummary,
    CaseEvaluationResult,
    GroundTruthParams,
    PerturbationCase,
    SyntheticPerturbationGenerator,
)
from crossstainwsi.domain import (
    CoordinateSpace,
    EvidenceRecord,
    EvidenceView,
    FailureCode,
    PyramidLevel,
    QCMetrics,
    RegistrationResult,
    RegistrationStatus,
    ROI,
    SlideSpec,
    TransformType,
)
from crossstainwsi.gui import CrossStainWSIGUI, launch_gui
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
    NormalizedMutualInformationMatcher,
    PhaseCorrelationMatcher,
    SiftMatcher,
    TemplateMatcher,
    compute_nmi,
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
from crossstainwsi.representation import (
    CanonicalRepresentationSet,
    ChannelEvidenceSelector,
    FluorescenceAdapter,
    GenericBrightfieldAdapter,
    IHCDeconvolutionAdapter,
    RepresentationBuilder,
)
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
    "EvidenceView",
    "EvidenceRecord",
    "FailureCode",
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
    "NormalizedMutualInformationMatcher",
    "compute_nmi",
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
    "CrossStainWSIGUI",
    "launch_gui",
    "CanonicalRepresentationSet",
    "RepresentationBuilder",
    "GenericBrightfieldAdapter",
    "IHCDeconvolutionAdapter",
    "ChannelEvidenceSelector",
    "FluorescenceAdapter",
    "GroundTruthParams",
    "PerturbationCase",
    "SyntheticPerturbationGenerator",
    "BenchmarkEvaluator",
    "BenchmarkSummary",
    "CaseEvaluationResult",
    "BenchmarkHarness",
]
