# CrossStainWSI — Agent Context

## Project identity

CrossStainWSI is an auditable, adaptive cross-stain whole-slide image (WSI) registration and multi-scale region extraction toolkit.

It unifies cross-stain histology registration by decoupling **Input Evidence** (WSI files, existing crop screenshots, native coordinates, annotations) from **Output Requirements** (custom physical FOV, publication-resolution 4x/20x/10x views, 300 DPI TIFFs, overlays, QC audits).

Detailed Technical Documents:
- **Product Requirements Document (PRD)**: [`docs/PRD.md`](docs/PRD.md)
- **Detailed Design Specification**: [`docs/DESIGN.md`](docs/DESIGN.md)

The system workflow is:

```text
ASSET DISCOVERY (inventory)
        ↓
WORKFLOW PLANNER (planning) → ExecutionPlan (TaskType, ConfidenceTier)
        ↓
PREFLIGHT & VALIDATION (missing required stain checks)
        ↓
REFERENCE ROI (Native Coordinates or Dual-Scale SIFT/NCC Anchor)
        ↓
CROSS-STAIN REGISTRATION (Canonical Representations + Pluggable Matchers)
        ↓
TRANSFORM GRAPH (Parent-Child Geometry)
        ↓
SINGLE-PASS LEVEL-0 SAMPLING (WSISampler with WARP_INVERSE_MAP)
        ↓
QC EVALUATION & SAFETY GATING (PASS / REVIEW / ABSTAIN)
        ↓
ARTIFACT ROUTING (final/ vs review/ vs debug/)
        ↓
STRUCTURED REPORTING
```

## Repository state

- Local repository: `D:\L\AI\CrossStainWSI`
- Branch: `main`
- Git remote: `https://github.com/LLLin000/CrossStainWSI` (Public)
- Version: `0.2.0`
- Tests: `16 passed in tests/`

Current package structure:

```text
CrossStainWSI/
├─ CONTEXT.md
├─ .gitignore
├─ pyproject.toml
├─ docs/
│  ├─ PRD.md              # 完整产品需求与技术规格文档
│  └─ DESIGN.md           # 详细架构与设计规范
├─ benchmarks/            # 本地测试切片 (OME-TIFF，已加入 .gitignore)
│  ├─ ihc/
│  └─ cycif/
├─ reference_impl/        # 历史单文件原型与参考实现
├─ src/
│  └─ crossstainwsi/
│     ├─ domain/          # SlideSpec, ROI, QCMetrics, RegistrationResult
│     ├─ inventory/       # SlideAsset, ROIEvidence, SampleAssets, AssetDiscoverer
│     ├─ planning/        # UserGoal, ViewSpec, AcquisitionProfile, ExecutionPlan, WorkflowPlanner
│     ├─ review/          # ArtifactTier (final/review/debug), ConfidenceTier, RunVerdict
│     ├─ io/              # SlideReader, KFBReader, ImageCropReader
│     ├─ tissue/          # TissueIsland, TissueSegmenter
│     ├─ matching/        # SiftMatcher, TemplateMatcher, LoFTRMatcher, PhaseCorrelationMatcher
│     ├─ registration/    # ReferenceAnchorLocator, GlobalRegistrar, LocalRefiner
│     ├─ transforms/      # geom (h, affine, rotations), TransformGraph
│     ├─ sampling/        # WSISampler (Level-0 and Level-2 inverse warp sampling)
│     ├─ qc/              # compute_same_image_metrics, QCRuleEngine
│     ├─ reporting/       # ReportGenerator, ContactSheetGenerator
│     ├─ pipeline/        # PipelineConfig, SampleRunner, BatchRunner
│     ├─ gui/             # 独立 Tkinter 工作台 (CrossStainWSIGUI)
│     └─ cli.py           # CLI: discover, plan, run, batch, gui
└─ tests/
   ├─ test_domain.py
   ├─ test_transforms.py
   ├─ test_tissue.py
   ├─ test_qc.py
   ├─ test_matching.py
   ├─ test_planning.py
   ├─ test_inventory.py
   └─ test_gui.py
```

## Core Tasks & Confidence Tiers

1. **Task A: Native ROI on Reference WSI (`TaskType.NATIVE_ROI_MATCH`)**
   - User provides Level-0 coordinates $(x, y, w, h)$ or selects on thumbnail.
   - Zero anchor ambiguity error; `ConfidenceTier.TIER_A_NATIVE`.
   - No TIFF inputs required.

2. **Task B: Reproduce Single 4x Crop (`TaskType.SINGLE_CROP_REPRODUCE`)**
   - User provides a 4x screenshot.
   - SIFT/NCC localization in Reference WSI; 20x sampled from WSI Level 0.
   - `ConfidenceTier.TIER_C_SINGLE_CROP`.

3. **Task C: Strict Dual-Scale Reproduction (`TaskType.DUAL_SCALE_REPRODUCE`)**
   - User provides both 4x and 20x screenshots.
   - 4x coarse retrieval + 20x independent high-resolution verification.
   - `ConfidenceTier.TIER_B_DUAL_SCALE`.

4. **Task D: High-Mag 20x-Only Assisted (`TaskType.HIGH_MAG_ASSISTED`)**
   - User provides only 20x crop. Requires approximate position or user guidance.
   - `ConfidenceTier.TIER_D_HIGH_MAG_ASSISTED`.

5. **Task E: Whole-Slide Registration (`TaskType.WHOLE_SLIDE_REGISTER`)**
   - Inter-slide coordinate transformation without specific ROI.
   - Outputs overview alignment matrix and transform graph.

## Safety & Artifact Routing Invariants

- **`final/`**: Strictly reserved for runs passing all criteria (`required stains present + anchor valid + global similarity accepted + requested views confirmed + Level 0 sampled`).
- **`review/`**: Marginal confidence, mild serial section variance, or fallback local refinement. Requires manual visual confirmation.
- **`debug/`**: Rejected runs, ambiguous anchors, scale deformations, or missing required stains (`INCOMPLETE`). **No fake publication TIFFs are generated in final/**.
