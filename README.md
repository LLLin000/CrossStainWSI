# CrossStainWSI

[English](README.md) | [简体中文](README_zh.md)

**CrossStainWSI** is an auditable, adaptive cross-stain whole-slide image (WSI) registration and multi-scale region extraction toolkit for digital pathology and biomedical research.
---

## Key Features

- **Decoupled Inputs & Outputs**: Separates *Input Evidence* (WSI files, optional historical screenshots, native Level-0 coordinates) from *Output Requirements* (publication-grade 300 DPI crops, overlays, contact sheets, arbitrary physical FOVs).
- **Adaptive Workflow Planner**: Automatically inspects available materials and formulates an `ExecutionPlan` tailored to your goal.
- **Deep Morphology Alignment**: Combines multi-angle LoFTR deep feature matching, connected tissue island isolation, and bounded local residual refinement (Local LoFTR + Sobel phase correlation).
- **Single-Pass Level-0 Sampling**: Direct inverse warp sampling (`cv2.warpAffine` + `WARP_INVERSE_MAP`) straight from WSI Level 0, completely avoiding cumulative rotation blur, artificial white corners, and shear deformations.
- **Strict QC & Safety Gating**: Segregates outputs into `final/` (PASS), `review/` (WARN / manual review), and `debug/` (ABSTAIN / INCOMPLETE). Never generates misleading publication TIFFs when alignment evidence is weak.

---

## Architecture Overview

```text
               User / Data Directory
                        │
                        ▼
             ┌─────────────────────┐
             │   Asset Discovery   │  (inventory: WSI, 4x/20x crops, native coords)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Workflow Planner   │  (planning: UserGoal + AcquisitionProfile)
             └──────────┬──────────┘
                        │
                        ▼
                 ExecutionPlan  (TaskType, ConfidenceTier, Gating)
                        │
                        ▼
             ┌─────────────────────┐
             │  Registration Core  │  (tissue islands, global LoFTR, local refiner)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Transform Graph   │  (M_20x_to_L0 = S · M_4x_to_L2 · M_local^-1 · M_20x_to_4x)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Level-0 Sampling   │  (WSISampler single-pass inverse map)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Artifact Routing   │  (final/ vs review/ vs debug/)
             └─────────────────────┘
```

---

## Task Types & Confidence Tiers

| Task Type | Input Available | Anchor Strategy | Confidence Tier |
| :--- | :--- | :--- | :--- |
| **Task A: `NATIVE_ROI_MATCH`** | WSI only (Coordinates / Box) | Exact Level-0 coordinates (0 anchor error) | **Tier A (Exact)** |
| **Task B: `SINGLE_CROP_REPRODUCE`** | WSI + 4× Screenshot | SIFT multi-angle + NCC template fallback | **Tier C (Medium)** |
| **Task C: `DUAL_SCALE_REPRODUCE`** | WSI + 4× & 20× Screenshots | 4× retrieval + 20× independent verification | **Tier B (High)** |
| **Task D: `HIGH_MAG_ASSISTED`** | WSI + 20× Screenshot only | High-magnification assisted search | **Tier D (Assisted)** |
| **Task E: `WHOLE_SLIDE_REGISTER`** | Multiple WSIs (No ROI) | Inter-slide global coordinate transformation | **Tier A (Exact)** |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/LLLin000/CrossStainWSI.git
cd CrossStainWSI

# Install in editable mode
pip install -e .
```

### Dependencies
- Python >= 3.10
- PyTorch & Kornia (for LoFTR deep morphological matching)
- OpenCV (`cv2`)
- `kfbslide` (or OpenSlide for digital pathology WSI I/O)
- Pillow, NumPy, SciPy

---

## Command Line Interface (CLI)

### 1. Discover Assets
Scan and summarize available WSI slides and existing screenshot evidence:
```bash
crossstainwsi discover --base-dir /path/to/wsi --tiff-dir /path/to/crops
```

### 2. Inspect Execution Plan
Preview how the engine will handle a specific sample before running heavy computation:
```bash
crossstainwsi plan 4W-5-3 --base-dir /path/to/wsi --tiff-dir /path/to/crops
```

### 3. Run Single Sample
Execute registration and extract publication-ready crops:
```bash
crossstainwsi run 4W-5-3 --base-dir /path/to/wsi --out-dir /path/to/output
```

### 4. Batch Processing
Run multiple samples with automatic checkpointing and no automatic shutdown:
```bash
crossstainwsi batch 4W-5-3 2-2W-1 3-4W-2 --out-dir /path/to/output
```

---

## Python API Usage

```python
from pathlib import Path
from crossstainwsi.pipeline import PipelineConfig, SampleRunner, BatchRunner
from crossstainwsi.planning import UserGoal, ViewSpec, StainRequirement

# 1. Define custom goals
goal = UserGoal(
    reference_stain="masson",
    stain_requirements=[
        StainRequirement("HE", is_required=True),
        StainRequirement("Gram", is_required=False),
    ],
    requested_views=[
        ViewSpec(name="4x", pixel_dimensions=(2257, 1310), magnification_approx=4.0),
        ViewSpec(name="20x", pixel_dimensions=(2257, 1310), magnification_approx=20.0),
    ],
)

# 2. Execute Sample Runner
cfg = PipelineConfig(
    base_dir=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"),
    output_dir=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"),
)
runner = SampleRunner(config=cfg, goal=goal)
report = runner.process("4W-5-3")

print(f"Overall Status: {report['overall_status']}")
print(f"Artifact Tier:  {report['artifact_tier']}")
```

---

## Output Artifact Structure

Outputs are organized strictly by verification status:

```text
output_dir/
└── 4W-5-3/
    └── final/                   # Only generated when ALL QC checks PASS
        ├── 4W-5-3-Masson-4x-300dpi.tif
        ├── 4W-5-3-Masson-20x-300dpi.tif
        ├── 4W-5-3-HE-4x-aligned-300dpi.tif
        ├── 4W-5-3-HE-20x-aligned-300dpi.tif
        ├── overlay-HE-4x-aligned.png
        ├── overlay-HE-20x-aligned.png
        ├── contact_sheet_4x.png
        ├── contact_sheet_20x.png
        └── registration_report.json
```

---

## Testing

Run unit and integration test suites:
```bash
pytest -v
```

---

## License

MIT License.
