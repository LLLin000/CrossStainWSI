# CrossStainWSI

[English](README.md) | [简体中文](README_zh.md)

**CrossStainWSI** is an auditable, adaptive cross-stain whole-slide image (WSI) registration and multi-scale region extraction toolkit for digital pathology and biomedical research.

---

## Key Features

- **Decoupled Inputs & Outputs**: Separates *Input Evidence* (WSI files, optional historical screenshots, native Level-0 coordinates) from *Output Requirements* (publication-grade 300 DPI crops, overlays, custom physical FOVs).
- **Adaptive Workflow Planner**: Automatically inspects available materials and formulates an `ExecutionPlan` tailored to your goal.
- **Deep Morphology Alignment**: Combines multi-angle LoFTR deep feature matching, connected tissue island isolation, and bounded local residual refinement (Local LoFTR + Sobel phase correlation).
- **Single-Pass Level-0 Sampling**: Direct inverse warp sampling (`cv2.warpAffine` + `WARP_INVERSE_MAP`) straight from WSI Level 0, completely avoiding cumulative rotation blur, artificial white corners, and shear deformations.
- **Strict QC & Safety Gating**: Segregates outputs into `final/` (PASS), `review/` (WARN / manual review), and `debug/` (ABSTAIN / INCOMPLETE). Never generates misleading publication TIFFs when alignment evidence is weak.

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

## Command Line Interface (CLI)

```bash
# 1. Discover Assets
crossstainwsi discover --base-dir /path/to/wsi --tiff-dir /path/to/crops

# 2. Inspect Execution Plan
crossstainwsi plan 4W-5-3 --base-dir /path/to/wsi --tiff-dir /path/to/crops

# 3. Run Single Sample (Default parameters)
crossstainwsi run 4W-5-3

# Custom parameters: specify reference stain, target stains, and 600 DPI
crossstainwsi run 4W-5-3 --ref-stain HE --stains HE Gram --dpi 600

# Horizontal mirror correction
crossstainwsi run 2-2W-1 --mirror

# 4. Batch Processing
crossstainwsi batch 4W-5-3 2-2W-1 3-4W-2 --out-dir /path/to/output
```

### CLI Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--ref-stain` | `masson` | Reference stain name (the stain used for manual crop capture) |
| `--stains` | `HE Gram` | Target stains to register and extract |
| `--dpi` | `300` | Output image DPI (default: 300) |
| `--scale-ratio` | `5.0` | Sampling scale ratio between 20x and 4x |
| `--mirror` | `False` | Force horizontal mirror correction for input crops |
| `--no-overlay` | `False` | Disable generating overlay comparison images |
| `--contact-sheet`| `False` | Enable generating side-by-side contact sheets (default: False) |

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
        └── registration_report.json
```

---

## Testing

```bash
pytest -v
```

---

## License

MIT License.
