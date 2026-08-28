# CrossStainWSI — Agent Context

## Project identity

CrossStainWSI is a research-grade toolkit for registering serial histology whole-slide images across stains, mapping a manually selected ROI between coordinate spaces, sampling publication-resolution crops from the original WSI, and emitting auditable QC artifacts.

The target workflow is:

```text
DISCOVER → VALIDATE → LOAD METADATA → REFERENCE ROI LOCALIZATION
→ GLOBAL REGISTRATION → LOCAL REFINEMENT → QC
→ TRANSFORM GRAPH → LEVEL-0 SAMPLING → EXPORT → REPORT
```

This repository is the beginning of the architecture migration. It is not yet the finished engine.

## Repository state

- Local repository: `D:\L\AI\CrossStainWSI`
- Branch: `main`
- Git remote: not configured
- Commits: none yet
- Raw KFB/WSI files are intentionally not copied into this repository.
- The current behavior baseline is under `reference_impl/`.

Current files:

```text
CrossStainWSI/
├─ CONTEXT.md
├─ .gitignore
├─ pyproject.toml
├─ reference_impl/
│  ├─ register_slices.py
│  ├─ register_difficult_samples.py
│  └─ reference_anchor_selfcheck.py
├─ src/
│  └─ crossstainwsi/
│     └─ __init__.py
├─ tests/
├─ presets/
└─ runs/
   └─ .gitkeep
```

## Current reference data

The working data is outside this repository:

```text
E:\研究数据\骨科\切片扫描\
├─ 2026-08-21\
│  ├─ masson\
│  ├─ *-HE.kfb
│  └─ *-Gram.kfb
├─ tiff\
├─ registered_crops_300dpi\
└─ tiff_mirrored\
```

There are 29 manually selected sample IDs in `tiff/`, each with a `-4x.tif` and `-20x.tif` input. The extra sample added during the previous session was `3-4w-2`.

## Provenance and known acquisition facts

- KFB pyramid levels use downsamples approximately `(1, 2, 4, 8, 16)`.
- Reported level-0 MPP is approximately `0.4424304962 µm/px` for the current scans.
- The working publication crop canvas is commonly `2257 × 1310`, but the new `4-4w-1` input is `3107 × 1833`; do not assume one fixed canvas size in the engine.
- The current manual 4x/20x screenshots represent the same viewport at approximately a 5x physical sampling ratio. This is an acquisition/profile assumption and must be represented in configuration, not hidden in registration code.
- Some manual screenshots were horizontally mirrored during capture. For those samples, the correct operational input is a horizontally flipped copy before ordinary registration. The engine should treat the corrected image as an ordinary ROI; mirror handling belongs in input provenance/preprocessing, not in the transform model.

Known mirrored samples from the current data review:

- `2-2w-1`
- `5-4w-2`

`3-4w-2` was checked as non-mirrored and ran through the ordinary flow.

## Verified behavioral baseline

The original working script can produce correct results when the manual input orientation is correct:

- `2-2w-1`: horizontal flip of the manual 4x/20x input followed by the original flow produced strong Masson localization, HE registration, and Gram registration.
- `5-4w-2`: the same horizontal flip rule produced strong Masson localization, HE registration, and Gram registration.
- `3-4w-2`: no flip was needed; ordinary registration completed.
- `4-4w-1`: the manually recaptured Masson ROI is matchable; HE is acceptable despite a genuine serial-section difference. Gram requires conservative QC/manual review if its high-resolution evidence is weak.

The previous session also proved a failure pattern: HE and Gram can agree with each other while both disagree with Masson when the reference ROI is mirrored, ambiguously localized, or mapped to a symmetric tissue island. This is a common-mode reference-anchor error, not evidence that both moving stains are independently correct.

## Non-negotiable invariants

1. **Reference anchor before cross-stain registration**
   - First prove that the manual Masson ROI can be recovered from the Masson WSI.
   - Same-stain self-check must be stricter than cross-stain matching.
   - If the anchor fails or remains ambiguous, stop before HE/Gram and report `REFERENCE_ANCHOR_ABSTAIN`.

2. **Full-canvas evidence**
   - Use tissue mask, outer contour, internal structural edges, blank/background layout, scale, position, and rotation.
   - Background is negative evidence, not disposable padding.
   - Do not crop away the blank layout before anchor verification.

3. **Parent-child transforms**
   - Preserve complete transforms, not only coarse angle labels.
   - The intended relationship is:

     ```text
     T_crop→moving = T_Masson→moving · T_crop→Masson
     T_20 = T_4 · T_20→4
     ```

   - 20x may estimate only a bounded local residual after inheriting the 4x parent transform.

4. **Single-pass original sampling**
   - Compute transforms at low resolution.
   - Sample final 4x/20x images directly from the original WSI, preferably Level 0 for high-resolution output.
   - Do not rotate a finite intermediate patch and then crop it again; that creates artificial white corners and loses coordinates.

5. **No forced winner**
   - Top-K is a candidate set, not a requirement to select one.
   - Independent 20x evidence must be allowed to reject every 4x candidate.
   - Low evidence produces `ABSTAIN`/`MANUAL_REVIEW`, not a guessed TIFF marked `PASS`.

6. **Transform model names must be exact**
   - `RIGID`: rotation + translation.
   - `SIMILARITY`: rotation + translation + isotropic scale.
   - `AFFINE`: includes shear/an-isotropic scale.
   - Current publication mode should default to `SIMILARITY`; a rigid-only preset should be available.

## Architecture target

The intended architecture is a small public interface over deep internal modules:

```text
CLI / Python API
        ↓
Pipeline Engine
        ↓
Registration / ROI / QC
        ↓
Transform Graph
        ↓
Slide and Image I/O adapters
        ↓
KFB / SVS / NDPI / TIFF / OME-TIFF
```

Recommended package shape:

```text
src/crossstainwsi/
├─ domain/
│  ├─ slide.py          # SlideSpec, metadata provenance
│  ├─ roi.py            # physical ROI, view specifications
│  ├─ transform.py      # immutable transform records
│  ├─ registration.py   # result contracts
│  └─ qc.py             # QCResult and verdicts
├─ io/
│  ├─ base.py           # SlideReader interface
│  ├─ kfb.py            # KFB adapter
│  ├─ openslide.py      # generic OpenSlide adapter
│  └─ image.py          # TIFF/PNG crop adapter
├─ tissue/
│  └─ islands.py        # masks, contours, distance fields, descriptors
├─ matching/
│  ├─ sift.py
│  ├─ loftr.py
│  ├─ phase_correlation.py
│  └─ edge.py
├─ registration/
│  ├─ reference_anchor.py
│  ├─ global_registration.py
│  ├─ local_registration.py
│  └─ policies.py
├─ transforms/
│  ├─ models.py
│  └─ graph.py
├─ sampling/
│  └─ sampler.py
├─ qc/
│  ├─ metrics.py
│  ├─ rules.py
│  └─ evaluator.py
├─ pipeline/
│  ├─ sample_pipeline.py
│  └─ batch_pipeline.py
└─ reporting/
   ├─ report.py
   └─ contact_sheet.py
```

Do not create every module as an empty placeholder in one pass. Add each module when its contract and a real behavior test exist.

## Domain contracts to establish first

### SlideSpec

Must carry:

- `id`
- `sample_id`
- `stain`
- `path`
- `format`
- `dimensions`
- `level_dimensions`
- `level_downsamples`
- `mpp_x`, `mpp_y`
- metadata values and provenance

### ROI

A physical observation, not merely a TIFF:

- `id`
- `source_slide`
- `coordinate_space`
- `center`
- `width_um`, `height_um`
- `rotation`
- acquisition/view profile
- provenance

### Transform

Must record:

- source coordinate space
- target coordinate space
- model (`rigid`, `similarity`, etc.)
- matrix
- estimator
- parent transforms
- metrics
- software/model/config provenance

### RegistrationResult

Use structured results for expected registration failures:

- `status`: `PASS`, `WARN`, `ABSTAIN`, `FAIL`
- `reason`
- `transform`
- `metrics`
- candidate/hypothesis details

Reserve exceptions for corrupted inputs, unavailable readers, programmer errors, and unrecoverable runtime failures.

## Multi-scale policy

4x and 20x do different jobs:

- 4x: tissue identity, island selection, gross orientation, global geometry, context.
- 20x: local bone/marrow landmarks, fine edge structure, bounded residual translation/rotation, high-resolution confirmation.

Shared between scales:

- tissue island identity
- complete parent transform
- physical scale/MPP prior
- viewport relationship
- predicted center

Estimated independently but bounded:

- local translation
- small local rotation
- local evidence metrics

Candidate fusion must separate evidence streams. NCC must not veto LoFTR by being used as the latter's residual. Each method gets its own residual and support flag, followed by a fusion policy.

## Reference-anchor policy

Candidate retrieval may use broad thresholds. Verification must use the full-canvas same-stain round trip:

```text
manual Masson 4x/20x
        ↓
Top-K candidate locations in Masson WSI
        ↓
re-sample each candidate from the original WSI
        ↓
compare 4x and 20x against the manual inputs
        ↓
select only a unique, high-confidence anchor
```

Suggested evidence fields:

- same-stain 4x/20x NCC
- same-stain SIFT/feature inliers and ratio
- median reprojection error
- tissue mask IoU/Dice
- background agreement
- edge correlation
- candidate margin over the runner-up
- mirror/reflection state

A repeated/symmetric candidate with no unique evidence is not a valid anchor; return `REFERENCE_ANCHOR_AMBIGUOUS`.

## Current implementation caveats

The files in `reference_impl/` are copied from the working prototype and are intentionally not yet the final package:

- Paths are hardcoded to the original Windows data directory.
- The prototype mixes I/O, matching, sampling, QC, and reporting in one class.
- Some historical scripts generated false `PASS` results; use the newer strict-gate behavior as the intended direction.
- `register_slices.py` has experimental edits and should be treated as a behavior reference, not blindly imported into the new package.
- Do not copy raw WSI/KFB data into Git.
- Do not commit generated TIFF/PNG runs.
- Do not add a GUI before the headless pipeline contracts are stable.

## Recommended next sequence

1. Define `SlideSpec`, `ROI`, `Transform`, `TransformGraph`, and `RegistrationResult`.
2. Add a reader adapter that can read metadata and regions without exposing KFB details to registration code.
3. Port reference-anchor localization and same-stain self-check first.
4. Port global registration as a separate stage.
5. Port bounded local refinement and direct Level-0 sampling.
6. Move QC rules out of matcher implementations.
7. Add a manifest/preset loader; move all paths and acquisition assumptions into YAML.
8. Add batch state/resume and auditable run artifacts.
9. Add CLI only after one sample and one batch path are deterministic.
10. Compare new outputs against `reference_impl/` on fixed input fixtures before changing algorithms.

## Handoff instructions

The next agent should:

- read this file first;
- treat `reference_impl/` as a baseline to extract behavior from;
- preserve the invariants above;
- avoid adding dependencies until existing installed libraries are checked;
- add one executable check for each non-trivial transform or policy change;
- keep real KFB/WSI data outside the repository;
- report any mismatch between the reference prototype and the new engine with input path, coordinate spaces, matrices, and QC metrics.
