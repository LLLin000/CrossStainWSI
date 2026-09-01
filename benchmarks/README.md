# CrossStainWSI Benchmark — Baseline v0.3.0 (Exploratory Hard-Negative, STRICT IoU=0)

> **Status:** Frozen baseline for all M5 ablation comparisons. Do not edit metrics retroactively.
> **Branch:** `feature/m0-m3-adapters` @ `FIXED hard-negative` (IoU=0, 300px, seed=42)
> **Date:** 2026-09-01
> **Bench code:** `src/crossstainwsi/benchmark/{generator,metrics,runner}.py` + `scripts/run_benchmarks.py` + `benchmarks/config_baseline_v0.3.0.yaml`
---

## 1. Experiment Protocol (Frozen)

### Dataset
- **Source:** `benchmarks/ihc/ihc_1.ome.tiff` (VALIS example, pyramidal OME-TIFF RGB)
- **ROI:** center crop `800×800 px` at Level 0, no additional stain normalization
- **Effective level:** L0-equivalent patch used as `image_original` for synthetic perturbations

### Positive Suite (Matchable)
- **n = 30** (large mode) / 6 (small smoke)
- **Generator:** `SyntheticPerturbationGenerator.generate(..., large=True, seed=42, n_large=30)`
- **Parameter grid (random sampling, deterministic seed=42):**
  - `rotation ∈ {0, ±10, ±30, ±60, ±90, 180} deg`
  - `translation ∈ {0, ±10, ±25, ±50, ±100} px` per axis
  - `scale ∈ {0.95, 0.98, 1.00, 1.02, 1.05}`
  - `parity ∈ {normal, mirror}` (mirror via `F_x = diag(-1,1) + [w-1,0]`)
- **Border:** constant `255` (white slide background)

### Negative Suite (Unmatchable, expected_matchable=False)
- **Easy negatives (n=2):**
  - `neg_blank` — constant `255` blank slide
- **Hard negatives — exploratory (n=3, STRICT IoU=0):**
  - `hard_neg_{j}` — two `300×300` crops from same `800×800` tissue patch, sampled with `IoU=0` and `center_distance > diagonal/2` (i.e. strictly non-overlapping wrong ROI), i.e. *same-tissue wrong ROI* (repetitive trabecular texture)
  - Implementation: `SyntheticPerturbationGenerator(..., large=True, crop=300, seed=42)` with overlap check `inter_area==0` and center distance gating.
  - **Note:** `n=3` is exploratory only; formal calibration requires `n ≥ 100` hard negatives sampled from larger WSI tissue region (not limited to single 800 patch).

### PASS Definition (QC Gates, Frozen)
| Backend | Gate |
|---|---|
| **Baseline SIFT** | `SiftMatcher(match) → is_valid && matrix != None` else `ABSTAIN/ FEATURE_MATCH_WEAK` |
| **Phase Correlation** | `PhaseCorrelationMatcher(max_displacement=80) → is_valid` (response ≥0.05) |
| **LoFTR** | `LoFTRMatcher(conf>0.38, RANSAC 8px) → is_valid && matrix != None` |
| **Fusion (SIFT-gated)** | `if LoFTR.inliers≥20 && inlier_ratio>0.15 && coverage>0.20 → LoFTR else Phase else ABSTAIN` |

### Metrics (Denominators Frozen)
- `Success` denominator = **matchable positives** (30)
- `Coverage` denominator = **matchable positives** (fraction with `matrix != None`)
- `Unsafe-PASS` = `positive && status==PASS && TRE > 35 px` / matchable
- `FAR_hard` = `hard_negative && status==PASS` / n_hard
- `FAR_easy` = `easy_negative && status==PASS` / n_easy
- `Correct Abstain` = `unmatchable && status==ABSTAIN/FAIL` / unmatchable
- `Conditional TRE (median / P90 / P95)` = **among estimated matchable cases** (`TRE < 900 px`), i.e. *TRE conditional on producing an estimate*, not conditional on success. Includes wrong estimates so tail risk is visible.
- `Mirror accuracy` = `(det_est<0) == (det_gt<0)` / matchable

---

## 2. Observed Results (Do Not Reinterpret as Claims)

### Synthetic Geometry Benchmark — Large Suite (30 pos / 3 hard / 2 easy) — STRICT non-overlap hard negatives

| Algorithm / Variant | n (pos/hard/easy) | Success | Coverage | Unsafe-PASS | FAR_hard (exploratory) | FAR_easy | Correct Abstain | cond. median TRE | cond. P90 TRE | cond. P95 TRE | Mirror Acc |
|---|:---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **Baseline SIFT** | 30 / 3 / 2 | **53.3%** (16/30) | 80.0% | **26.7%** (8/30) | **1/3 (33.3%, n=3)** | 0.0% | 0.59 px | 652.4 px | 660.4 px | 53.3% |
| **Phase Correlation** | 30 / 3 / 2 | 3.3% (1/30) | 3.3% | **0%** | **0/3** | 0.0% | 0.03 px | 0.0 px | 0.0 px | 3.3% |
| **LoFTR Matcher** | 30 / 3 / 2 | 23.3% (7/30) | **90.0%** | **76.7%** (23/30) | **3/3 (100%, n=3)** | 0.0% | 734.6 px | 823.5 px | 835.4 px | 53.3% |
| **Fusion (SIFT-gated → Phase)** | 30 / 3 / 2 | 20.0% (6/30) | 20.0% | **0%** | **0/3** | 0.0% | 0.26 px | 0.3 px | 0.3 px | 20.0% |

> **Small smoke suite (6/2) for reference — frozen baseline v0.3.0:** SIFT 66.7%/16.7% Unsafe, Phase 16.7%/0% Unsafe, LoFTR 33.3%/66.7% Unsafe, Fusion 33.3%/16.7% Unsafe — consistent direction, smaller sample noise ±16.7% per case.
### IHC Deconvolution Ablation (ihc_1 ↔ ihc_2, center 800×800, LoFTR only)

| Evidence | Inliers | Inlier Ratio | Coverage |
|---|---:|---:|---:|
| Raw grayscale | 212 | 0.65 | 0.62 |
| Ruifrok H↔H (H channel) | 153 | 0.63 | 0.62 |
| **Δ** | **-27.8%** | — | — |

> **Observed only.** Inlier count drop does not imply accuracy drop; same-marker serial IHC may favor raw. H↔H value is in `H&E ↔ IHC` cross-marker pairs.

### CyCIF Multi-Channel (single-channel L OME-TIFF, exploratory)

- **Channel selection:** `Channel_0` (single L channel, no DAPI metadata) — `NuclearChannelResolver` not exercised.
- **Phase residual:** `dx=-47.36 px, dy=-3.20 px, response=0.68`
- **LoFTR on fluorescence-like grayscale:** `Inliers=4336, ratio=1.00` — sanity check that pipeline runs; not a DAPI vs auxiliary ablation.

---

## 3. Interpretation (Separated from Observations)

- **Observed:** LoFTR coverage 90.0%, Unsafe-PASS 76.7%, FAR_hard 3/3 (100%, n=3, STRICT non-overlap), cond. P90 823 px.
  **Interpretation:** LoFTR behaves as a **high-recall hypothesis generator** in this stress suite and is **unsafe as a standalone PASS authority** under repetitive bone texture. Hard-negative FAR confirms this even with strict non-overlap.

- **Observed:** Phase coverage 3.3%, Unsafe 0%, FAR_hard 0/3, cond. median 0.03 px.
  **Interpretation:** Phase is a **high-precision verifier inside a small basin**, not a global matcher. Low conditional TRE reflects its tiny accepted subset.

- **Observed:** Fusion (SIFT-gated) reduces Unsafe 26.7% → 0% and FAR_hard 33.3% → 0% (strict non-overlap) at cost of coverage 80% → 20%.
  **Interpretation:** Gating trades coverage for safety; **hard-negative FAR now 0/3 after strict IoU fix** (vs 3/3 before) shows strict non-overlap definition matters. Still, `FAR_hard=0/3` is exploratory (n=3); formal `FAR_hard <1%` requires `n≥100` and M5 consensus must push coverage up at fixed safety, not just push Unsafe down.

- **Observed:** IHC H↔H inlier drop −27.8%.
  **Interpretation:** No accuracy claim possible without geometry (TRE / transform consistency). Future ablation: `raw | H-only | raw+H dual evidence` on `H&E ↔ IHC` pairs.
---

## 4. M5 Target (Frozen for Next Phase)

**Primary objective:** `FAR_hard < 1%` (on `n ≥ 100` same-tissue hard negatives) **and** `Unsafe-PASS < 1%`.

**Constraint:** maximize Coverage at that operating point.

**Required output:** Coverage–Unsafe/FAR frontier by sweeping consensus threshold, then select operating point — not a single gate value.

**Hypothesis Consensus Engine (planned):**
```
Hypothesis Hi = {rotation, translation, scale, parity}
Evidence votes: LoFTR + SIFT + Contour/Distance + Phase + NMI + Cross-scale + MPP prior
PASS iff independent sources reach consensus; LoFTR alone never sufficient.
```

---

## 5. Reproducibility

```bash
# Small smoke (6 pos / 2 neg)
python scripts/run_benchmarks.py

# Large stress suite (30 pos / 3 hard / 2 easy, seed=42, deterministic)
python -c "
from pathlib import Path; from PIL import Image; import numpy as np
from crossstainwsi.benchmark import SyntheticPerturbationGenerator, BenchmarkHarness
# ... see scripts/run_benchmarks.py
"
```

**Seeds:** `SyntheticPerturbationGenerator(..., seed=42)` uses `np.random.default_rng(seed)` for noise and large-suite sampling; case `case_id` encodes provenance.

**Version:** Baseline v0.3.0 exploratory hard-negative benchmark. Hard negatives to be expanded to `n ≥ 100` before threshold calibration.
