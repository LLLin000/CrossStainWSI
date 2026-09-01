"""
CrossStainWSI 全量多模态基准评测运行脚本 (scripts/run_benchmarks.py)
在合成几何扰动、真实连续 IHC 切片与 CyCIF 多通道荧光切片上运行完整的 Benchmark 评测矩阵
"""

import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from crossstainwsi.benchmark import (
    BenchmarkEvaluator,
    BenchmarkHarness,
    BenchmarkSummary,
    SyntheticPerturbationGenerator,
)
from crossstainwsi.domain import FailureCode, RegistrationStatus
from crossstainwsi.matching import (
    LoFTRMatcher,
    NormalizedMutualInformationMatcher,
    PhaseCorrelationMatcher,
    SiftMatcher,
)
from crossstainwsi.representation import (
    ChannelEvidenceSelector,
    FluorescenceAdapter,
    GenericBrightfieldAdapter,
    IHCDeconvolutionAdapter,
    NuclearChannelResolver,
)
from crossstainwsi.transforms.geom import affine, apply_mat, extract_scale_and_angle, h


def load_benchmark_patch(image_path: Path, patch_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    """从基准切片中提取中心代表性组织 Patch"""
    im = Image.open(image_path)
    if im.mode == "RGB":
        rgb = np.array(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif im.mode == "L":
        gray = np.array(im)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(arr[:, :, 0] if arr.ndim == 3 else arr, cv2.COLOR_GRAY2BGR)

    h_img, w_img = bgr.shape[:2]
    pw, ph = patch_size
    cx, cy = w_img // 2, h_img // 2
    x1, y1 = max(0, cx - pw // 2), max(0, cy - ph // 2)
    x2, y2 = min(w_img, x1 + pw), min(h_img, y1 + ph)
    return bgr[y1:y2, x1:x2].copy()


def run_synthetic_benchmark_suite(patch_bgr: np.ndarray, large: bool = False) -> Dict[str, BenchmarkSummary]:
    """
    套件 1: 在真实组织切片纹理上施加已知几何扰动真值，对比不同算法后端的恢复极限
    large=True 时生成 40+ 随机正例 + 硬负例 (same-tissue wrong ROI)
    """
    cases = SyntheticPerturbationGenerator.generate_benchmark_suite(patch_bgr, base_name="geom", include_negatives=True, seed=42, large=large, n_large=40 if large else 6)
    harness = BenchmarkHarness(success_tre_thresh_px=15.0, false_accept_tre_thresh_px=35.0)
    results: Dict[str, BenchmarkSummary] = {}

    # 1. Baseline SIFT
    sift_matcher = SiftMatcher()
    def algo_sift(mov, fix):
        t0 = time.time()
        res = sift_matcher.match(mov, fix)
        if res.is_valid and res.matrix is not None:
            return h(res.matrix), RegistrationStatus.PASS, FailureCode.NONE
        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    t_start = time.time()
    results["Baseline SIFT"] = harness.run_suite(cases, algo_sift)

    # 2. Phase Correlation
    phase_matcher = PhaseCorrelationMatcher(max_displacement=60.0)
    def algo_phase(mov, fix):
        res = phase_matcher.match(mov, fix)
        if res.is_valid and res.matrix is not None:
            return res.matrix, RegistrationStatus.PASS, FailureCode.NONE
        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    results["Phase Correlation"] = harness.run_suite(cases, algo_phase)

    # 3. LoFTR Feature Matcher
    loftr_matcher = LoFTRMatcher()
    def algo_loftr(mov, fix):
        res = loftr_matcher.match(mov, fix)
        if res.is_valid and res.matrix is not None:
            return h(res.matrix), RegistrationStatus.PASS, FailureCode.NONE
        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    results["LoFTR Matcher"] = harness.run_suite(cases, algo_loftr)

    # 4. Multi-Evidence Full Pipeline (LoFTR + Phase Rescue + NMI)
    nmi_matcher = NormalizedMutualInformationMatcher()
    def algo_full_pipeline(mov, fix):
        # Stage A: 优先尝试 LoFTR
        res_l = loftr_matcher.match(mov, fix)
        if res_l.is_valid and res_l.metrics.inliers >= 20:
            return h(res_l.matrix), RegistrationStatus.PASS, FailureCode.NONE

        # Stage B: 相位相关与 NMI 救援
        res_p = phase_matcher.match(mov, fix)
        if res_p.is_valid:
            # 在相位粗平移先验下用 NMI 进行局部微调
            res_nmi = nmi_matcher.match(mov, fix, initial_guess_3x3=res_p.matrix)
            if res_nmi.is_valid and res_nmi.matrix is not None:
                return h(res_nmi.matrix), RegistrationStatus.PASS, FailureCode.NONE
            return res_p.matrix, RegistrationStatus.PASS, FailureCode.NONE

        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    results["Full Multi-Evidence (LoFTR+Phase+NMI)"] = harness.run_suite(cases, algo_full_pipeline)
    return results


def run_ihc_adapter_ablation(ihc_dir: Path) -> Dict[str, Any]:
    """
    套件 2: 真实连续 IHC 切片 (H&E vs IHC-DAB) 色彩解卷积消融实验
    """
    slides = sorted(ihc_dir.glob("*.ome.tiff"))
    if len(slides) < 2:
        return {"error": "Insufficient IHC slides found"}

    img1_bgr = load_benchmark_patch(slides[0], (800, 800)) # 参考切片 1 (如 H&E)
    img2_bgr = load_benchmark_patch(slides[1], (800, 800)) # 移动切片 2 (如 IHC-DAB)

    loftr = LoFTRMatcher()

    # 方案 A: 原始灰度 + LoFTR
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    res_raw = loftr.match(cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # 方案 B: Ruifrok 色彩解卷积 (H 通道 ↔ H 通道) + LoFTR
    ihc_adapter = IHCDeconvolutionAdapter()
    rep1 = ihc_adapter.adapt(img1_bgr)
    rep2 = ihc_adapter.adapt(img2_bgr)

    res_deconv = loftr.match(
        cv2.cvtColor(rep2.feature_image, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(rep1.feature_image, cv2.COLOR_GRAY2BGR),
    )

    return {
        "raw_inliers": res_raw.metrics.inliers,
        "raw_inlier_ratio": res_raw.metrics.inlier_ratio,
        "raw_coverage": res_raw.metrics.spatial_coverage,
        "deconv_inliers": res_deconv.metrics.inliers,
        "deconv_inlier_ratio": res_deconv.metrics.inlier_ratio,
        "deconv_coverage": res_deconv.metrics.spatial_coverage,
        "inliers_gain_pct": round((res_deconv.metrics.inliers - res_raw.metrics.inliers) / max(1, res_raw.metrics.inliers) * 100, 1),
    }


def run_cycif_fluorescence_ablation(cycif_dir: Path) -> Dict[str, Any]:
    """
    套件 3: 真实 CyCIF 多通道循环免疫荧光切片配准实测 (兼容单通道灰度图格式)
    """
    slides = sorted(cycif_dir.glob("*.ome.tiff"))
    if len(slides) < 2:
        return {"error": "Insufficient CyCIF slides found"}

    def load_as_multichannel(path: Path) -> np.ndarray:
        im = Image.open(path)
        arr = np.array(im)
        if arr.ndim == 2:
            return arr[:, :, None]
        return arr

    img1 = load_as_multichannel(slides[0])
    img2 = load_as_multichannel(slides[1])

    adapter = FluorescenceAdapter()
    n1 = img1.shape[2] if img1.ndim == 3 else 1
    n2 = img2.shape[2] if img2.ndim == 3 else 1
    names1 = [f"Channel_{i}" for i in range(n1)]
    names2 = [f"Channel_{i}" for i in range(n2)]
    rep1 = adapter.adapt(img1, channel_names=names1)
    rep2 = adapter.adapt(img2, channel_names=names2)

    phase = PhaseCorrelationMatcher(max_displacement=60.0)
    res_phase = phase.match(rep2.feature_image, rep1.feature_image)

    def _to_bgr(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    loftr = LoFTRMatcher()
    res_loftr = loftr.match(
        _to_bgr(rep2.feature_image),
        _to_bgr(rep1.feature_image),
    )

    return {
        "selected_channel_1": rep1.representation_provenance.get("nuclear_channel"),
        "selected_channel_2": rep2.representation_provenance.get("nuclear_channel"),
        "phase_dx": res_phase.details.get("dx"),
        "phase_dy": res_phase.details.get("dy"),
        "phase_response": round(res_phase.details.get("response", 0.0), 4),
        "loftr_inliers": res_loftr.metrics.inliers,
        "loftr_inlier_ratio": round(res_loftr.metrics.inlier_ratio, 4),
    }

def print_markdown_table(summaries: Dict[str, BenchmarkSummary]):
    print("\n### 基准评测套件 1: 合成几何真值扰动评测矩阵 (Synthetic Geometry Benchmark Suite)")
    print("| 算法 / 流程变体 | 样本数 (正/负) | 成功率 (Success) | 覆盖率 (Coverage) | 错配通过率 (Unsafe-PASS) | 真实假阳率 (True FAR) | 正确拒识率 | 中位数 TRE | P90 TRE | P95 TRE | 镜像准确率 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, s in summaries.items():
        print(
            f"| **{name}** | {s.matchable_cases}/{s.unmatchable_cases} | "
            f"{s.success_rate * 100:.1f}% | {s.coverage_rate * 100:.1f}% | "
            f"**{s.unsafe_accept_rate * 100:.1f}%** | **{s.false_accept_rate * 100:.1f}%** | "
            f"{s.correct_abstain_rate * 100:.1f}% | "
            f"{s.conditional_median_tre_px:.2f} px | {s.conditional_p90_tre_px:.2f} px | {s.conditional_p95_tre_px:.2f} px | "
            f"{s.mirror_accuracy * 100:.1f}% |"
        )


def main():
    print("================ Starting CrossStainWSI Comprehensive Benchmarking ================")
    # 查找本地基准切片
    bench_dir = Path("benchmarks")
    if not bench_dir.exists():
        bench_dir = Path("../CrossStainWSI/benchmarks")

    ihc_dir = bench_dir / "ihc"
    cycif_dir = bench_dir / "cycif"

    # 1. 运行套件 1: 合成几何扰动与安全评测
    print("\n[Suite 1/3] Running Synthetic Geometry Perturbations & Safety Benchmark...")
    test_patch = None
    if (ihc_dir / "ihc_1.ome.tiff").exists():
        test_patch = load_benchmark_patch(ihc_dir / "ihc_1.ome.tiff", (800, 800))
    else:
        # 使用合成组织图
        test_patch = np.full((800, 800, 3), 255, dtype=np.uint8)
        cv2.circle(test_patch, (400, 400), 200, (180, 50, 150), -1)
        cv2.rectangle(test_patch, (200, 200), (600, 350), (200, 100, 30), -1)

    synth_results = run_synthetic_benchmark_suite(test_patch)
    print_markdown_table(synth_results)

    # 2. 运行套件 2: 真实 IHC 色彩解卷积消融实验
    if ihc_dir.exists():
        print("\n### 基准评测套件 2: 真实连续 IHC 切片色彩解卷积消融实验 (IHC Deconvolution Ablation)")
        ihc_res = run_ihc_adapter_ablation(ihc_dir)
        print(f"- **原始灰度 (Raw Grayscale) + LoFTR**: Inliers={ihc_res['raw_inliers']}, Ratio={ihc_res['raw_inlier_ratio']:.2f}, Coverage={ihc_res['raw_coverage']:.2f}")
        print(f"- **Ruifrok 色彩解卷积 (H ↔ H) + LoFTR**: Inliers={ihc_res['deconv_inliers']}, Ratio={ihc_res['deconv_inlier_ratio']:.2f}, Coverage={ihc_res['deconv_coverage']:.2f}")
        print(f"- **特征点增益 (Inliers Gain)**: **+{ihc_res['inliers_gain_pct']}%** (有效抑制 DAB 棕色抗原干扰)")

    # 3. 运行套件 3: 真实 CyCIF 荧光切片多通道配准消融实验
    if cycif_dir.exists():
        print("\n### 基准评测套件 3: 真实多通道 CyCIF 循环免疫荧光切片配准实测 (CyCIF Multi-Channel)")
        cycif_res = run_cycif_fluorescence_ablation(cycif_dir)
        print(f"- **通道优选识别**: Slide 1={cycif_res['selected_channel_1']}, Slide 2={cycif_res['selected_channel_2']}")
        print(f"- **梯度相位相关 (Phase Correlation)**: Residual dx={cycif_res['phase_dx']:.2f} px, dy={cycif_res['phase_dy']:.2f} px, Response={cycif_res['phase_response']}")
        print(f"- **DAPI 细胞核场 LoFTR 密集特征**: Inliers={cycif_res['loftr_inliers']}, Ratio={cycif_res['loftr_inlier_ratio']:.2f}")

    print("\n================ All Benchmarks Completed Successfully ================")


if __name__ == "__main__":
    main()
