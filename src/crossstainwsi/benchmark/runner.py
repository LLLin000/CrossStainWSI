"""
基准测试执行套件 (Benchmark Runner)
自动批量运行几何扰动、消融实验与多指标报告汇总
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from crossstainwsi.benchmark.generator import PerturbationCase, SyntheticPerturbationGenerator
from crossstainwsi.benchmark.metrics import BenchmarkEvaluator, BenchmarkSummary, CaseEvaluationResult
from crossstainwsi.domain import FailureCode, RegistrationStatus


class BenchmarkHarness:
    """
    负责在基准测试用例集上自动化评估匹配算法与适配器的精度与安全性
    """
    def __init__(
        self,
        success_tre_thresh_px: float = 15.0,
        false_accept_tre_thresh_px: float = 35.0,
    ):
        self.success_tre_thresh_px = success_tre_thresh_px
        self.false_accept_tre_thresh_px = false_accept_tre_thresh_px

    def run_suite(
        self,
        cases: List[PerturbationCase],
        algorithm_fn: Callable[[np.ndarray, np.ndarray], Tuple[Optional[np.ndarray], RegistrationStatus, FailureCode]],
    ) -> BenchmarkSummary:
        """
        algorithm_fn: 输入 (moving_bgr, fixed_bgr) -> (mat_est_3x3, status, failure_code)
        """
        results: List[CaseEvaluationResult] = []

        for case in cases:
            # 真实原图为 fixed，扰动后的图像为 moving，求解 M(moving -> fixed)
            mat_est, status, fail_code = algorithm_fn(case.image_perturbed, case.image_original)

            # case.matrix_gt_3x3 描述的是 original -> perturbed
            # 所以 moving -> fixed 的真实几何真值为 matrix_inverse_gt_3x3!
            eval_res = BenchmarkEvaluator.evaluate_case(
                case_id=case.case_id,
                mat_estimated_3x3=mat_est,
                mat_gt_3x3=case.matrix_inverse_gt_3x3,
                image_shape=case.image_original.shape[:2],
                status=status,
                failure_code=fail_code,
                expected_matchable=case.expected_matchable,
                success_tre_thresh_px=self.success_tre_thresh_px,
                unsafe_tre_thresh_px=self.false_accept_tre_thresh_px,
            )
            results.append(eval_res)

        return BenchmarkEvaluator.aggregate_benchmark(results)
