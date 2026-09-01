"""
基准测试套件单元测试 (tests/test_benchmark_harness.py)
"""

import numpy as np
import pytest

from crossstainwsi.benchmark.generator import SyntheticPerturbationGenerator
from crossstainwsi.benchmark.metrics import BenchmarkEvaluator
from crossstainwsi.benchmark.runner import BenchmarkHarness
from crossstainwsi.domain import FailureCode, RegistrationStatus


def test_synthetic_perturbation_generator():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[50:150, 50:150] = (100, 50, 200)

    case = SyntheticPerturbationGenerator.generate(
        image_bgr=img,
        angle_deg=30.0,
        dx_px=10.0,
        dy_px=-15.0,
        scale=1.05,
        is_mirrored=True,
        case_id="test_case",
    )

    assert case.case_id == "test_case"
    assert case.params.is_mirrored is True
    assert case.matrix_gt_3x3.shape == (3, 3)
    assert case.matrix_inverse_gt_3x3.shape == (3, 3)

    # 验证逆矩阵乘积为单位阵
    prod = case.matrix_gt_3x3 @ case.matrix_inverse_gt_3x3
    assert np.allclose(prod, np.eye(3), atol=1e-4)


def test_benchmark_evaluator_metrics():
    mat_gt = np.eye(3)
    # 模拟估算结果带有 2 像素平移误差
    mat_est = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    res = BenchmarkEvaluator.evaluate_case(
        case_id="case_01",
        mat_estimated_3x3=mat_est,
        mat_gt_3x3=mat_gt,
        image_shape=(100, 100),
        status=RegistrationStatus.PASS,
    )

    assert res.is_success is True
    assert res.is_false_accept is False
    assert np.isclose(res.translation_error_px, 2.0, atol=1e-3)
    assert np.isclose(res.angle_error_deg, 0.0, atol=1e-3)


def test_benchmark_harness_suite_execution():
    img = np.full((150, 150, 3), 255, dtype=np.uint8)
    img[30:120, 30:120] = (50, 150, 50)

    cases = SyntheticPerturbationGenerator.generate_benchmark_suite(img, base_name="test_suite")
    assert len(cases) >= 5

    # 模拟一个返回精确真值的算法函数
    def perfect_algo(mov, fix):
        # 寻找对应的 case
        for c in cases:
            if np.array_equal(c.image_perturbed, mov):
                return c.matrix_inverse_gt_3x3, RegistrationStatus.PASS, FailureCode.NONE
        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    harness = BenchmarkHarness(success_tre_thresh_px=10.0)
    summary = harness.run_suite(cases, perfect_algo)

    assert summary.total_cases == len(cases)
    assert summary.success_rate == 1.0
    assert summary.false_accept_count == 0
    assert np.isclose(summary.median_tre_px, 0.0, atol=1e-2)
