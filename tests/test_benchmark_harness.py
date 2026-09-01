"""
基准测试套件单元测试 (tests/test_benchmark_harness.py)
涵盖正例精度 (Conditional TRE)、负例安全拒识 (Correct Abstain) 与假阳性接受率 (True FAR)
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
        expected_matchable=True,
    )

    assert res.is_success is True
    assert res.is_unsafe_accept is False
    assert res.is_false_accept is False
    assert np.isclose(res.translation_error_px, 2.0, atol=1e-3)
    assert np.isclose(res.angle_error_deg, 0.0, atol=1e-3)


def test_benchmark_evaluator_negative_cases():
    """
    测试负例评测逻辑：
    1. 负例被拒绝 (ABSTAIN) -> is_correct_abstain = True, is_false_accept = False
    2. 负例被误判为通过 (PASS) -> is_false_accept = True
    """
    mat_gt = np.eye(3)

    # 情况 1: 负例被安全拒绝
    res_neg_abstain = BenchmarkEvaluator.evaluate_case(
        case_id="neg_01",
        mat_estimated_3x3=None,
        mat_gt_3x3=mat_gt,
        image_shape=(100, 100),
        status=RegistrationStatus.ABSTAIN,
        failure_code=FailureCode.LOW_INFORMATION,
        expected_matchable=False,
    )
    assert res_neg_abstain.is_correct_abstain is True
    assert res_neg_abstain.is_false_accept is False

    # 情况 2: 负例被错误给 PASS (真正的 False Accept 漏洞)
    res_neg_pass = BenchmarkEvaluator.evaluate_case(
        case_id="neg_02",
        mat_estimated_3x3=np.eye(3),
        mat_gt_3x3=mat_gt,
        image_shape=(100, 100),
        status=RegistrationStatus.PASS,
        expected_matchable=False,
    )
    assert res_neg_pass.is_false_accept is True
    assert res_neg_pass.is_correct_abstain is False


def test_benchmark_harness_suite_execution_with_positives_and_negatives():
    img = np.full((150, 150, 3), 255, dtype=np.uint8)
    img[30:120, 30:120] = (50, 150, 50)

    cases = SyntheticPerturbationGenerator.generate_benchmark_suite(img, base_name="test_suite", include_negatives=True)
    assert len(cases) == 8  # 6 个正例 + 2 个负例

    # 模拟一个优秀的算法函数 (正例返回精确逆真值，负例安全 ABSTAIN)
    def smart_algo(mov, fix):
        for c in cases:
            if np.array_equal(c.image_perturbed, mov):
                if c.expected_matchable:
                    return c.matrix_inverse_gt_3x3, RegistrationStatus.PASS, FailureCode.NONE
                else:
                    return None, RegistrationStatus.ABSTAIN, FailureCode.LOW_INFORMATION
        return None, RegistrationStatus.ABSTAIN, FailureCode.FEATURE_MATCH_WEAK

    harness = BenchmarkHarness(success_tre_thresh_px=10.0)
    summary = harness.run_suite(cases, smart_algo)

    assert summary.total_cases == 8
    assert summary.matchable_cases == 6
    assert summary.unmatchable_cases == 2
    assert summary.success_rate == 1.0
    assert summary.unsafe_accept_count == 0
    assert summary.false_accept_count == 0
    assert summary.correct_abstain_rate == 1.0
    assert np.isclose(summary.conditional_median_tre_px, 0.0, atol=1e-2)
