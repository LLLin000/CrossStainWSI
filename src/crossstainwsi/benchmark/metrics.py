"""
基准测试评测指标与安全评估器 (Benchmark Metrics & Tail-Risk Evaluator)
支持条件 P90/P95 尾部误差 (Conditional TRE)、不安全接受率 (Unsafe-PASS) 与真实拒识假阳性率 (True FAR)
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from crossstainwsi.domain import FailureCode, RegistrationStatus
from crossstainwsi.transforms.geom import affine, apply_mat, extract_scale_and_angle, h


@dataclass
class CaseEvaluationResult:
    """单条基准测试评估记录"""
    case_id: str
    status: RegistrationStatus
    failure_code: FailureCode
    expected_matchable: bool            # 是否为理论可配准的正例
    angle_error_deg: float
    translation_error_px: float
    scale_error: float
    corner_tre_px: float
    mirror_parity_correct: bool
    is_success: bool                    # 正例且成功 (TRE <= 门限 且 状态 != ABSTAIN)
    is_unsafe_accept: bool              # 正例但配错仍给 PASS (Wrong-but-PASS, TRE > 门限)
    is_false_accept: bool               # 负例 (本不可配) 却给 PASS (True False Accept!)
    is_correct_abstain: bool            # 负例且正确给出 ABSTAIN/FAIL (安全拒识)
    is_missed_match: bool               # 正例但被误杀拒绝 (Missed Match / False Reject)


@dataclass
class BenchmarkSummary:
    """全批次/基准套件聚合汇总"""
    total_cases: int
    matchable_cases: int
    unmatchable_cases: int
    success_count: int
    success_rate: float                 # 正例成功率: success_count / matchable_cases
    coverage_rate: float                # 总体有效估计覆盖率 (非拒识率)
    abstain_rate: float
    unsafe_accept_count: int
    unsafe_accept_rate: float           # 错配但给 PASS 率 (Wrong-but-PASS Rate)
    false_accept_count: int
    false_accept_rate: float            # 真实不可配负例的假阳性接受率 (True FAR)
    correct_abstain_count: int
    correct_abstain_rate: float         # 真实负例的安全拒识率
    conditional_median_tre_px: float    # 仅在成功估计用例上的中位数 TRE
    conditional_p90_tre_px: float       # 90th percentile TRE (ACROBAT 核心评价指标)
    conditional_p95_tre_px: float       # 95th percentile TRE
    median_angle_error_deg: float
    median_translation_error_px: float
    mirror_accuracy: float
    case_results: List[CaseEvaluationResult] = field(default_factory=list)


class BenchmarkEvaluator:
    """
    负责对比算法估计矩阵与真实几何真值
    """
    @staticmethod
    def evaluate_case(
        case_id: str,
        mat_estimated_3x3: Optional[np.ndarray],
        mat_gt_3x3: np.ndarray,
        image_shape: Tuple[int, int],
        status: RegistrationStatus = RegistrationStatus.PASS,
        failure_code: FailureCode = FailureCode.NONE,
        expected_matchable: bool = True,
        success_tre_thresh_px: float = 15.0,
        unsafe_tre_thresh_px: float = 35.0,
    ) -> CaseEvaluationResult:
        h_img, w_img = image_shape

        corners = np.array([
            [0.0, 0.0],
            [float(w_img), 0.0],
            [0.0, float(h_img)],
            [float(w_img), float(h_img)],
        ], dtype=np.float32)

        pts_gt = apply_mat(affine(mat_gt_3x3), corners)

        # 处理拒识情况 (ABSTAIN 或估计失败)
        if mat_estimated_3x3 is None or status in (RegistrationStatus.FAIL, RegistrationStatus.ABSTAIN):
            is_correct_abs = (not expected_matchable)
            is_missed = expected_matchable
            return CaseEvaluationResult(
                case_id=case_id,
                status=status,
                failure_code=failure_code,
                expected_matchable=expected_matchable,
                angle_error_deg=180.0,
                translation_error_px=999.0,
                scale_error=1.0,
                corner_tre_px=999.0,
                mirror_parity_correct=False,
                is_success=False,
                is_unsafe_accept=False,
                is_false_accept=False,
                is_correct_abstain=is_correct_abs,
                is_missed_match=is_missed,
            )

        pts_est = apply_mat(affine(mat_estimated_3x3), corners)

        # 1. 四角 Target Registration Error (TRE)
        corner_errors = np.linalg.norm(pts_est - pts_gt, axis=1)
        corner_tre = float(np.mean(corner_errors))

        # 2. 尺度与角度分解误差
        scale_est, ang_est = extract_scale_and_angle(mat_estimated_3x3)
        scale_gt, ang_gt = extract_scale_and_angle(mat_gt_3x3)

        scale_err = abs(scale_est - scale_gt)
        ang_diff = (ang_est - ang_gt) % 360.0
        if ang_diff > 180.0:
            ang_diff = 360.0 - ang_diff
        angle_err = abs(ang_diff)

        # 3. 平移中心误差
        c_est = apply_mat(affine(mat_estimated_3x3), np.array([[w_img / 2.0, h_img / 2.0]], dtype=np.float32))[0]
        c_gt = apply_mat(affine(mat_gt_3x3), np.array([[w_img / 2.0, h_img / 2.0]], dtype=np.float32))[0]
        trans_err = float(np.linalg.norm(c_est - c_gt))

        # 4. 镜像奇偶性判定
        det_est = np.linalg.det(mat_estimated_3x3[:2, :2])
        det_gt = np.linalg.det(mat_gt_3x3[:2, :2])
        mirror_correct = bool((det_est < 0) == (det_gt < 0))

        if expected_matchable:
            is_success = (corner_tre <= success_tre_thresh_px) and (status != RegistrationStatus.ABSTAIN)
            is_unsafe_accept = (status == RegistrationStatus.PASS) and (corner_tre > unsafe_tre_thresh_px)
            is_false_accept = False
            is_correct_abs = False
            is_missed = False
        else:
            # 负例场景
            is_success = False
            is_unsafe_accept = False
            is_false_accept = (status == RegistrationStatus.PASS)
            is_correct_abs = (status in (RegistrationStatus.ABSTAIN, RegistrationStatus.FAIL))
            is_missed = False

        return CaseEvaluationResult(
            case_id=case_id,
            status=status,
            failure_code=failure_code,
            expected_matchable=expected_matchable,
            angle_error_deg=angle_err,
            translation_error_px=trans_err,
            scale_error=scale_err,
            corner_tre_px=corner_tre,
            mirror_parity_correct=mirror_correct,
            is_success=is_success,
            is_unsafe_accept=is_unsafe_accept,
            is_false_accept=is_false_accept,
            is_correct_abstain=is_correct_abs,
            is_missed_match=is_missed,
        )

    @classmethod
    def aggregate_benchmark(
        cls,
        case_results: List[CaseEvaluationResult],
    ) -> BenchmarkSummary:
        total = len(case_results)
        if total == 0:
            return BenchmarkSummary(
                total_cases=0, matchable_cases=0, unmatchable_cases=0, success_count=0,
                success_rate=0.0, coverage_rate=0.0, abstain_rate=0.0,
                unsafe_accept_count=0, unsafe_accept_rate=0.0, false_accept_count=0,
                false_accept_rate=0.0, correct_abstain_count=0, correct_abstain_rate=0.0,
                conditional_median_tre_px=0.0, conditional_p90_tre_px=0.0,
                conditional_p95_tre_px=0.0, median_angle_error_deg=0.0,
                median_translation_error_px=0.0, mirror_accuracy=0.0,
            )

        pos_cases = [c for c in case_results if c.expected_matchable]
        neg_cases = [c for c in case_results if not c.expected_matchable]

        success_count = sum(1 for c in pos_cases if c.is_success)
        unsafe_accept_count = sum(1 for c in pos_cases if c.is_unsafe_accept)
        false_accept_count = sum(1 for c in neg_cases if c.is_false_accept)
        correct_abstain_count = sum(1 for c in neg_cases if c.is_correct_abstain)
        abstain_count = sum(1 for c in case_results if c.status in (RegistrationStatus.ABSTAIN, RegistrationStatus.FAIL))
        mirror_correct_count = sum(1 for c in pos_cases if c.mirror_parity_correct)

        # 仅在估计成功的正例上计算条件 TRE 统计量 (Conditional TRE)
        valid_pos_tres = [c.corner_tre_px for c in pos_cases if c.corner_tre_px < 900.0]

        if valid_pos_tres:
            med_tre = float(np.median(valid_pos_tres))
            p90_tre = float(np.percentile(valid_pos_tres, 90))
            p95_tre = float(np.percentile(valid_pos_tres, 95))
            med_ang = float(np.median([c.angle_error_deg for c in pos_cases if c.angle_error_deg < 170.0] or [0.0]))
            med_trans = float(np.median([c.translation_error_px for c in pos_cases if c.translation_error_px < 900.0] or [0.0]))
        else:
            med_tre = 999.0
            p90_tre = 999.0
            p95_tre = 999.0
            med_ang = 180.0
            med_trans = 999.0

        n_pos = len(pos_cases)
        n_neg = len(neg_cases)

        return BenchmarkSummary(
            total_cases=total,
            matchable_cases=n_pos,
            unmatchable_cases=n_neg,
            success_count=success_count,
            success_rate=float(success_count / n_pos) if n_pos > 0 else 0.0,
            coverage_rate=float(len(valid_pos_tres) / n_pos) if n_pos > 0 else 0.0,
            abstain_rate=float(abstain_count / total),
            unsafe_accept_count=unsafe_accept_count,
            unsafe_accept_rate=float(unsafe_accept_count / n_pos) if n_pos > 0 else 0.0,
            false_accept_count=false_accept_count,
            false_accept_rate=float(false_accept_count / n_neg) if n_neg > 0 else 0.0,
            correct_abstain_count=correct_abstain_count,
            correct_abstain_rate=float(correct_abstain_count / n_neg) if n_neg > 0 else 0.0,
            conditional_median_tre_px=med_tre,
            conditional_p90_tre_px=p90_tre,
            conditional_p95_tre_px=p95_tre,
            median_angle_error_deg=med_ang,
            median_translation_error_px=med_trans,
            mirror_accuracy=float(mirror_correct_count / n_pos) if n_pos > 0 else 0.0,
            case_results=case_results,
        )
