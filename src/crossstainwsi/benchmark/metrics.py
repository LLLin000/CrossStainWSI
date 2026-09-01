"""
基准测试评测指标与安全评估器 (Benchmark Metrics & Tail-Risk Evaluator)
支持解析几何误差、角点 TRE、P90/P95 尾部误差以及假阳性接受率 (False Accept Rate / Wrong-but-PASS)
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
    angle_error_deg: float
    translation_error_px: float
    scale_error: float
    corner_tre_px: float
    mirror_parity_correct: bool
    is_success: bool                    # 角点 TRE < 门限 且 状态不为 ABSTAIN/FAIL
    is_false_accept: bool               # 状态为 PASS 但实际 TRE > 门限 (最危险的 Wrong-but-PASS!)
    is_correct_abstain: bool            # 真实无法配准时正确拒绝


@dataclass
class BenchmarkSummary:
    """全批次/基准套件聚合汇总"""
    total_cases: int
    success_count: int
    success_rate: float
    abstain_count: int
    abstain_rate: float
    false_accept_count: int
    false_accept_rate: float            # 假阳性接受率 (Wrong-but-PASS)
    median_tre_px: float
    p90_tre_px: float                   # 90th percentile TRE (ACROBAT 核心评价指标)
    p95_tre_px: float                   # 95th percentile TRE
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
        success_tre_thresh_px: float = 15.0,
        false_accept_tre_thresh_px: float = 35.0,
    ) -> CaseEvaluationResult:
        h_img, w_img = image_shape

        corners = np.array([
            [0.0, 0.0],
            [float(w_img), 0.0],
            [0.0, float(h_img)],
            [float(w_img), float(h_img)],
        ], dtype=np.float32)

        pts_gt = apply_mat(affine(mat_gt_3x3), corners)

        if mat_estimated_3x3 is None or status in (RegistrationStatus.FAIL, RegistrationStatus.ABSTAIN):
            return CaseEvaluationResult(
                case_id=case_id,
                status=status,
                failure_code=failure_code,
                angle_error_deg=180.0,
                translation_error_px=999.0,
                scale_error=1.0,
                corner_tre_px=999.0,
                mirror_parity_correct=False,
                is_success=False,
                is_false_accept=False,
                is_correct_abstain=True,
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

        is_success = (corner_tre <= success_tre_thresh_px) and (status != RegistrationStatus.ABSTAIN)
        is_false_accept = (status == RegistrationStatus.PASS) and (corner_tre > false_accept_tre_thresh_px)

        return CaseEvaluationResult(
            case_id=case_id,
            status=status,
            failure_code=failure_code,
            angle_error_deg=angle_err,
            translation_error_px=trans_err,
            scale_error=scale_err,
            corner_tre_px=corner_tre,
            mirror_parity_correct=mirror_correct,
            is_success=is_success,
            is_false_accept=is_false_accept,
            is_correct_abstain=False,
        )

    @classmethod
    def aggregate_benchmark(
        cls,
        case_results: List[CaseEvaluationResult],
    ) -> BenchmarkSummary:
        total = len(case_results)
        if total == 0:
            return BenchmarkSummary(
                total_cases=0, success_count=0, success_rate=0.0,
                abstain_count=0, abstain_rate=0.0, false_accept_count=0,
                false_accept_rate=0.0, median_tre_px=0.0, p90_tre_px=0.0,
                p95_tre_px=0.0, median_angle_error_deg=0.0,
                median_translation_error_px=0.0, mirror_accuracy=0.0,
            )

        success_count = sum(1 for c in case_results if c.is_success)
        abstain_count = sum(1 for c in case_results if c.status in (RegistrationStatus.ABSTAIN, RegistrationStatus.FAIL))
        false_accept_count = sum(1 for c in case_results if c.is_false_accept)
        mirror_correct_count = sum(1 for c in case_results if c.mirror_parity_correct)

        # 仅在估计成功的用例上计算 TRE 统计量 (避免未对齐的 999.0 严重扭曲中位数)
        valid_tres = [c.corner_tre_px for c in case_results if c.corner_tre_px < 900.0]

        if valid_tres:
            med_tre = float(np.median(valid_tres))
            p90_tre = float(np.percentile(valid_tres, 90))
            p95_tre = float(np.percentile(valid_tres, 95))
            med_ang = float(np.median([c.angle_error_deg for c in case_results if c.angle_error_deg < 170.0] or [0.0]))
            med_trans = float(np.median([c.translation_error_px for c in case_results if c.translation_error_px < 900.0] or [0.0]))
        else:
            med_tre = 999.0
            p90_tre = 999.0
            p95_tre = 999.0
            med_ang = 180.0
            med_trans = 999.0

        return BenchmarkSummary(
            total_cases=total,
            success_count=success_count,
            success_rate=float(success_count / total),
            abstain_count=abstain_count,
            abstain_rate=float(abstain_count / total),
            false_accept_count=false_accept_count,
            false_accept_rate=float(false_accept_count / total),
            median_tre_px=med_tre,
            p90_tre_px=p90_tre,
            p95_tre_px=p95_tre,
            median_angle_error_deg=med_ang,
            median_translation_error_px=med_trans,
            mirror_accuracy=float(mirror_correct_count / total),
            case_results=case_results,
        )
