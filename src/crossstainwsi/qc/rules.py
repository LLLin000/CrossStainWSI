"""
质量控制 (QC) 规则引擎与状态判定 (支持细粒度 FailureCode 与物理残差缩放校准)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from crossstainwsi.domain import FailureCode, QCMetrics, RegistrationStatus


@dataclass
class QCRuleConfig:
    # 跨染色全局配准门限
    min_global_inliers_pass: int = 35
    min_global_inliers_warn: int = 15
    min_inlier_ratio_pass: float = 0.12
    min_spatial_coverage_pass: float = 0.20
    min_scale_pass: float = 0.95
    max_scale_pass: float = 1.05
    min_scale_warn: float = 0.88
    max_scale_warn: float = 1.12

    # 参考切片锚点自检门限
    min_anchor_sift_inliers: int = 15
    min_anchor_ncc: float = 0.30


class QCRuleEngine:
    """
    负责根据配准指标严格评估输出质量等级与细粒度失败归因代码
    """
    def __init__(self, config: Optional[QCRuleConfig] = None):
        self.cfg = config or QCRuleConfig()

    def evaluate_reference_anchor(self, metrics: QCMetrics) -> Tuple[RegistrationStatus, FailureCode, str]:
        """
        评估参考切片锚点定位质量
        """
        if metrics.inliers >= self.cfg.min_anchor_sift_inliers:
            return (
                RegistrationStatus.PASS,
                FailureCode.NONE,
                f"High confidence SIFT anchor ({metrics.inliers} inliers)",
            )

        if metrics.ncc_score is not None and metrics.ncc_score >= self.cfg.min_anchor_ncc:
            return (
                RegistrationStatus.WARN,
                FailureCode.NONE,
                f"Template NCC anchor (NCC={metrics.ncc_score:.3f})",
            )

        return (
            RegistrationStatus.ABSTAIN,
            FailureCode.REFERENCE_ANCHOR_FAIL,
            f"Reference anchor ambiguous or rejected: inliers={metrics.inliers}, ncc={metrics.ncc_score}",
        )

    def evaluate_cross_stain(
        self,
        metrics: QCMetrics,
        expected_scale_from_mpp: float = 1.0,
    ) -> Tuple[RegistrationStatus, FailureCode, str]:
        """
        评估跨染色配准输出质量
        expected_scale_from_mpp: 两个扫描仪 MPP 差异引起的预期物理尺度缩放 (mpp_fixed / mpp_moving)
        """
        # 归一化残差缩放尺度 (去除不同扫描仪物理像素大小差异)
        residual_scale = metrics.scale / max(1e-5, expected_scale_from_mpp)

        # 1. 尺度异常拦截
        if not (self.cfg.min_scale_warn <= residual_scale <= self.cfg.max_scale_warn):
            return (
                RegistrationStatus.ABSTAIN,
                FailureCode.STRUCTURE_CONFLICT,
                f"Abnormal residual scale ({residual_scale:.3f}) outside [{self.cfg.min_scale_warn}, {self.cfg.max_scale_warn}]",
            )

        # 2. 内点极少直接放弃
        if metrics.inliers < self.cfg.min_global_inliers_warn:
            return (
                RegistrationStatus.ABSTAIN,
                FailureCode.FEATURE_MATCH_WEAK,
                f"Insufficient inliers ({metrics.inliers} < {self.cfg.min_global_inliers_warn})",
            )

        # 3. 黄金标准 PASS
        is_pass = (
            metrics.inliers >= self.cfg.min_global_inliers_pass
            and metrics.inlier_ratio >= self.cfg.min_inlier_ratio_pass
            and metrics.spatial_coverage >= self.cfg.min_spatial_coverage_pass
            and (self.cfg.min_scale_pass <= residual_scale <= self.cfg.max_scale_pass)
        )

        if is_pass:
            return (
                RegistrationStatus.PASS,
                FailureCode.NONE,
                f"Strong alignment (Inliers={metrics.inliers}, Ratio={metrics.inlier_ratio:.2f}, "
                f"Coverage={metrics.spatial_coverage:.2f}, Scale={metrics.scale:.3f})",
            )

        # 4. 处于临界区给出 WARN / MANUAL_REVIEW
        return (
            RegistrationStatus.WARN,
            FailureCode.SECTION_CORRESPONDENCE_WEAK,
            f"Marginal alignment (Inliers={metrics.inliers}, Ratio={metrics.inlier_ratio:.2f}, "
            f"Coverage={metrics.spatial_coverage:.2f}, Scale={metrics.scale:.3f}) -> Requires Review",
        )
