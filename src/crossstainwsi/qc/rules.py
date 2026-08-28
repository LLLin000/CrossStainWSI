"""
质量控制 (QC) 规则引擎与状态判定
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from crossstainwsi.domain import QCMetrics, RegistrationStatus


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
    负责根据配准指标严格评估输出质量等级 (PASS, WARN, ABSTAIN, MANUAL_REVIEW, FAIL)
    禁止通过强行选择弱候选宣布虚假 PASS
    """
    def __init__(self, config: Optional[QCRuleConfig] = None):
        self.cfg = config or QCRuleConfig()

    def evaluate_reference_anchor(self, metrics: QCMetrics) -> Tuple[RegistrationStatus, str]:
        """
        评估 Masson 参考切片锚点定位质量
        """
        if metrics.inliers >= self.cfg.min_anchor_sift_inliers:
            return RegistrationStatus.PASS, f"High confidence SIFT anchor ({metrics.inliers} inliers)"

        if metrics.ncc_score is not None and metrics.ncc_score >= self.cfg.min_anchor_ncc:
            return RegistrationStatus.WARN, f"Template NCC anchor (NCC={metrics.ncc_score:.3f})"

        return RegistrationStatus.ABSTAIN, (
            f"Reference anchor ambiguous or rejected: inliers={metrics.inliers}, "
            f"ncc={metrics.ncc_score}"
        )

    def evaluate_cross_stain(self, metrics: QCMetrics) -> Tuple[RegistrationStatus, str]:
        """
        评估 HE / Gram 等跨染色配准输出质量
        """
        # 1. 尺度异常拦截
        if not (self.cfg.min_scale_warn <= metrics.scale <= self.cfg.max_scale_warn):
            return RegistrationStatus.ABSTAIN, (
                f"Abnormal scale deformation ({metrics.scale:.3f}) outside "
                f"[{self.cfg.min_scale_warn}, {self.cfg.max_scale_warn}]"
            )

        # 2. 内点极少直接放弃
        if metrics.inliers < self.cfg.min_global_inliers_warn:
            return RegistrationStatus.ABSTAIN, (
                f"Insufficient inliers ({metrics.inliers} < {self.cfg.min_global_inliers_warn})"
            )

        # 3. 黄金标准 PASS
        is_pass = (
            metrics.inliers >= self.cfg.min_global_inliers_pass
            and metrics.inlier_ratio >= self.cfg.min_inlier_ratio_pass
            and metrics.spatial_coverage >= self.cfg.min_spatial_coverage_pass
            and (self.cfg.min_scale_pass <= metrics.scale <= self.cfg.max_scale_pass)
        )

        if is_pass:
            return RegistrationStatus.PASS, (
                f"Strong alignment (Inliers={metrics.inliers}, Ratio={metrics.inlier_ratio:.2f}, "
                f"Coverage={metrics.spatial_coverage:.2f}, Scale={metrics.scale:.3f})"
            )

        # 4. 处于临界区给出 WARN / MANUAL_REVIEW
        return RegistrationStatus.WARN, (
            f"Marginal alignment (Inliers={metrics.inliers}, Ratio={metrics.inlier_ratio:.2f}, "
            f"Coverage={metrics.spatial_coverage:.2f}, Scale={metrics.scale:.3f}) -> Requires Review"
        )
