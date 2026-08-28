"""
局部形态学高精微调器 (LocalRefiner)
执行局部 LoFTR 特征吸合 -> 相位相关回退 -> 恒等保底
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.matching.phase_correlation import PhaseCorrelationMatcher
from crossstainwsi.transforms.geom import affine, h


@dataclass
class LocalRefineResult:
    aligned_image_bgr: np.ndarray
    mat_local_3x3: np.ndarray
    metrics: QCMetrics
    method: str
    details: Dict[str, Any]


class LocalRefiner:
    """
    负责在 4x 初步提取图像与参考图像之间求解局部微小形变与对齐残差
    """
    def __init__(
        self,
        loftr_matcher: Optional[LoFTRMatcher] = None,
        max_loftr_shift: float = 60.0,
        min_loftr_inliers: int = 8,
        min_scale: float = 0.96,
        max_scale: float = 1.04,
    ):
        self.loftr = loftr_matcher or LoFTRMatcher()
        self.phase_matcher = PhaseCorrelationMatcher(max_displacement=60.0, min_response=0.05)
        self.max_loftr_shift = max_loftr_shift
        self.min_loftr_inliers = min_loftr_inliers
        self.min_scale = min_scale
        self.max_scale = max_scale

    def refine(
        self,
        moving_crop_bgr: np.ndarray,
        fixed_ref_bgr: np.ndarray,
    ) -> LocalRefineResult:
        h_f, w_f = fixed_ref_bgr.shape[:2]

        # 1. 优先尝试局部 LoFTR 深度特征匹配
        loftr_res = self.loftr.match(moving_crop_bgr, fixed_ref_bgr)
        if loftr_res.is_valid and loftr_res.matrix is not None:
            m = loftr_res.metrics
            mat = loftr_res.matrix
            dx = float(mat[0, 2])
            dy = float(mat[1, 2])

            if (
                m.inliers >= self.min_loftr_inliers
                and self.min_scale <= m.scale <= self.max_scale
                and abs(dx) <= self.max_loftr_shift
                and abs(dy) <= self.max_loftr_shift
            ):
                mat_3x3 = h(mat)
                aligned = cv2.warpAffine(
                    moving_crop_bgr,
                    affine(mat_3x3),
                    (w_f, h_f),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
                return LocalRefineResult(
                    aligned_image_bgr=aligned,
                    mat_local_3x3=mat_3x3,
                    metrics=m,
                    method="Local_LoFTR",
                    details={"dx": dx, "dy": dy, "scale": m.scale, "inliers": m.inliers},
                )

        # 2. 回退到 Sobel 梯度相位相关平移对齐
        pc_res = self.phase_matcher.match(moving_crop_bgr, fixed_ref_bgr)
        if pc_res.is_valid and pc_res.matrix is not None:
            mat_3x3 = pc_res.matrix
            aligned = cv2.warpAffine(
                moving_crop_bgr,
                affine(mat_3x3),
                (w_f, h_f),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            return LocalRefineResult(
                aligned_image_bgr=aligned,
                mat_local_3x3=mat_3x3,
                metrics=pc_res.metrics,
                method="Phase_Correlation",
                details=pc_res.details,
            )

        # 3. 恒等保底 (保持初始全局变换结果不变)
        identity_3x3 = np.eye(3, dtype=np.float64)
        return LocalRefineResult(
            aligned_image_bgr=moving_crop_bgr.copy(),
            mat_local_3x3=identity_3x3,
            metrics=QCMetrics(scale=1.0, method="Identity_Fallback"),
            method="Identity_Fallback",
            details={"reason": "No local refinement converged"},
        )
