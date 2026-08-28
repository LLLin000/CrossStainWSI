"""
基于 Sobel 梯度幅值的相位相关微平移残差估计器
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import translation_matrix


class PhaseCorrelationMatcher(ImageMatcher):
    """
    基于 Sobel 边缘梯度幅值的相位相关法，求解小范围平移残差 (dx, dy)
    """
    def __init__(self, max_displacement: float = 60.0, min_response: float = 0.05):
        self.max_displacement = max_displacement
        self.min_response = min_response

    def _sobel_magnitude(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag

    def match(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
    ) -> MatchResult:
        if moving_bgr.shape[:2] != fixed_bgr.shape[:2]:
            moving_aligned = cv2.resize(
                moving_bgr, (fixed_bgr.shape[1], fixed_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            moving_aligned = moving_bgr

        mag_f = self._sobel_magnitude(fixed_bgr)
        mag_m = self._sobel_magnitude(moving_aligned)

        (dx, dy), response = cv2.phaseCorrelate(mag_f, mag_m)

        # moving 需要平移 (-dx, -dy) 与 fixed 对齐
        shift_x = -float(dx)
        shift_y = -float(dy)

        is_valid = (
            abs(shift_x) <= self.max_displacement
            and abs(shift_y) <= self.max_displacement
            and response >= self.min_response
        )

        mat_3x3 = np.array([
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        metrics = QCMetrics(
            scale=1.0,
            rotation_deg=0.0,
            method="Phase_Correlation",
            details={"dx": shift_x, "dy": shift_y, "response": float(response)},
        )

        return MatchResult(
            matrix=mat_3x3 if is_valid else None,
            metrics=metrics,
            is_valid=is_valid,
            details={"dx": shift_x, "dy": shift_y, "response": float(response)},
        )
