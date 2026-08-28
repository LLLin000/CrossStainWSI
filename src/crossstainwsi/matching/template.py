"""
基于物理采样尺度与多角度旋转归一化互相关 (NCC) 模板匹配器
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import affine, h, rotation_matrix_2d, scale_matrix, translation_matrix


class TemplateMatcher(ImageMatcher):
    """
    基于先验物理尺度与多角度模板匹配
    """
    def __init__(
        self,
        physical_scale: float = 1.0,
        angle_range: Tuple[int, int] = (-180, 180),
        angle_step: int = 5,
        min_ncc: float = 0.30,
    ):
        self.physical_scale = physical_scale
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.min_ncc = min_ncc
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess_gray(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def match(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
    ) -> MatchResult:
        w_m, h_m = moving_bgr.shape[1], moving_bgr.shape[0]
        scaled_w = round(w_m * self.physical_scale)
        scaled_h = round(h_m * self.physical_scale)
        scaled_moving = cv2.resize(moving_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        g_f = self._preprocess_gray(fixed_bgr)
        best_ncc = -1.0
        best_mat = None
        best_angle = 0
        best_loc = (0, 0)

        s_mat = scale_matrix(self.physical_scale, self.physical_scale)

        for angle in range(self.angle_range[0], self.angle_range[1], self.angle_step):
            r_mat = rotation_matrix_2d((scaled_w / 2.0, scaled_h / 2.0), angle, 1.0)
            rotated_scaled = cv2.warpAffine(
                scaled_moving,
                affine(r_mat),
                (scaled_w, scaled_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            template = self._preprocess_gray(rotated_scaled)

            # 模板必须小于搜索图
            if template.shape[0] >= g_f.shape[0] or template.shape[1] >= g_f.shape[1]:
                continue

            score_map = cv2.matchTemplate(g_f, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(score_map)

            if score > best_ncc:
                best_ncc = float(score)
                best_angle = angle
                best_loc = location
                t_mat = translation_matrix(location[0], location[1])
                total_mat = affine(t_mat @ r_mat @ s_mat)
                best_mat = total_mat

        is_valid = best_mat is not None and best_ncc >= self.min_ncc
        metrics = QCMetrics(
            ncc_score=best_ncc,
            scale=self.physical_scale,
            rotation_deg=float(best_angle),
            method="PHYSICAL_SCALE_NCC",
            details={"location": best_loc, "search_angle": best_angle},
        )
        return MatchResult(
            matrix=best_mat,
            metrics=metrics,
            is_valid=is_valid,
            details={"ncc": best_ncc, "angle": best_angle},
        )
