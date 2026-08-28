"""
参考切片 (Masson) 锚点定位与同染色自检验证器
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics, ROI, CoordinateSpace
from crossstainwsi.io.base import SlideReader
from crossstainwsi.matching.sift import SiftMatcher
from crossstainwsi.matching.template import TemplateMatcher
from crossstainwsi.transforms.geom import affine, apply_mat, h


@dataclass
class AnchorResult:
    is_valid: bool
    mat_crop4_to_lvl4: Optional[np.ndarray]
    center_lvl0: Tuple[float, float]
    center_lvl4: Tuple[float, float]
    crop_size: Tuple[int, int]
    metrics: QCMetrics
    localization_method: str
    details: Dict[str, Any]


class ReferenceAnchorLocator:
    """
    负责在参考切片 (Masson) WSI 中高置信度锁定手工截图视场
    """
    def __init__(
        self,
        min_sift_inliers: int = 15,
        min_ncc_score: float = 0.30,
        ref_level: int = 4,
    ):
        self.min_sift_inliers = min_sift_inliers
        self.min_ncc_score = min_ncc_score
        self.ref_level = ref_level
        self.sift_matcher = SiftMatcher(
            nfeatures=4000,
            contrast_threshold=0.01,
            ratio_threshold=0.78,
            ransac_threshold=5.0,
            angle_step=15,
            angle_range=(-60, 60),
        )

    def locate(
        self,
        crop4_bgr: np.ndarray,
        slide_reader: SlideReader,
    ) -> AnchorResult:
        """
        执行双阶段锚点定位:
        Stage 1: SIFT 多角度粗定位
        Stage 2: 物理尺度 NCC 模板搜索回退
        """
        crop_h, crop_w = crop4_bgr.shape[:2]
        lvl4_bgr, ds4, dims4 = slide_reader.read_level_image(self.ref_level)
        spec = slide_reader.read_metadata()

        # 1. 尝试 SIFT 多角度搜索
        sift_res = self.sift_matcher.match(crop4_bgr, lvl4_bgr)
        best_mat = None
        best_inliers = sift_res.metrics.inliers
        loc_method = "SIFT_RANSAC"
        metrics = sift_res.metrics

        if sift_res.is_valid and best_inliers >= self.min_sift_inliers:
            best_mat = sift_res.matrix
        else:
            # 2. 回退到物理尺度 NCC 模板搜索
            # 手工 4x 截图像素与 Level 0 像素物理比例约为 5.0
            physical_scale = 5.0 / ds4
            tmpl_matcher = TemplateMatcher(
                physical_scale=physical_scale,
                angle_range=(-180, 180),
                angle_step=5,
                min_ncc=self.min_ncc_score,
            )
            tmpl_res = tmpl_matcher.match(crop4_bgr, lvl4_bgr)
            if tmpl_res.is_valid:
                best_mat = tmpl_res.matrix
                loc_method = "PHYSICAL_SCALE_NCC"
                metrics = tmpl_res.metrics

        if best_mat is None:
            return AnchorResult(
                is_valid=False,
                mat_crop4_to_lvl4=None,
                center_lvl0=(0.0, 0.0),
                center_lvl4=(0.0, 0.0),
                crop_size=(crop_w, crop_h),
                metrics=metrics,
                localization_method="FAILED",
                details={"reason": "Anchor localization rejected by both SIFT and NCC"},
            )

        mat_crop4_to_lvl4 = affine(best_mat)
        center_crop = np.float32([[crop_w / 2.0, crop_h / 2.0]])
        center_lvl4_pt = apply_mat(mat_crop4_to_lvl4, center_crop)[0]
        center_lvl0_pt = (float(center_lvl4_pt[0] * ds4), float(center_lvl4_pt[1] * ds4))

        return AnchorResult(
            is_valid=True,
            mat_crop4_to_lvl4=mat_crop4_to_lvl4,
            center_lvl0=center_lvl0_pt,
            center_lvl4=(float(center_lvl4_pt[0]), float(center_lvl4_pt[1])),
            crop_size=(crop_w, crop_h),
            metrics=metrics,
            localization_method=loc_method,
            details={
                "inliers": metrics.inliers,
                "ncc_score": metrics.ncc_score,
                "scale": metrics.scale,
                "rotation_deg": metrics.rotation_deg,
            },
        )
