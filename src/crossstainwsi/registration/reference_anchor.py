"""
参考切片锚点定位与自检验证器 (支持 Normal vs Mirrored 双奇偶性假设搜索)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import FailureCode, QCMetrics, ROI, CoordinateSpace
from crossstainwsi.io.base import SlideReader
from crossstainwsi.matching.sift import SiftMatcher
from crossstainwsi.matching.template import TemplateMatcher
from crossstainwsi.transforms.geom import affine, apply_mat, h


@dataclass
class AnchorResult:
    is_valid: bool
    mat_anchor_to_lvl4: Optional[np.ndarray]
    center_lvl0: Tuple[float, float]
    center_lvl4: Tuple[float, float]
    anchor_size: Tuple[int, int]
    metrics: QCMetrics
    localization_method: str
    is_mirrored: bool = False
    failure_code: FailureCode = FailureCode.NONE
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    # 兼容旧代码属性
    @property
    def mat_crop4_to_lvl4(self) -> Optional[np.ndarray]:
        return self.mat_anchor_to_lvl4

    @property
    def crop_size(self) -> Tuple[int, int]:
        return self.anchor_size


class ReferenceAnchorLocator:
    """
    负责在参考切片 WSI 中高置信度锁定手工截图视场 (支持正反向镜像自动决策)
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

    def _search_single_parity(
        self,
        crop_bgr: np.ndarray,
        lvl4_bgr: np.ndarray,
        ds4: float,
    ) -> Tuple[Optional[np.ndarray], QCMetrics, str]:
        # 1. 尝试 SIFT 多角度搜索
        sift_res = self.sift_matcher.match(crop_bgr, lvl4_bgr)
        if sift_res.is_valid and sift_res.metrics.inliers >= self.min_sift_inliers:
            return sift_res.matrix, sift_res.metrics, "SIFT_RANSAC"

        # 2. 回退到物理尺度 NCC 模板搜索 (手工 4x 截图相对 Level 0 像素比例约为 5.0)
        physical_scale = 5.0 / max(1.0, ds4)
        tmpl_matcher = TemplateMatcher(
            physical_scale=physical_scale,
            angle_range=(-180, 180),
            angle_step=5,
            min_ncc=self.min_ncc_score,
        )
        tmpl_res = tmpl_matcher.match(crop_bgr, lvl4_bgr)
        if tmpl_res.is_valid:
            return tmpl_res.matrix, tmpl_res.metrics, "PHYSICAL_SCALE_NCC"

        best_metrics = sift_res.metrics if sift_res.metrics.inliers >= (tmpl_res.metrics.inliers) else tmpl_res.metrics
        return None, best_metrics, "FAILED"

    def locate(
        self,
        crop_bgr: np.ndarray,
        slide_reader: SlideReader,
        force_mirror: Optional[bool] = None,
    ) -> AnchorResult:
        """
        执行双奇偶性假设 (Normal vs Mirrored) 锚点定位
        """
        crop_h, crop_w = crop_bgr.shape[:2]
        lvl4_bgr, ds4, dims4 = slide_reader.read_level_image(self.ref_level)

        if force_mirror is True:
            # 明确指定为镜像切片
            flipped = cv2.flip(crop_bgr, 1)
            mat, metrics, method = self._search_single_parity(flipped, lvl4_bgr, ds4)
            chosen_mirror = True
            is_valid = mat is not None
            failure_code = FailureCode.NONE if is_valid else FailureCode.REFERENCE_ANCHOR_FAIL
        elif force_mirror is False:
            # 明确指定为正向切片
            mat, metrics, method = self._search_single_parity(crop_bgr, lvl4_bgr, ds4)
            chosen_mirror = False
            is_valid = mat is not None
            failure_code = FailureCode.NONE if is_valid else FailureCode.REFERENCE_ANCHOR_FAIL
        else:
            # 双假设并行探索 (Normal vs Mirrored)
            mat_norm, met_norm, meth_norm = self._search_single_parity(crop_bgr, lvl4_bgr, ds4)
            flipped = cv2.flip(crop_bgr, 1)
            mat_mirr, met_mirr, meth_mirr = self._search_single_parity(flipped, lvl4_bgr, ds4)

            score_norm = met_norm.inliers if meth_norm == "SIFT_RANSAC" else (met_norm.ncc_score or 0.0) * 50.0
            score_mirr = met_mirr.inliers if meth_mirr == "SIFT_RANSAC" else (met_mirr.ncc_score or 0.0) * 50.0

            if mat_norm is None and mat_mirr is None:
                mat, metrics, method = None, met_norm, "FAILED"
                chosen_mirror = False
                is_valid = False
                failure_code = FailureCode.REFERENCE_ANCHOR_FAIL
            elif mat_mirr is not None and (mat_norm is None or score_mirr > score_norm + 15):
                mat, metrics, method = mat_mirr, met_mirr, meth_mirr
                chosen_mirror = True
                is_valid = True
                failure_code = FailureCode.NONE
            elif mat_norm is not None and (mat_mirr is None or score_norm > score_mirr + 15):
                mat, metrics, method = mat_norm, met_norm, meth_norm
                chosen_mirror = False
                is_valid = True
                failure_code = FailureCode.NONE
            else:
                # 两边得分非常接近，存在镜像二义性
                mat, metrics, method = mat_norm, met_norm, meth_norm
                chosen_mirror = False
                is_valid = True
                failure_code = FailureCode.MIRROR_AMBIGUOUS

        if not is_valid or mat is None:
            return AnchorResult(
                is_valid=False,
                mat_anchor_to_lvl4=None,
                center_lvl0=(0.0, 0.0),
                center_lvl4=(0.0, 0.0),
                anchor_size=(crop_w, crop_h),
                metrics=metrics,
                localization_method="FAILED",
                is_mirrored=chosen_mirror,
                failure_code=failure_code,
                details={"reason": "Reference anchor rejected under both normal and mirrored hypotheses"},
            )

        mat_anchor_to_lvl4 = affine(mat)
        center_crop = np.float32([[crop_w / 2.0, crop_h / 2.0]])
        center_lvl4_pt = apply_mat(mat_anchor_to_lvl4, center_crop)[0]
        center_lvl0_pt = (float(center_lvl4_pt[0] * ds4), float(center_lvl4_pt[1] * ds4))

        return AnchorResult(
            is_valid=True,
            mat_anchor_to_lvl4=mat_anchor_to_lvl4,
            center_lvl0=center_lvl0_pt,
            center_lvl4=(float(center_lvl4_pt[0]), float(center_lvl4_pt[1])),
            anchor_size=(crop_w, crop_h),
            metrics=metrics,
            localization_method=method,
            is_mirrored=chosen_mirror,
            failure_code=failure_code,
            details={
                "inliers": metrics.inliers,
                "ncc_score": metrics.ncc_score,
                "scale": metrics.scale,
                "rotation_deg": metrics.rotation_deg,
                "is_mirrored": chosen_mirror,
            },
        )
