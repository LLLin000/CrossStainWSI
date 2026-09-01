"""
跨染色全局深度形态配准器 (GlobalRegistrar)
支持多角度 LoFTR 搜索、组织岛隔离匹配与全片级回退
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.tissue.islands import TissueIsland, TissueSegmenter
from crossstainwsi.transforms.geom import affine, h, standard_90deg_rotation, translation_matrix


@dataclass
class GlobalAlignmentResult:
    is_valid: bool
    mat_moving_to_ref_lvl4: Optional[np.ndarray]
    angle: int
    island_idx: Optional[int]
    metrics: QCMetrics
    is_fallback_full_wsi: bool
    details: Dict[str, Any]


class GlobalRegistrar:
    """
    负责在待配准切片 (Moving) 与参考切片 (Reference) 之间建立全局几何关联
    """
    def __init__(self, loftr_matcher: Optional[LoFTRMatcher] = None):
        self.loftr = loftr_matcher or LoFTRMatcher()

    def align_multiangle(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
        expected_scale_from_mpp: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], QCMetrics, int, float]:
        """
        在 0°, 90°, 180°, 270° 四个正交旋转角度下运行 LoFTR，寻找最高得分对齐
        评分公式: score = inliers * inlier_ratio * (1.0 + spatial_coverage)
        """
        h_m, w_m = moving_bgr.shape[:2]
        best_score = -1.0
        best_mat = None
        best_angle = 0
        best_metrics = QCMetrics(method="LoFTR_MultiAngle")

        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rot_img = moving_bgr
            elif angle == 90:
                rot_img = cv2.rotate(moving_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rot_img = cv2.rotate(moving_bgr, cv2.ROTATE_180)
            elif angle == 270:
                rot_img = cv2.rotate(moving_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rot_mat_3x3 = standard_90deg_rotation(angle, w_m, h_m)
            match_res = self.loftr.match(rot_img, fixed_bgr)

            if match_res.is_valid and match_res.matrix is not None:
                m = match_res.metrics
                score = m.inliers * m.inlier_ratio * (1.0 + m.spatial_coverage)

                # 物理残差尺度约束: 0.88 ~ 1.12 (相对于两张切片 MPP 理论倍率比)
                residual_scale = m.scale / max(1e-5, expected_scale_from_mpp)
                if 0.88 <= residual_scale <= 1.12 and score > best_score:
                    best_score = score
                    best_angle = angle
                    # 将旋转补偿合并进总变换矩阵: M_total = M_loftr @ M_rot
                    total_mat = affine(h(match_res.matrix) @ rot_mat_3x3)
                    best_mat = total_mat
                    best_metrics = m

        return best_mat, best_metrics, best_angle, best_score

    def register_stain(
        self,
        moving_lvl4_bgr: np.ndarray,
        fixed_lvl4_bgr: np.ndarray,
        target_ref_island: TissueIsland,
        expected_scale_from_mpp: float = 1.0,
        min_island_inliers: int = 20,
        min_fallback_inliers: int = 12,
    ) -> GlobalAlignmentResult:
        """
        执行多组织岛配准与全片回退配准
        """
        moving_islands = TissueSegmenter.find_tissue_islands(moving_lvl4_bgr)
        best_candidate = None
        best_cand_score = -1.0

        # 1. 优先尝试与各组织岛进行局部隔离匹配
        for idx, mov_isl in enumerate(moving_islands):
            mat_isl, metrics_isl, angle_isl, score_isl = self.align_multiangle(
                mov_isl.image,
                target_ref_island.image,
                expected_scale_from_mpp=expected_scale_from_mpp,
            )
            if mat_isl is not None and score_isl > best_cand_score:
                best_cand_score = score_isl
                best_candidate = {
                    "island_idx": idx,
                    "mat": mat_isl,
                    "metrics": metrics_isl,
                    "angle": angle_isl,
                    "moving_island": mov_isl,
                }

        # 2. 检查组织岛匹配质量，必要时启用全片回退
        if best_candidate is None or best_candidate["metrics"].inliers < min_island_inliers:
            mat_full, metrics_full, angle_full, score_full = self.align_multiangle(
                moving_lvl4_bgr,
                fixed_lvl4_bgr,
                expected_scale_from_mpp=expected_scale_from_mpp,
            )
            if mat_full is None or metrics_full.inliers < min_fallback_inliers:
                return GlobalAlignmentResult(
                    is_valid=False,
                    mat_moving_to_ref_lvl4=None,
                    angle=angle_full,
                    island_idx=None,
                    metrics=metrics_full,
                    is_fallback_full_wsi=True,
                    details={"reason": f"Insufficient inliers in global alignment ({metrics_full.inliers})"},
                )

            return GlobalAlignmentResult(
                is_valid=True,
                mat_moving_to_ref_lvl4=mat_full,
                angle=angle_full,
                island_idx=None,
                metrics=metrics_full,
                is_fallback_full_wsi=True,
                details={"inliers": metrics_full.inliers, "angle": angle_full},
            )

        # 3. 组织岛匹配成功，计算全局坐标补偿矩阵:
        # M_global = T_ref_off @ M_island @ T_mov_off^(-1)
        mov_isl = best_candidate["moving_island"]
        off_ref_x, off_ref_y = target_ref_island.offset
        off_mov_x, off_mov_y = mov_isl.offset

        t_ref = translation_matrix(off_ref_x, off_ref_y)
        t_mov_inv = translation_matrix(-off_mov_x, -off_mov_y)

        mat_global = affine(t_ref @ h(best_candidate["mat"]) @ t_mov_inv)

        return GlobalAlignmentResult(
            is_valid=True,
            mat_moving_to_ref_lvl4=mat_global,
            angle=best_candidate["angle"],
            island_idx=best_candidate["island_idx"],
            metrics=best_candidate["metrics"],
            is_fallback_full_wsi=False,
            details={
                "island_idx": best_candidate["island_idx"],
                "inliers": best_candidate["metrics"].inliers,
                "angle": best_candidate["angle"],
            },
        )
