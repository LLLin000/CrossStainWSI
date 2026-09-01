"""
参考切片锚点定位与自检验证器 (支持 EvidenceView、Normal vs Mirrored 物理反射矩阵合成与严格二义性拦截)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from crossstainwsi.domain import CoordinateSpace, EvidenceView, FailureCode, QCMetrics, ROI
from crossstainwsi.io.base import SlideReader
from crossstainwsi.matching.sift import SiftMatcher
from crossstainwsi.matching.template import TemplateMatcher
from crossstainwsi.transforms.geom import affine, apply_mat, h


@dataclass
class AnchorResult:
    is_valid: bool
    mat_anchor_to_lvl4: Optional[np.ndarray] # 严格表示从原始输入截图坐标 (x, y) 到 WSI Level 4 的变换 (含镜像反射)
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

    @property
    def mat_crop4_to_lvl4(self) -> Optional[np.ndarray]:
        return self.mat_anchor_to_lvl4

    @property
    def crop_size(self) -> Tuple[int, int]:
        return self.anchor_size


class ReferenceAnchorLocator:
    """
    负责在参考切片 WSI 中高置信度锁定证据视场 (支持任意 EvidenceView 倍率与反射变换自洽合成)
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
        nominal_mag: float = 4.0,
        evidence_mpp: Optional[float] = None,
        ref_l0_mpp: float = 0.44243,
    ) -> Tuple[Optional[np.ndarray], QCMetrics, str]:
        # 1. 尝试 SIFT 多角度搜索
        sift_res = self.sift_matcher.match(crop_bgr, lvl4_bgr)
        if sift_res.is_valid and sift_res.metrics.inliers >= self.min_sift_inliers:
            return sift_res.matrix, sift_res.metrics, "SIFT_RANSAC"

        # 2. 回退到物理尺度 NCC 模板搜索:
        # 缩放因子 s = (证据图物理像素大小 MPP_evidence) / (Level 4 物理像素大小 MPP_ref_L0 * ds4)
        if evidence_mpp is not None and evidence_mpp > 0:
            physical_scale = evidence_mpp / max(1e-5, ref_l0_mpp * ds4)
        else:
            native_scan_mag = 20.0
            crop_to_l0_ratio = native_scan_mag / max(0.1, float(nominal_mag))
            physical_scale = crop_to_l0_ratio / max(1.0, ds4)

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
        anchor_input: Union[EvidenceView, np.ndarray],
        slide_reader: SlideReader,
        force_mirror: Optional[bool] = None,
        crop_bgr_override: Optional[np.ndarray] = None,
    ) -> AnchorResult:
        """
        执行双奇偶性假设 (Normal vs Mirrored) 锚点定位
        当镜像假设胜出时，自动将水平反射矩阵 F_x 合并进返回的变换矩阵中
        """
        if isinstance(anchor_input, EvidenceView):
            crop_w, crop_h = anchor_input.width_px, anchor_input.height_px
            nominal_mag = anchor_input.nominal_magnification
            crop_bgr = crop_bgr_override
            evidence_mpp = anchor_input.mpp_xy[0] if anchor_input.mpp_xy else None
        else:
            crop_bgr = anchor_input
            crop_h, crop_w = crop_bgr.shape[:2]
            nominal_mag = 4.0
            evidence_mpp = None

        if crop_bgr is None:
            raise ValueError("Image array must be provided for anchor localization")

        lvl4_bgr, ds4, dims4 = slide_reader.read_level_image(self.ref_level)
        ref_spec = slide_reader.read_metadata()
        ref_l0_mpp = ref_spec.mpp_x

        # 构造水平翻转反射齐次矩阵 F_x: x' = w - 1 - x, y' = y
        f_x = np.array([
            [-1.0, 0.0, float(crop_w - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        if force_mirror is True:
            flipped = cv2.flip(crop_bgr, 1)
            mat_mirr, metrics, method = self._search_single_parity(
                flipped, lvl4_bgr, ds4, nominal_mag, evidence_mpp, ref_l0_mpp
            )
            chosen_mirror = True
            is_valid = mat_mirr is not None
            failure_code = FailureCode.NONE if is_valid else FailureCode.REFERENCE_ANCHOR_FAIL
            mat = affine(h(mat_mirr) @ f_x) if is_valid else None

        elif force_mirror is False:
            mat_norm, metrics, method = self._search_single_parity(
                crop_bgr, lvl4_bgr, ds4, nominal_mag, evidence_mpp, ref_l0_mpp
            )
            chosen_mirror = False
            is_valid = mat_norm is not None
            failure_code = FailureCode.NONE if is_valid else FailureCode.REFERENCE_ANCHOR_FAIL
            mat = mat_norm

        else:
            # 双假设并行探索 (Normal vs Mirrored)
            mat_norm, met_norm, meth_norm = self._search_single_parity(
                crop_bgr, lvl4_bgr, ds4, nominal_mag, evidence_mpp, ref_l0_mpp
            )
            flipped = cv2.flip(crop_bgr, 1)
            mat_mirr, met_mirr, meth_mirr = self._search_single_parity(
                flipped, lvl4_bgr, ds4, nominal_mag, evidence_mpp, ref_l0_mpp
            )

            score_norm = met_norm.inliers if meth_norm == "SIFT_RANSAC" else (met_norm.ncc_score or 0.0) * 50.0
            score_mirr = met_mirr.inliers if meth_mirr == "SIFT_RANSAC" else (met_mirr.ncc_score or 0.0) * 50.0

            if mat_norm is None and mat_mirr is None:
                mat, metrics, method = None, met_norm, "FAILED"
                chosen_mirror = False
                is_valid = False
                failure_code = FailureCode.REFERENCE_ANCHOR_FAIL
            elif mat_mirr is not None and (mat_norm is None or score_mirr > score_norm + 15):
                # 镜像假设显著胜出，合成包含 F_x 的完整映射矩阵
                mat = affine(h(mat_mirr) @ f_x)
                metrics, method = met_mirr, meth_mirr
                chosen_mirror = True
                is_valid = True
                failure_code = FailureCode.NONE
            elif mat_norm is not None and (mat_mirr is None or score_norm > score_mirr + 15):
                mat = mat_norm
                metrics, method = met_norm, meth_norm
                chosen_mirror = False
                is_valid = True
                failure_code = FailureCode.NONE
            else:
                # 两边得分非常接近，存在镜像二义性 -> 严格拦截拒绝 (is_valid=False)
                mat, metrics, method = None, met_norm, meth_norm
                chosen_mirror = False
                is_valid = False
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
                details={"reason": f"Reference anchor rejected ({failure_code.value})"},
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
