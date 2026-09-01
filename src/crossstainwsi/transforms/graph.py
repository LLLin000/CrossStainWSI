"""
坐标变换图 (TransformGraph)
管理从各证据视场 (EvidenceView)、Native ROI, Reference WSI 各 Level 到 Moving WSI 各 Level 的变换链条与复合求解
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from crossstainwsi.domain import CoordinateSpace, EvidenceView
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.transforms.geom import affine, apply_mat, h, invert_transform, scale_matrix, translation_matrix


class TransformGraph:
    """
    通用跨模态切片配准变换拓扑图
    支持任意主证据视场 (Anchor EvidenceView)、多尺度子视场及任意目标提取视图
    """
    def __init__(
        self,
        anchor_view: Optional[Union[EvidenceView, Tuple[int, int]]] = None,
        secondary_view: Optional[Union[EvidenceView, Tuple[int, int]]] = None,
        ref_ds_lvl2: float = 4.0,
        ref_ds_lvl4: float = 16.0,
        moving_ds_lvl2: float = 4.0,
        moving_ds_lvl4: float = 16.0,
        acquisition_profile: Optional[AcquisitionProfile] = None,
        crop4_size: Optional[Tuple[int, int]] = None,
        crop20_size: Optional[Tuple[int, int]] = None,
    ):
        if anchor_view is None:
            anchor_view = crop4_size or (2257, 1310)
        if secondary_view is None and crop20_size is not None:
            secondary_view = crop20_size
        if isinstance(anchor_view, EvidenceView):
            self.anchor_w, self.anchor_h = anchor_view.width_px, anchor_view.height_px
            self.anchor_mag = anchor_view.nominal_magnification
        else:
            self.anchor_w, self.anchor_h = anchor_view
            self.anchor_mag = 4.0

        if secondary_view is not None:
            if isinstance(secondary_view, EvidenceView):
                self.secondary_w, self.secondary_h = secondary_view.width_px, secondary_view.height_px
                self.secondary_mag = secondary_view.nominal_magnification
            else:
                self.secondary_w, self.secondary_h = secondary_view
                self.secondary_mag = 20.0
        else:
            self.secondary_w, self.secondary_h = self.anchor_w, self.anchor_h
            self.secondary_mag = 20.0

        # 向后兼容属性
        self.crop4_w, self.crop4_h = self.anchor_w, self.anchor_h
        self.crop20_w, self.crop20_h = self.secondary_w, self.secondary_h

        self.ref_ds_lvl2 = ref_ds_lvl2
        self.ref_ds_lvl4 = ref_ds_lvl4
        self.moving_ds_lvl2 = moving_ds_lvl2
        self.moving_ds_lvl4 = moving_ds_lvl4
        self.profile = acquisition_profile or AcquisitionProfile()

        # 推导主视场与子视场之间的相对像素映射
        self.mat_secondary_to_anchor = self.profile.derive_crop20_to_crop4_matrix(
            (self.anchor_w, self.anchor_h),
            (self.secondary_w, self.secondary_h),
        )
        self.mat_crop20_to_crop4 = self.mat_secondary_to_anchor

        # 存储拓扑节点间的变换 (3x3 齐次矩阵)
        self.mat_anchor_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_anchor_to_ref_lvl2: Optional[np.ndarray] = None
        self.mat_moving_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_ref_to_moving_lvl4: Optional[np.ndarray] = None
        self.mat_anchor_to_moving_lvl2: Optional[np.ndarray] = None
        self.mat_local_refinement_3x3: np.ndarray = np.eye(3, dtype=np.float64)

    # 兼容旧属性名
    @property
    def mat_crop4_to_ref_lvl4(self) -> Optional[np.ndarray]:
        return self.mat_anchor_to_ref_lvl4

    @property
    def mat_crop4_to_ref_lvl2(self) -> Optional[np.ndarray]:
        return self.mat_anchor_to_ref_lvl2

    @property
    def mat_crop4_to_moving_lvl2(self) -> Optional[np.ndarray]:
        return self.mat_anchor_to_moving_lvl2

    def set_reference_anchor(self, mat_anchor_to_ref_lvl4: np.ndarray) -> None:
        """
        设置主证据视场 (Anchor EvidenceView) 到 Reference Slide Level 4 的锚点变换
        """
        self.mat_anchor_to_ref_lvl4 = h(mat_anchor_to_ref_lvl4)
        scale_lvl4_to_lvl2 = scale_matrix(
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
        )
        self.mat_anchor_to_ref_lvl2 = scale_lvl4_to_lvl2 @ self.mat_anchor_to_ref_lvl4

    def set_native_reference_roi(
        self,
        center_lvl0: Tuple[float, float],
        target_size_lvl0: Tuple[float, float],
    ) -> None:
        """
        设置原生 WSI 框选 ROI (0 锚点反查误差)
        """
        cx0, cy0 = center_lvl0
        w0, h0 = target_size_lvl0
        scale_anchor_to_l0_x = w0 / max(1.0, float(self.anchor_w))
        scale_anchor_to_l0_y = h0 / max(1.0, float(self.anchor_h))
        tx0 = cx0 - (w0 / 2.0)
        ty0 = cy0 - (h0 / 2.0)

        m_anchor_to_l0 = np.array([
            [scale_anchor_to_l0_x, 0.0, tx0],
            [0.0, scale_anchor_to_l0_y, ty0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        s_l0_to_l4 = scale_matrix(1.0 / self.ref_ds_lvl4, 1.0 / self.ref_ds_lvl4)
        s_l0_to_l2 = scale_matrix(1.0 / self.ref_ds_lvl2, 1.0 / self.ref_ds_lvl2)

        self.mat_anchor_to_ref_lvl4 = s_l0_to_l4 @ m_anchor_to_l0
        self.mat_anchor_to_ref_lvl2 = s_l0_to_l2 @ m_anchor_to_l0

    def set_global_cross_stain(self, mat_moving_to_ref_lvl4: np.ndarray) -> None:
        """
        设置跨染色全局变换 (Moving L4 -> Ref L4)
        自动计算 Ref L4 -> Moving L4 与 Anchor -> Moving L2
        """
        if self.mat_anchor_to_ref_lvl2 is None:
            raise ValueError("Reference anchor or native ROI must be set before global cross stain registration")

        self.mat_moving_to_ref_lvl4 = h(mat_moving_to_ref_lvl4)
        self.mat_ref_to_moving_lvl4 = invert_transform(self.mat_moving_to_ref_lvl4)

        scale_m = scale_matrix(
            self.moving_ds_lvl4 / self.moving_ds_lvl2,
            self.moving_ds_lvl4 / self.moving_ds_lvl2,
        )
        scale_f = scale_matrix(
            self.ref_ds_lvl2 / self.ref_ds_lvl4,
            self.ref_ds_lvl2 / self.ref_ds_lvl4,
        )

        self.mat_anchor_to_moving_lvl2 = (
            scale_m @ self.mat_ref_to_moving_lvl4 @ scale_f @ self.mat_anchor_to_ref_lvl2
        )

    def set_local_refinement(self, mat_local_3x3: np.ndarray) -> None:
        self.mat_local_refinement_3x3 = h(mat_local_3x3)

    def get_anchor_to_moving_lvl2(self) -> np.ndarray:
        if self.mat_anchor_to_moving_lvl2 is None:
            raise ValueError("Cross-stain global registration has not been computed")
        return self.mat_anchor_to_moving_lvl2

    def get_crop4_to_moving_lvl2(self) -> np.ndarray:
        return self.get_anchor_to_moving_lvl2()

    def get_view_to_moving_lvl0(
        self,
        target_mag: float = 4.0,
        base_mag: float = 4.0,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        计算任意请求输出视图 (如 4x, 10x, 20x, 40x 或自定义尺寸) 像素空间直接到 Moving 切片 Level 0 的复合逆映射矩阵:
        M_total = S(lvl2->lvl0) @ M(anchor->moving_lvl2) @ M_local^(-1) @ M(view->anchor)
        """
        if self.mat_anchor_to_moving_lvl2 is None:
            raise ValueError("Cross-stain registration has not been initialized")

        tw, th = target_size or (self.anchor_w, self.anchor_h)
        ratio = max(0.1, target_mag / max(0.1, base_mag))
        scale = 1.0 / ratio

        # 保持几何中心对齐
        tx = (self.anchor_w / 2.0) - scale * (tw / 2.0)
        ty = (self.anchor_h / 2.0) - scale * (th / 2.0)

        m_view_to_anchor = np.array([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        scale_l2_to_l0 = scale_matrix(self.moving_ds_lvl2, self.moving_ds_lvl2)
        m_local_inv = invert_transform(self.mat_local_refinement_3x3)

        m_total_view_to_l0 = (
            scale_l2_to_l0
            @ self.mat_anchor_to_moving_lvl2
            @ m_local_inv
            @ m_view_to_anchor
        )
        return m_total_view_to_l0

    def get_crop20_to_moving_lvl0(self) -> np.ndarray:
        return self.get_view_to_moving_lvl0(target_mag=20.0, base_mag=4.0, target_size=(self.secondary_w, self.secondary_h))

    def get_crop4_to_moving_lvl0(self) -> np.ndarray:
        return self.get_view_to_moving_lvl0(target_mag=4.0, base_mag=4.0, target_size=(self.anchor_w, self.anchor_h))
