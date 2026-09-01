"""
坐标变换图 (TransformGraph)
管理从 Crop 各尺度、Native ROI, Reference WSI 各 Level 到 Moving WSI 各 Level 的变换链条与复合求解
"""

from typing import Dict, Optional, Tuple
import numpy as np

from crossstainwsi.domain import CoordinateSpace
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.transforms.geom import affine, apply_mat, h, invert_transform, scale_matrix, translation_matrix


class TransformGraph:
    """
    单样本跨染色配准变换拓扑图 (支持自适应先验与 Native ROI)
    """
    def __init__(
        self,
        crop4_size: Tuple[int, int],
        crop20_size: Tuple[int, int],
        ref_ds_lvl2: float,
        ref_ds_lvl4: float,
        moving_ds_lvl2: float,
        moving_ds_lvl4: float,
        acquisition_profile: Optional[AcquisitionProfile] = None,
    ):
        self.crop4_w, self.crop4_h = crop4_size
        self.crop20_w, self.crop20_h = crop20_size
        self.ref_ds_lvl2 = ref_ds_lvl2
        self.ref_ds_lvl4 = ref_ds_lvl4
        self.moving_ds_lvl2 = moving_ds_lvl2
        self.moving_ds_lvl4 = moving_ds_lvl4
        self.profile = acquisition_profile or AcquisitionProfile()

        # 根据采集协议先验自动推导多视场之间的相对像素映射
        self.mat_crop20_to_crop4 = self.profile.derive_crop20_to_crop4_matrix(
            (self.crop4_w, self.crop4_h),
            (self.crop20_w, self.crop20_h),
        )

        # 存储拓扑节点间的变换 (3x3 齐次矩阵)
        self.mat_crop4_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_crop4_to_ref_lvl2: Optional[np.ndarray] = None
        self.mat_moving_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_ref_to_moving_lvl4: Optional[np.ndarray] = None
        self.mat_crop4_to_moving_lvl2: Optional[np.ndarray] = None
        self.mat_local_refinement_3x3: np.ndarray = np.eye(3, dtype=np.float64)

    def set_reference_anchor(self, mat_crop4_to_ref_lvl4: np.ndarray) -> None:
        """
        设置基于图像反查的参考切片锚点矩阵 (Crop4 -> Ref L4)
        """
        self.mat_crop4_to_ref_lvl4 = h(mat_crop4_to_ref_lvl4)
        scale_lvl4_to_lvl2 = scale_matrix(
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
        )
        self.mat_crop4_to_ref_lvl2 = scale_lvl4_to_lvl2 @ self.mat_crop4_to_ref_lvl4

    def set_native_reference_roi(
        self,
        center_lvl0: Tuple[float, float],
        target_size_lvl0: Tuple[float, float],
    ) -> None:
        """
        设置原生 WSI 框选 ROI (0 锚点反查误差, 直接通过物理坐标构建映射)
        """
        cx0, cy0 = center_lvl0
        w0, h0 = target_size_lvl0
        # 从 Crop4 (w4, h4) 映射到 Level 0
        scale_crop_to_l0_x = w0 / max(1.0, float(self.crop4_w))
        scale_crop_to_l0_y = h0 / max(1.0, float(self.crop4_h))
        tx0 = cx0 - (w0 / 2.0)
        ty0 = cy0 - (h0 / 2.0)

        m_crop_to_l0 = np.array([
            [scale_crop_to_l0_x, 0.0, tx0],
            [0.0, scale_crop_to_l0_y, ty0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 映射到 Level 4 与 Level 2
        s_l0_to_l4 = scale_matrix(1.0 / self.ref_ds_lvl4, 1.0 / self.ref_ds_lvl4)
        s_l0_to_l2 = scale_matrix(1.0 / self.ref_ds_lvl2, 1.0 / self.ref_ds_lvl2)

        self.mat_crop4_to_ref_lvl4 = s_l0_to_l4 @ m_crop_to_l0
        self.mat_crop4_to_ref_lvl2 = s_l0_to_l2 @ m_crop_to_l0

    def set_global_cross_stain(self, mat_moving_to_ref_lvl4: np.ndarray) -> None:
        """
        设置跨染色全局变换 (Moving L4 -> Ref L4)
        自动计算 Ref L4 -> Moving L4 与 Crop4 -> Moving L2
        """
        if self.mat_crop4_to_ref_lvl2 is None:
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

        self.mat_crop4_to_moving_lvl2 = (
            scale_m @ self.mat_ref_to_moving_lvl4 @ scale_f @ self.mat_crop4_to_ref_lvl2
        )

    def set_local_refinement(self, mat_local_3x3: np.ndarray) -> None:
        self.mat_local_refinement_3x3 = h(mat_local_3x3)

    def get_crop4_to_moving_lvl2(self) -> np.ndarray:
        if self.mat_crop4_to_moving_lvl2 is None:
            raise ValueError("Cross-stain global registration has not been computed")
        return self.mat_crop4_to_moving_lvl2

    def get_crop20_to_moving_lvl0(self) -> np.ndarray:
        """
        计算 20x 截图像素空间直接到 Moving 切片 Level 0 像素坐标的完整复合矩阵:
        M_total = S(lvl2->lvl0) @ M(crop4->moving_lvl2) @ M_local^(-1) @ M(crop20->crop4)
        """
        if self.mat_crop4_to_moving_lvl2 is None:
            raise ValueError("Cross-stain registration has not been initialized")

        scale_l2_to_l0 = scale_matrix(self.moving_ds_lvl2, self.moving_ds_lvl2)
        m_local_inv = invert_transform(self.mat_local_refinement_3x3)

        m_total_20x_to_l0 = (
            scale_l2_to_l0
            @ self.mat_crop4_to_moving_lvl2
            @ m_local_inv
            @ self.mat_crop20_to_crop4
        )
        return m_total_20x_to_l0

    def get_view_to_moving_lvl0(
        self,
        target_mag: float = 4.0,
        base_mag: float = 4.0,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        计算任意倍率 (如 4x, 10x, 20x, 40x) 视图直接到 Moving 切片 Level 0 的复合映射矩阵
        """
        if self.mat_crop4_to_moving_lvl2 is None:
            raise ValueError("Cross-stain registration has not been initialized")

        tw, th = target_size or (self.crop4_w, self.crop4_h)
        ratio = max(0.1, target_mag / max(0.1, base_mag))
        scale = 1.0 / ratio

        # 保持中心对齐
        tx = (self.crop4_w / 2.0) - scale * (tw / 2.0)
        ty = (self.crop4_h / 2.0) - scale * (th / 2.0)

        m_view_to_crop4 = np.array([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        scale_l2_to_l0 = scale_matrix(self.moving_ds_lvl2, self.moving_ds_lvl2)
        m_local_inv = invert_transform(self.mat_local_refinement_3x3)

        m_total_view_to_l0 = (
            scale_l2_to_l0
            @ self.mat_crop4_to_moving_lvl2
            @ m_local_inv
            @ m_view_to_crop4
        )
        return m_total_view_to_l0

    def get_crop4_to_moving_lvl0(self) -> np.ndarray:
        return self.get_view_to_moving_lvl0(target_mag=4.0, base_mag=4.0, target_size=(self.crop4_w, self.crop4_h))
