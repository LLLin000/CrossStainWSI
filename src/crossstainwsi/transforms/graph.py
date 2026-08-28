"""
坐标变换图 (TransformGraph)
管理从 Crop4x, Crop20x, Reference WSI 各 Level 到 Moving WSI 各 Level 的变换链条与复合求解
"""

from typing import Dict, Optional, Tuple
import numpy as np

from crossstainwsi.domain import CoordinateSpace
from crossstainwsi.transforms.geom import affine, apply_mat, h, invert_transform, scale_matrix


class TransformGraph:
    """
    单样本跨染色配准变换拓扑图
    """
    def __init__(
        self,
        crop4_size: Tuple[int, int],
        crop20_size: Tuple[int, int],
        ref_ds_lvl2: float,
        ref_ds_lvl4: float,
        moving_ds_lvl2: float,
        moving_ds_lvl4: float,
    ):
        self.crop4_w, self.crop4_h = crop4_size
        self.crop20_w, self.crop20_h = crop20_size
        self.ref_ds_lvl2 = ref_ds_lvl2
        self.ref_ds_lvl4 = ref_ds_lvl4
        self.moving_ds_lvl2 = moving_ds_lvl2
        self.moving_ds_lvl4 = moving_ds_lvl4

        # 4x 到 20x 的固定物理视场比例变换 (20x视场中心与4x重合，像素分辨率5倍精细)
        self.mat_crop20_to_crop4 = np.array([
            [0.2, 0.0, self.crop4_w / 2.0 - 0.2 * self.crop20_w / 2.0],
            [0.0, 0.2, self.crop4_h / 2.0 - 0.2 * self.crop20_h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 存储注册节点间的变换 (3x3 齐次矩阵)
        self.mat_crop4_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_crop4_to_ref_lvl2: Optional[np.ndarray] = None
        self.mat_moving_to_ref_lvl4: Optional[np.ndarray] = None
        self.mat_ref_to_moving_lvl4: Optional[np.ndarray] = None
        self.mat_crop4_to_moving_lvl2: Optional[np.ndarray] = None
        self.mat_local_refinement_3x3: np.ndarray = np.eye(3, dtype=np.float64)

    def set_reference_anchor(self, mat_crop4_to_ref_lvl4: np.ndarray) -> None:
        """
        设置参考切片（如 Masson）中 4x 截图到 Level 4 的锚点变换
        """
        self.mat_crop4_to_ref_lvl4 = h(mat_crop4_to_ref_lvl4)
        # 从 Level 4 映射到 Level 2 (乘以尺度因子 ds4 / ds2)
        scale_lvl4_to_lvl2 = np.diag([
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
            self.ref_ds_lvl4 / self.ref_ds_lvl2,
            1.0,
        ]).astype(np.float64)
        self.mat_crop4_to_ref_lvl2 = scale_lvl4_to_lvl2 @ self.mat_crop4_to_ref_lvl4

    def set_global_cross_stain(self, mat_moving_to_ref_lvl4: np.ndarray) -> None:
        """
        设置跨染色全局变换 (Moving L4 -> Ref L4)
        自动计算 Ref L4 -> Moving L4 与 Crop4 -> Moving L2
        """
        if self.mat_crop4_to_ref_lvl2 is None:
            raise ValueError("Reference anchor must be set before global cross stain registration")

        self.mat_moving_to_ref_lvl4 = h(mat_moving_to_ref_lvl4)
        self.mat_ref_to_moving_lvl4 = invert_transform(self.mat_moving_to_ref_lvl4)

        # scale_m: moving_lvl4 -> moving_lvl2
        scale_m = scale_matrix(
            self.moving_ds_lvl4 / self.moving_ds_lvl2,
            self.moving_ds_lvl4 / self.moving_ds_lvl2
        )
        # scale_f: ref_lvl2 -> ref_lvl4
        scale_f = scale_matrix(
            self.ref_ds_lvl2 / self.ref_ds_lvl4,
            self.ref_ds_lvl2 / self.ref_ds_lvl4
        )

        # Crop4 -> Ref L2 -> Ref L4 -> Moving L4 -> Moving L2
        self.mat_crop4_to_moving_lvl2 = (
            scale_m @ self.mat_ref_to_moving_lvl4 @ scale_f @ self.mat_crop4_to_ref_lvl2
        )

    def set_local_refinement(self, mat_local_3x3: np.ndarray) -> None:
        """
        设置局部残差微调变换 (例如 Local LoFTR 或 Phase Correlation 解算的 3x3)
        """
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

    def get_crop4_to_moving_lvl0(self) -> np.ndarray:
        """
        计算 4x 截图像素空间直接到 Moving 切片 Level 0 像素坐标的完整复合矩阵
        """
        if self.mat_crop4_to_moving_lvl2 is None:
            raise ValueError("Cross-stain registration has not been initialized")

        scale_l2_to_l0 = scale_matrix(self.moving_ds_lvl2, self.moving_ds_lvl2)
        m_local_inv = invert_transform(self.mat_local_refinement_3x3)

        m_total_4x_to_l0 = (
            scale_l2_to_l0
            @ self.mat_crop4_to_moving_lvl2
            @ m_local_inv
        )
        return m_total_4x_to_l0
