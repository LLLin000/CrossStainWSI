"""
切片扫描与视场采集物理特征配置 (AcquisitionProfile)
将硬编码的 0.2 / 5x 视场参数提升为可配置、基于 MPP 与物理尺度的自适应协议
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np


@dataclass
class AcquisitionProfile:
    """
    数字病理软件视场采集先验特征
    """
    # 软件在不同放大倍率下是否保持视场几何中心对齐 (KFSlicer 等软件默认为 True)
    same_center: bool = True

    # 默认采样倍率比例 (当无法获取 MPP 时的备用兜底比例)
    sampling_scale_ratio: float = 5.0

    def derive_view_to_anchor_matrix(
        self,
        anchor_view: Any,
        target_view: Any,
    ) -> np.ndarray:
        """
        根据物理 MPP 或名义倍率先验，生成从 Target View 到 Anchor View 的 3x3 齐次映射矩阵
        """
        # 1. 提取尺寸
        if hasattr(anchor_view, "width_px") and hasattr(anchor_view, "height_px"):
            w_anc, h_anc = anchor_view.width_px, anchor_view.height_px
            mpp_anc = getattr(anchor_view, "mpp_xy", None)
            mag_anc = getattr(anchor_view, "nominal_magnification", 4.0)
        elif isinstance(anchor_view, (tuple, list)):
            w_anc, h_anc = anchor_view[0], anchor_view[1]
            mpp_anc = None
            mag_anc = 4.0
        else:
            w_anc, h_anc = 2257, 1310
            mpp_anc = None
            mag_anc = 4.0

        if hasattr(target_view, "width_px") and hasattr(target_view, "height_px"):
            w_tgt, h_tgt = target_view.width_px, target_view.height_px
            mpp_tgt = getattr(target_view, "mpp_xy", None)
            mag_tgt = getattr(target_view, "nominal_magnification", 20.0)
        elif hasattr(target_view, "pixel_dimensions"):
            w_tgt, h_tgt = target_view.pixel_dimensions
            mpp_tgt = None
            mag_tgt = getattr(target_view, "magnification_approx", 20.0)
        elif isinstance(target_view, (tuple, list)):
            w_tgt, h_tgt = target_view[0], target_view[1]
            mpp_tgt = None
            mag_tgt = 20.0
        else:
            w_tgt, h_tgt = w_anc, h_anc
            mpp_tgt = None
            mag_tgt = 20.0

        # 2. 尺度因子计算 (优先使用 MPP, 否则使用名义倍率比)
        if mpp_anc is not None and mpp_tgt is not None and mpp_anc[0] > 0 and mpp_tgt[0] > 0:
            scale = float(mpp_tgt[0] / mpp_anc[0])
        else:
            ratio = max(0.01, float(mag_tgt) / max(0.01, float(mag_anc)))
            scale = 1.0 / ratio

        # 3. 中心对齐平移量
        if self.same_center:
            tx = (w_anc / 2.0) - scale * (w_tgt / 2.0)
            ty = (h_anc / 2.0) - scale * (h_tgt / 2.0)
        else:
            tx, ty = 0.0, 0.0

        mat = np.array([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return mat

    def derive_crop20_to_crop4_matrix(
        self,
        crop4_size: Tuple[int, int],
        crop20_size: Tuple[int, int],
    ) -> np.ndarray:
        """向后兼容接口"""
        return self.derive_view_to_anchor_matrix(crop4_size, crop20_size)
