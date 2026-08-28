"""
切片扫描与视场采集物理特征配置 (AcquisitionProfile)
将硬编码的 0.2 / 5x 视场参数提升为可配置、可推导的物理协议
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class AcquisitionProfile:
    """
    数字病理软件视场采集先验特征
    """
    # 软件在不同放大倍率下是否保持视场几何中心对齐 (KFSlicer 等软件默认为 True)
    same_center: bool = True

    # 高倍率与低倍率物理采样分辨率比率 (例如 20x 相比于 4x 像素精细 5.0 倍)
    sampling_scale_ratio: float = 5.0

    def derive_crop20_to_crop4_matrix(
        self,
        crop4_size: Tuple[int, int],
        crop20_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        根据物理视场先验，生成从 20x Crop 像素到 4x Crop 像素的 3x3 齐次映射矩阵
        """
        w4, h4 = crop4_size
        w20, h20 = crop20_size
        scale = 1.0 / self.sampling_scale_ratio

        if self.same_center:
            tx = (w4 / 2.0) - scale * (w20 / 2.0)
            ty = (h4 / 2.0) - scale * (h20 / 2.0)
        else:
            tx, ty = 0.0, 0.0

        mat = np.array([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return mat
