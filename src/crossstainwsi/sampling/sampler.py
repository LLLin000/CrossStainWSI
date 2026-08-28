"""
直接从 WSI 金字塔各层 (Level 0 / Level 2) 进行逆映射单次重采样器 (WSISampler)
避免多次旋转裁剪产生的累积插值模糊与人工白边
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.io.base import SlideReader
from crossstainwsi.transforms.geom import affine, apply_mat, h


class WSISampler:
    """
    负责将任意目标截图像素坐标空间通过逆映射矩阵直接从 WSI 提取高保真图像
    """
    @staticmethod
    def sample_patch(
        slide_reader: SlideReader,
        mat_target_to_slide_level: np.ndarray,
        target_size: Tuple[int, int],
        level: int = 0,
        pad_pixels: int = 64,
        interpolation: int = cv2.INTER_LINEAR,
        border_value: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        以目标截图为基准，将 target_size 像素区域从 slide_reader 的指定 level 进行单次直接采样
        mat_target_to_slide_level: 将目标截图坐标 (0..w, 0..h) 映射到该 level 像素坐标的 2x3 或 3x3 矩阵
        """
        tw, th = target_size
        corners_target = np.float32([
            [0, 0],
            [tw, 0],
            [0, th],
            [tw, th],
        ])

        m_target_to_level = h(mat_target_to_slide_level)
        pts_in_level = apply_mat(affine(m_target_to_level), corners_target)

        spec = slide_reader.read_metadata()
        level_dims = spec.get_level_dimensions(level)
        level_ds = spec.get_level_downsample(level)

        min_x = int(max(0, pts_in_level[:, 0].min() - pad_pixels))
        min_y = int(max(0, pts_in_level[:, 1].min() - pad_pixels))
        max_x = int(min(level_dims[0], pts_in_level[:, 0].max() + pad_pixels))
        max_y = int(min(level_dims[1], pts_in_level[:, 1].max() + pad_pixels))

        if max_x <= min_x or max_y <= min_y:
            raise ValueError(
                f"Sampling ROI is out of slide bounds at level {level}: "
                f"bounding box ({min_x}, {min_y}, {max_x}, {max_y}) vs dims {level_dims}"
            )

        fetch_w = max_x - min_x
        fetch_h = max_y - min_y

        # read_region 的起始点在 Level 0 物理坐标空间
        loc_l0 = (int(round(min_x * level_ds)), int(round(min_y * level_ds)))
        patch_bgr = slide_reader.read_region(loc_l0, level, (fetch_w, fetch_h))

        # 调整映射矩阵以对齐局部 patch 的左上角偏移 (min_x, min_y)
        t_shift = np.array([
            [1.0, 0.0, -float(min_x)],
            [0.0, 1.0, -float(min_y)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        m_patch_target = affine(t_shift @ m_target_to_level)

        # 使用 WARP_INVERSE_MAP 直接将 patch_bgr 采样到 (tw, th)
        sampled = cv2.warpAffine(
            patch_bgr,
            m_patch_target,
            (tw, th),
            flags=interpolation | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return sampled
