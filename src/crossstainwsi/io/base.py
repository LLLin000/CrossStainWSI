"""
SlideReader 抽象基类与接口契约
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from crossstainwsi.domain import SlideSpec


class SlideReader(ABC):
    """
    全切片数字病理图像 (WSI) 读取器抽象接口
    统一屏蔽底层驱动细节 (KFB, OpenSlide, TIFF 等)
    """
    def __init__(self, path: Path, default_mpp: float = 0.44243):
        self.path = Path(path)
        self.default_mpp = default_mpp

    @abstractmethod
    def read_metadata(self) -> SlideSpec:
        """
        读取切片层级、物理分辨率 (MPP)、尺寸等元数据
        """
        pass

    @abstractmethod
    def read_level_image(self, level: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        读取指定金字塔层级的完整图像 (BGR 格式)
        返回: (img_bgr, downsample_factor, (width, height))
        """
        pass

    @abstractmethod
    def read_region(
        self,
        location_l0: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        以 Level 0 坐标原点 (x, y) 为基准，在指定 level 提取指定大小的局部图像块 (BGR 格式)
        location_l0: (x_l0, y_l0)
        size: (width, height) 在该 level 上的像素尺寸
        """
        pass

    def close(self) -> None:
        """释放底层句柄"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
