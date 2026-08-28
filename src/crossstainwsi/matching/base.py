"""
匹配器抽象基类与结果契约
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from crossstainwsi.domain import QCMetrics


@dataclass
class MatchResult:
    """
    匹配执行结果
    """
    matrix: Optional[np.ndarray]   # 2x3 或 3x3 仿射变换矩阵
    metrics: QCMetrics
    is_valid: bool
    details: Dict[str, Any]


class ImageMatcher(ABC):
    """
    特征或图像匹配器抽象基类
    """
    @abstractmethod
    def match(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
    ) -> MatchResult:
        """
        在 moving 和 fixed 图像之间寻找几何变换 (moving -> fixed)
        """
        pass
