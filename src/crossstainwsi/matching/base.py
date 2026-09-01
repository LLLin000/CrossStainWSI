"""
匹配器抽象基类与结果契约 (可插拔三级匹配架构)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from crossstainwsi.domain import EvidenceRecord, QCMetrics


@dataclass
class MatchResult:
    """
    匹配执行结果
    """
    matrix: Optional[np.ndarray]   # 2x3 或 3x3 仿射/相似性变换矩阵
    metrics: QCMetrics
    is_valid: bool
    evidence: Optional[EvidenceRecord] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ImageMatcher(ABC):
    """
    特征、结构或信息匹配器抽象基类
    """
    @abstractmethod
    def match(
        self,
        moving: Union[np.ndarray, Any],
        fixed: Union[np.ndarray, Any],
    ) -> MatchResult:
        """
        在 moving 和 fixed 之间求解几何变换 (moving -> fixed)
        """
        pass
