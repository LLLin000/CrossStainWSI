"""
规范结构表征集契约模型 (Canonical Representation Set Contracts)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CanonicalRepresentationSet:
    """
    任意输入切片在适配器处理后生成的规范结构表征集
    解耦具体成像光谱，统一输出几何、轮廓与细胞核概率场
    """
    # 1. 空间掩模
    valid_mask: np.ndarray                 # 有效成像区域布尔掩模 (非暗场死区/非纯白边缘)
    tissue_mask: Optional[np.ndarray] = None # 组织前景二值掩模 (255 表示组织实质)
    artifact_mask: Optional[np.ndarray] = None # 气泡、折皱、划痕、强饱和高光等人工假象掩模

    # 2. 几何与宏观轮廓
    coarse_contour: Optional[np.ndarray] = None # 宏观组织外轮廓二值图 (255 表示轮廓边缘)
    distance_field: Optional[np.ndarray] = None # 组织边缘欧氏距离变换场 (EDT, 浮点型)

    # 3. 结构梯度与核结构场
    gradient_pyramid: Tuple[np.ndarray, ...] = field(default_factory=tuple) # 多尺度 Sobel 梯度幅值
    nuclear_density: Optional[np.ndarray] = None # 跨模态同源核结构响应场 (0.0 ~ 1.0 浮点概率图)
    feature_image: Optional[np.ndarray] = None   # 供深度特征匹配器 (LoFTR/DISK) 直接使用的增强图

    # 4. 物理元数据与溯源
    mpp_xy: Optional[Tuple[float, float]] = None # 物理分辨率 (微米/像素)
    source_level: int = 4                        # 生成该表征所用的金字塔层级
    modality: str = "generic"                    # "brightfield", "ihc_dab", "fluorescence"
    representation_provenance: Dict[str, Any] = field(default_factory=dict) # 记录核密度场提取机理 (H通道 / DAPI / LoG)
    informativeness: Dict[str, float] = field(default_factory=dict)         # 结构信息量指标 (熵, 组织占比, 梯度能量)
