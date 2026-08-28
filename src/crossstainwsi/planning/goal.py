"""
用户目标与输出需求规范 (User Goal & View Specifications)
将输出要求与输入证据彻底解耦
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ViewSpec:
    """
    用户请求输出的视野规范 (例如 4x, 20x, 10x 或指定物理尺寸)
    """
    name: str                           # 视图名称，如 "4x", "20x", "detail"
    pixel_dimensions: Tuple[int, int]   # 输出像素尺寸 (width, height)，如 (2257, 1310)
    magnification_approx: float = 4.0   # 近似倍率 (仅供展示与排版标签)
    physical_fov_um: Optional[Tuple[float, float]] = None # 物理视场尺寸 (width_um, height_um)


@dataclass
class StainRequirement:
    """染色需求"""
    stain_name: str
    is_required: bool = True            # True 为必须包含（缺失则无法 PASS），False 为可选


@dataclass
class UserGoal:
    """
    用户业务目标定义
    """
    reference_stain: str = "masson"
    stain_requirements: List[StainRequirement] = field(default_factory=lambda: [
        StainRequirement("HE", is_required=True),
        StainRequirement("Gram", is_required=False),
    ])
    requested_views: List[ViewSpec] = field(default_factory=lambda: [
        ViewSpec(name="4x", pixel_dimensions=(2257, 1310), magnification_approx=4.0),
        ViewSpec(name="20x", pixel_dimensions=(2257, 1310), magnification_approx=20.0),
    ])
    export_overlays: bool = True
    export_contact_sheets: bool = True
    dpi: Tuple[int, int] = (300, 300)

    def get_required_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements if r.is_required]

    def get_optional_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements if not r.is_required]

    def get_all_target_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements]
