"""
用户目标与输出需求规范 (User Goal & View Specifications)
将输出要求与输入证据彻底解耦，支持直接指定倍率 (如 4x, 10x, 20x, 40x)
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple


@dataclass
class ViewSpec:
    """
    用户请求输出的视野规范 (例如 4x, 20x, 10x 或指定物理尺寸)
    """
    name: str                           # 视图名称，如 "4x", "20x", "10x", "detail"
    pixel_dimensions: Tuple[int, int]   # 输出像素尺寸 (width, height)，如 (2257, 1310)
    magnification_approx: float = 4.0   # 近似倍率 (例如 4.0, 10.0, 20.0, 40.0)
    physical_fov_um: Optional[Tuple[float, float]] = None # 物理视场尺寸 (width_um, height_um)

    @classmethod
    def from_string(
        cls,
        mag_str: str,
        default_size: Tuple[int, int] = (2257, 1310)
    ) -> "ViewSpec":
        """
        从字符串解析倍率规范 (如 '4x', '10x', '20x', '40x', '2.5x')
        """
        s = mag_str.strip().lower()
        match = re.match(r"^(\d+(?:\.\d+)?)\s*x?$", s)
        if match:
            mag_val = float(match.group(1))
            name = f"{int(mag_val) if mag_val.is_integer() else mag_val}x"
        else:
            mag_val = 4.0
            name = s

        return cls(
            name=name,
            pixel_dimensions=default_size,
            magnification_approx=mag_val,
        )


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
    export_contact_sheets: bool = False
    dpi: Tuple[int, int] = (300, 300)

    @classmethod
    def from_magnifications(
        cls,
        mags: List[str],
        reference_stain: str = "masson",
        target_stains: Optional[List[str]] = None,
        dpi: int = 300,
    ) -> "UserGoal":
        """
        从倍率列表 (如 ['4x', '20x', '10x']) 快速构建用户目标
        """
        views = [ViewSpec.from_string(m) for m in mags]
        targets = target_stains or ["HE", "Gram"]
        stain_reqs = [StainRequirement(s, is_required=(i == 0)) for i, s in enumerate(targets)]

        return cls(
            reference_stain=reference_stain,
            stain_requirements=stain_reqs,
            requested_views=views,
            dpi=(dpi, dpi),
        )

    def get_required_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements if r.is_required]

    def get_optional_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements if not r.is_required]

    def get_all_target_stains(self) -> List[str]:
        return [r.stain_name for r in self.stain_requirements]
