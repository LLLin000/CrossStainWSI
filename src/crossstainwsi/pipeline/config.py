"""
流水线配置与预设定义 (PipelineConfig)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch


@dataclass
class PipelineConfig:
    """
    跨染色配准流水线全局配置
    """
    base_dir: Path = Path(r"E:\研究数据\骨科\切片扫描\2026-08-21")
    tiff_dir: Path = Path(r"E:\研究数据\骨科\切片扫描\tiff")
    output_dir: Path = Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi")

    reference_stain: str = "masson"
    moving_stains: List[str] = field(default_factory=lambda: ["HE", "Gram"])

    # 历史人工截图标注时发生水平镜像的样本列表 (在进入算法前自动翻转)
    mirrored_samples: Set[str] = field(default_factory=lambda: {"2-2w-1", "5-4w-2"})

    # 默认物理分辨率 MPP (um/px)
    default_mpp: float = 0.44243

    # 设备配置
    device: Optional[str] = None  # "cuda", "cpu" 或 None 自动探测

    # 算法开关
    enable_local_refine: bool = True
    enable_tissue_islands: bool = True
    save_overlays: bool = True
    save_contact_sheets: bool = True

    def get_torch_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
