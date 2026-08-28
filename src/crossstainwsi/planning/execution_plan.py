"""
执行计划模型 (ExecutionPlan & TaskType)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from crossstainwsi.planning.goal import ViewSpec
from crossstainwsi.review.states import ConfidenceTier


class TaskType(str, Enum):
    """
    自适应工作流任务类别
    """
    NATIVE_ROI_MATCH = "NATIVE_ROI_MATCH"               # 任务 A: 用户直接在 Reference WSI 上框选 ROI (0 锚点误差)
    SINGLE_CROP_REPRODUCE = "SINGLE_CROP_REPRODUCE"     # 任务 B: 复现已有的单张 4x 截图
    DUAL_SCALE_REPRODUCE = "DUAL_SCALE_REPRODUCE"       # 任务 C: 双尺度 4x + 20x 严格复现验证
    HIGH_MAG_ASSISTED = "HIGH_MAG_ASSISTED"             # 任务 D: 仅有 20x 截图 (辅助搜索)
    WHOLE_SLIDE_REGISTER = "WHOLE_SLIDE_REGISTER"       # 任务 E: 全切片坐标系对齐 (无需 ROI)


@dataclass
class ExecutionPlan:
    """
    由 WorkflowPlanner 生成的自包含执行计划
    在运行前清晰告知用户与引擎将如何处理材料
    """
    sample_id: str
    task_type: TaskType
    confidence_tier: ConfidenceTier
    reference_stain: str
    target_stains_available: List[str]
    missing_required_stains: List[str] = field(default_factory=list)
    missing_optional_stains: List[str] = field(default_factory=list)
    anchor_strategy: str = "Direct_Native"
    roi_source_description: str = "Native coordinates"
    requested_views: List[ViewSpec] = field(default_factory=list)
    is_executable: bool = True
    block_reason: Optional[str] = None
    plan_details: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        views_str = ", ".join([v.name for v in self.requested_views])
        lines = [
            f"=== Execution Plan for Sample [{self.sample_id}] ===",
            f"Task Type:       {self.task_type.value}",
            f"Confidence Tier: {self.confidence_tier.value}",
            f"Reference Stain: {self.reference_stain}",
            f"Target Stains:   {', '.join(self.target_stains_available)}",
            f"Anchor Strategy: {self.anchor_strategy}",
            f"ROI Source:      {self.roi_source_description}",
            f"Requested Views: {views_str}",
            f"Executable:      {self.is_executable}",
        ]
        if not self.is_executable:
            lines.append(f"BLOCK REASON:    {self.block_reason}")
        if self.missing_required_stains:
            lines.append(f"MISSING REQUIRED:{', '.join(self.missing_required_stains)}")
        if self.missing_optional_stains:
            lines.append(f"Missing Optional:{', '.join(self.missing_optional_stains)}")
        return "\n".join(lines)
