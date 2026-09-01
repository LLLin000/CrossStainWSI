"""
自适应工作流规划器 (WorkflowPlanner)
根据已有资产材料与用户需求自动生成最优最小可行执行计划
"""

from typing import Optional

from crossstainwsi.inventory.assets import SampleAssets
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.execution_plan import ExecutionPlan, TaskType
from crossstainwsi.planning.goal import UserGoal
from crossstainwsi.review.states import ConfidenceTier


class WorkflowPlanner:
    """
    负责将用户目标与实际资产智能撮合为 ExecutionPlan
    """
    def __init__(
        self,
        goal: Optional[UserGoal] = None,
        acquisition_profile: Optional[AcquisitionProfile] = None,
    ):
        self.goal = goal or UserGoal()
        self.profile = acquisition_profile or AcquisitionProfile()

    def plan(self, assets: SampleAssets) -> ExecutionPlan:
        sample_id = assets.sample_id
        # 优先使用从截图文件名自动识别出的基准染色 (如果该切片确实存在)
        inferred_stain = assets.roi_evidence.inferred_reference_stain
        if inferred_stain and assets.has_stain(inferred_stain):
            ref_stain = inferred_stain
        else:
            ref_stain = self.goal.reference_stain

        # 1. 检查参考切片是否存在
        ref_slide = assets.get_slide(ref_stain)
        if not ref_slide:
            return ExecutionPlan(
                sample_id=sample_id,
                task_type=TaskType.WHOLE_SLIDE_REGISTER,
                confidence_tier=ConfidenceTier.TIER_E_AMBIGUOUS,
                reference_stain=ref_stain,
                target_stains_available=[],
                is_executable=False,
                block_reason=f"Reference stain '{ref_stain}' WSI is missing from sample assets",
            )

        # 2. 检查目标染色是否存在并区分 必选 vs 可选
        available_targets = []
        missing_required = []
        missing_optional = []

        for req in self.goal.stain_requirements:
            if assets.has_stain(req.stain_name):
                available_targets.append(req.stain_name)
            else:
                if req.is_required:
                    missing_required.append(req.stain_name)
                else:
                    missing_optional.append(req.stain_name)

        if missing_required:
            is_executable = False
            block_reason = f"Required stains missing: {', '.join(missing_required)}"
        elif not available_targets:
            is_executable = False
            block_reason = "No target stains available for registration"
        else:
            is_executable = True
            block_reason = None

        # 3. 根据证据类型智能决断 TaskType 与 ConfidenceTier
        ev = assets.roi_evidence

        if ev.has_native_roi:
            task_type = TaskType.NATIVE_ROI_MATCH
            conf_tier = ConfidenceTier.TIER_A_NATIVE
            anchor_strategy = "Exact_Native_Coordinates"
            roi_desc = f"Native ROI center={ev.native_center_l0}, size={ev.native_size_l0}"
        elif len(ev.evidence_views) >= 2:
            task_type = TaskType.DUAL_SCALE_REPRODUCE
            conf_tier = ConfidenceTier.TIER_B_DUAL_SCALE
            v_anc = ev.get_primary_anchor_evidence()
            v_sec = ev.get_secondary_verification_evidence()
            anchor_strategy = f"Dual_Scale_{v_anc.nominal_magnification}x_and_{v_sec.nominal_magnification}x_Validation"
            roi_desc = f"Dual-scale imported evidence ({v_anc.nominal_magnification}x anchor + {v_sec.nominal_magnification}x verifier)"
        elif len(ev.evidence_views) == 1:
            v_single = ev.evidence_views[0]
            if v_single.nominal_magnification <= 10.0:
                task_type = TaskType.SINGLE_CROP_REPRODUCE
                conf_tier = ConfidenceTier.TIER_C_SINGLE_CROP
                anchor_strategy = f"Single_Scale_{v_single.nominal_magnification}x_Locator"
                roi_desc = f"Single imported evidence ({v_single.nominal_magnification}x)"
            else:
                task_type = TaskType.HIGH_MAG_ASSISTED
                conf_tier = ConfidenceTier.TIER_D_HIGH_MAG_ASSISTED
                anchor_strategy = f"High_Mag_{v_single.nominal_magnification}x_Assisted_Search"
                roi_desc = f"High-magnification evidence ({v_single.nominal_magnification}x) - assisted search"
        else:
            task_type = TaskType.WHOLE_SLIDE_REGISTER
            conf_tier = ConfidenceTier.TIER_A_NATIVE
            anchor_strategy = "No_ROI_Whole_Slide_Registration"
            roi_desc = "Whole slide registration (No ROI evidence)"
        return ExecutionPlan(
            sample_id=sample_id,
            task_type=task_type,
            confidence_tier=conf_tier,
            reference_stain=ref_stain,
            target_stains_available=available_targets,
            missing_required_stains=missing_required,
            missing_optional_stains=missing_optional,
            anchor_strategy=anchor_strategy,
            roi_source_description=roi_desc,
            requested_views=self.goal.requested_views,
            is_executable=is_executable,
            block_reason=block_reason,
            plan_details={
                "is_mirrored": ev.is_mirrored,
                "sampling_scale_ratio": self.profile.sampling_scale_ratio,
            },
        )
