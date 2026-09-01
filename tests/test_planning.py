from pathlib import Path
from crossstainwsi.inventory.assets import ROIEvidence, SampleAssets, SlideAsset
from crossstainwsi.planning.goal import StainRequirement, UserGoal, ViewSpec
from crossstainwsi.planning.execution_plan import TaskType
from crossstainwsi.planning.planner import WorkflowPlanner
from crossstainwsi.review.states import ConfidenceTier


def test_planner_dual_scale():
    assets = SampleAssets(
        sample_id="sample-1",
        slides={
            "masson": SlideAsset(stain="masson", path=Path("s1-masson.kfb"), format="kfb"),
            "HE": SlideAsset(stain="HE", path=Path("s1-HE.kfb"), format="kfb"),
            "Gram": SlideAsset(stain="Gram", path=Path("s1-Gram.kfb"), format="kfb"),
        },
        roi_evidence=ROIEvidence(
            crop_4x_path=Path("s1-4x.tif"),
            crop_20x_path=Path("s1-20x.tif"),
        ),
    )
    planner = WorkflowPlanner()
    plan = planner.plan(assets)

    assert plan.task_type == TaskType.DUAL_SCALE_REPRODUCE
    assert plan.confidence_tier == ConfidenceTier.TIER_B_DUAL_SCALE
    assert plan.is_executable
    assert "HE" in plan.target_stains_available


def test_planner_native_roi():
    assets = SampleAssets(
        sample_id="sample-2",
        slides={
            "masson": SlideAsset(stain="masson", path=Path("s2-masson.kfb"), format="kfb"),
            "HE": SlideAsset(stain="HE", path=Path("s2-HE.kfb"), format="kfb"),
        },
        roi_evidence=ROIEvidence(
            native_center_l0=(15000.0, 20000.0),
            native_size_l0=(11000, 6500),
        ),
    )
    planner = WorkflowPlanner()
    plan = planner.plan(assets)

    assert plan.task_type == TaskType.NATIVE_ROI_MATCH
    assert plan.confidence_tier == ConfidenceTier.TIER_A_NATIVE
    assert plan.is_executable


def test_planner_missing_required_stain():
    assets = SampleAssets(
        sample_id="sample-3",
        slides={
            "masson": SlideAsset(stain="masson", path=Path("s3-masson.kfb"), format="kfb"),
            "Gram": SlideAsset(stain="Gram", path=Path("s3-Gram.kfb"), format="kfb"),
        },
        roi_evidence=ROIEvidence(crop_4x_path=Path("s3-4x.tif")),
    )
    # HE 是必选染色，Gram 是可选
    goal = UserGoal(
        stain_requirements=[
            StainRequirement("HE", is_required=True),
            StainRequirement("Gram", is_required=False),
        ]
    )
    planner = WorkflowPlanner(goal=goal)
    plan = planner.plan(assets)

    assert not plan.is_executable
    assert "HE" in plan.missing_required_stains
    assert "Required stains missing" in plan.block_reason


def test_view_spec_parsing():
    v4 = ViewSpec.from_string("4x")
    assert v4.magnification_approx == 4.0
    assert v4.name == "4x"

    v10 = ViewSpec.from_string("10X")
    assert v10.magnification_approx == 10.0
    assert v10.name == "10x"

    v40 = ViewSpec.from_string("40")
    assert v40.magnification_approx == 40.0

    goal = UserGoal.from_magnifications(["4x", "10x", "20x", "40x"], dpi=600)
    assert len(goal.requested_views) == 4
    assert goal.dpi == (600, 600)
