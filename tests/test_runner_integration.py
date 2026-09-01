"""
SampleRunner 与完整管线集成回归测试 (Pipeline Integration & Regression Tests)
"""

import numpy as np
import pytest

from crossstainwsi.domain import EvidenceView, FailureCode, RegistrationStatus
from crossstainwsi.inventory.assets import ROIEvidence, SampleAssets, SlideAsset
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.pipeline.sample_runner import SampleRunner
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.execution_plan import TaskType
from crossstainwsi.planning.goal import StainRequirement, UserGoal, ViewSpec
from crossstainwsi.planning.planner import WorkflowPlanner
from crossstainwsi.review.states import RunVerdict
from crossstainwsi.transforms.geom import affine, apply_mat
from crossstainwsi.transforms.graph import TransformGraph

def test_runner_native_roi_view_to_l0_matrix_generation():
    """
    测试 Native ROI 模式下不同输出倍率到 Level 0 像素坐标的映射矩阵:
    中心为 (15000, 20000)
    - 4x 视图 (2257x1310): Level-0 跨度为 2257 * 5 = 11285 px
    - 20x 视图 (2257x1310): Level-0 跨度为 2257 * 1 = 2257 px
    """
    runner = SampleRunner(config=PipelineConfig())
    center_l0 = (15000.0, 20000.0)
    size_l0 = (11285.0, 6550.0)

    v4 = ViewSpec(name="4x", pixel_dimensions=(2257, 1310), magnification_approx=4.0)
    v20 = ViewSpec(name="20x", pixel_dimensions=(2257, 1310), magnification_approx=20.0)

    m4 = runner._get_native_view_to_l0_matrix(center_l0, size_l0, v4, ref_l0_mpp=0.44243)
    m20 = runner._get_native_view_to_l0_matrix(center_l0, size_l0, v20, ref_l0_mpp=0.44243)

    # 1. 验证中心点均映射到 (15000, 20000)
    p_center = np.array([[2257.0 / 2.0, 1310.0 / 2.0]], dtype=np.float32)
    p4_mapped = apply_mat(affine(m4), p_center)[0]
    p20_mapped = apply_mat(affine(m20), p_center)[0]

    assert np.allclose(p4_mapped, [15000.0, 20000.0], atol=1e-3)
    assert np.allclose(p20_mapped, [15000.0, 20000.0], atol=1e-3)

    # 2. 验证 4x 视场总宽度在 Level 0 占 11285 像素 (5x 跨度)
    p_left = apply_mat(affine(m4), np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p_right = apply_mat(affine(m4), np.array([[2257.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p_right - p_left), 11285.0, atol=1e-3)

    # 3. 验证 20x 视场总宽度在 Level 0 占 2257 像素 (1x 跨度)
    p20_left = apply_mat(affine(m20), np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p20_right = apply_mat(affine(m20), np.array([[2257.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p20_right - p20_left), 2257.0, atol=1e-3)


def test_anchor_frame_local_refinement_scale_consistency():
    """
    验证 Local Refinement 在 Anchor Frame 下进行时，
    无论输入是 10x 还是 20x 证据，Anchor Frame 与基准参考图的物理尺寸严格一致
    """
    view_anchor_10x = EvidenceView(id="anchor_10x", width_px=1800, height_px=900, nominal_magnification=10.0)

    graph = TransformGraph(
        anchor_view=view_anchor_10x,
        ref_ds_lvl2=4.0,
        ref_ds_lvl4=16.0,
        moving_ds_lvl2=4.0,
        moving_ds_lvl4=16.0,
    )
    graph.set_reference_anchor(np.eye(3))
    graph.set_global_cross_stain(np.eye(3))

    # 设置局部微调平移 (+12, -8)
    mat_local_3x3 = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]])
    graph.set_local_refinement(mat_local_3x3)

    # 从 10x Anchor Frame 采样 40x 视图
    v40 = ViewSpec(name="40x", pixel_dimensions=(1800, 900), magnification_approx=40.0)
    m40 = graph.get_view_to_moving_lvl0(target_view=v40)
    # 10x -> 40x 缩放为 0.25, 加上 Level 4 到 Level 0 的 16x 金字塔全尺度，总复合尺度为 16.0 * 0.25 = 4.0
    scale_x = np.linalg.norm(m40[:2, 0])
    assert np.isclose(scale_x, 4.0, atol=1e-3)
def test_decoupled_multi_view_inventory_planning():
    """
    验证从含有 10x 证据的资产中自动规划并生成多倍率 (4x, 10x, 20x, 40x) 执行计划
    """
    ev = ROIEvidence()
    # 模拟发现 10x 截图证据
    ev.add_evidence_path(path=pytest.importorskip("pathlib").Path("sampleA-10x.tif"), nominal_mag=10.0)

    assets = SampleAssets(
        sample_id="sampleA",
        slides={
            "masson": SlideAsset(stain="masson", path=pytest.importorskip("pathlib").Path("sA-masson.kfb"), format="kfb"),
            "HE": SlideAsset(stain="HE", path=pytest.importorskip("pathlib").Path("sA-HE.kfb"), format="kfb"),
        },
        roi_evidence=ev,
    )

    goal = UserGoal.from_magnifications(["4x", "10x", "20x", "40x"], reference_stain="masson")
    planner = WorkflowPlanner(goal=goal)
    plan = planner.plan(assets)

    assert plan.task_type == TaskType.SINGLE_CROP_REPRODUCE
    assert len(plan.requested_views) == 4
    assert plan.is_executable
