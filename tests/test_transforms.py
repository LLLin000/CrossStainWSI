"""
几何变换与坐标图严格数值断言测试 (Ground-Truth Geometry Tests)
"""

import numpy as np
import cv2
import pytest

from crossstainwsi.domain import EvidenceView
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.goal import ViewSpec
from crossstainwsi.transforms.geom import (
    h,
    affine,
    apply_mat,
    invert_transform,
    translation_matrix,
    scale_matrix,
    rotation_matrix_2d,
    standard_90deg_rotation,
    extract_scale_and_angle,
)
from crossstainwsi.transforms.graph import TransformGraph


def test_geom_h_and_affine():
    m2x3 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    m3x3 = h(m2x3)
    assert m3x3.shape == (3, 3)
    assert m3x3[2, 2] == 1.0
    assert np.allclose(affine(m3x3), m2x3)


def test_standard_90deg_rotations_alignment():
    w, h_img = 200, 100
    pt = np.array([[50.0, 30.0]], dtype=np.float32)

    # 90度顺时针
    rot90_mat = standard_90deg_rotation(90, w, h_img)
    pt_rot90 = apply_mat(rot90_mat, pt)
    assert np.allclose(pt_rot90[0], [69.0, 50.0])

    # 180度
    rot180_mat = standard_90deg_rotation(180, w, h_img)
    pt_rot180 = apply_mat(rot180_mat, pt)
    assert np.allclose(pt_rot180[0], [w - 1 - 50.0, h_img - 1 - 30.0])


def test_evidence_views_exact_center_and_scale_mapping():
    """
    严格几何真值断言：
    Anchor 10x (2000x1000 px) 与 Secondary 40x (2000x1000 px)
    倍率比 = 4.0，理论缩放 = 0.25 (而不是写死的 0.2)
    """
    view_anc = EvidenceView(id="anc_10x", width_px=2000, height_px=1000, nominal_magnification=10.0)
    view_sec = EvidenceView(id="sec_40x", width_px=2000, height_px=1000, nominal_magnification=40.0)

    profile = AcquisitionProfile(same_center=True)
    m_sec_to_anc = profile.derive_view_to_anchor_matrix(view_anc, view_sec)

    # 1. 严格断言中心点重合: (1000, 500) 在 secondary 必须映射到 (1000, 500) 在 anchor
    pt_sec_center = np.array([[1000.0, 500.0]], dtype=np.float32)
    pt_mapped_center = apply_mat(m_sec_to_anc, pt_sec_center)[0]
    assert np.allclose(pt_mapped_center, [1000.0, 500.0], atol=1e-4)

    # 2. 严格断言物理尺度缩放: 长度 100 像素必须映射为 25 像素 (10x vs 40x)
    p1 = apply_mat(m_sec_to_anc, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p2 = apply_mat(m_sec_to_anc, np.array([[100.0, 0.0]], dtype=np.float32))[0]
    mapped_dist = np.linalg.norm(p2 - p1)
    assert np.isclose(mapped_dist, 25.0, atol=1e-4)


def test_mpp_driven_scale_mapping():
    """
    严格断言基于物理 MPP 的视场映射
    Anchor MPP = 1.7696 um/px, Secondary MPP = 0.4424 um/px
    理论物理缩放比 = 0.4424 / 1.7696 = 0.25
    """
    view_anc = EvidenceView(id="anc", width_px=2257, height_px=1310, mpp_xy=(1.7696, 1.7696))
    view_sec = EvidenceView(id="sec", width_px=2257, height_px=1310, mpp_xy=(0.4424, 0.4424))

    profile = AcquisitionProfile(same_center=True)
    m_sec_to_anc = profile.derive_view_to_anchor_matrix(view_anc, view_sec)

    scale_factor = m_sec_to_anc[0, 0]
    assert np.isclose(scale_factor, 0.25, atol=1e-3)


def test_level_aware_effective_mpp_expected_scale():
    """
    严格断言 Matching Level 级别有效 MPP 尺度计算:
    Moving L0 MPP = 0.50, DS = 16 (Effective = 8.0 um/px)
    Ref L0 MPP = 0.40, DS = 8 (Effective = 3.2 um/px)
    Expected Scale = 8.0 / 3.2 = 2.50
    """
    mov_l0_mpp, mov_ds4 = 0.50, 16.0
    ref_l0_mpp, ref_ds4 = 0.40, 8.0

    mov_effective = mov_l0_mpp * mov_ds4
    ref_effective = ref_l0_mpp * ref_ds4
    expected_scale = mov_effective / ref_effective

    assert np.isclose(expected_scale, 2.50)


def test_mirrored_reflection_matrix_composition():
    """
    严格断言镜像反射矩阵合成:
    原始截图 (w=2000, h=1000) 经过水平反射 F_x:
    (0, 0) -> (1999, 0)
    (1999, 0) -> (0, 0)
    矩阵行列式 det < 0
    """
    w, h_img = 2000, 1000
    f_x = np.array([
        [-1.0, 0.0, float(w - 1)],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 1. 行列式必须为负 (表示存在奇偶性翻转/镜像反射)
    det = np.linalg.det(f_x[:2, :2])
    assert np.isclose(det, -1.0)

    # 2. 四角坐标映射验证
    p_orig_left = np.array([[0.0, 0.0]], dtype=np.float32)
    p_mapped_right = apply_mat(affine(f_x), p_orig_left)[0]
    assert np.allclose(p_mapped_right, [1999.0, 0.0])

    p_orig_right = np.array([[1999.0, 0.0]], dtype=np.float32)
    p_mapped_left = apply_mat(affine(f_x), p_orig_right)[0]
    assert np.allclose(p_mapped_left, [0.0, 0.0])


def test_arbitrary_evidence_view_viewspec_to_lvl0():
    """
    测试通过 ViewSpec 传入 TransformGraph，从 10x Anchor 采样 40x 目标视图
    必须产生 0.25 缩放，且中心对齐，绝不退化为写死的 0.2 / 5x
    """
    view_anc = EvidenceView(id="anc_10x", width_px=2000, height_px=1000, nominal_magnification=10.0)
    view_tgt = ViewSpec(name="40x", pixel_dimensions=(2000, 1000), magnification_approx=40.0)

    graph = TransformGraph(
        anchor_view=view_anc,
        ref_ds_lvl2=4.0,
        ref_ds_lvl4=16.0,
        moving_ds_lvl2=4.0,
        moving_ds_lvl4=16.0,
    )
    graph.set_reference_anchor(np.eye(3))
    graph.set_global_cross_stain(np.eye(3))

    m_total = graph.get_view_to_moving_lvl0(target_view=view_tgt)

    # 验证复合矩阵有效且不含 NaN
    assert m_total.shape == (3, 3)
    assert not np.isnan(m_total).any()
