import numpy as np
import cv2
import pytest

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
    # 测试 90/180/270 度齐次矩阵与 cv2.rotate 对图像坐标点的映射完全一致
    w, h_img = 200, 100
    img = np.zeros((h_img, w, 3), dtype=np.uint8)
    pt = np.array([[50.0, 30.0]], dtype=np.float32)

    # 90度顺时针
    rot90_mat = standard_90deg_rotation(90, w, h_img)
    pt_rot90 = apply_mat(rot90_mat, pt)
    # (x, y) 顺时针旋转90度后，新x = (h_img-1) - y = 99 - 30 = 69, 新y = x = 50
    assert np.allclose(pt_rot90[0], [69.0, 50.0])

    # 180度
    rot180_mat = standard_90deg_rotation(180, w, h_img)
    pt_rot180 = apply_mat(rot180_mat, pt)
    assert np.allclose(pt_rot180[0], [w - 1 - 50.0, h_img - 1 - 30.0])


def test_transform_graph_composite():
    graph = TransformGraph(
        crop4_size=(2257, 1310),
        crop20_size=(2257, 1310),
        ref_ds_lvl2=4.0,
        ref_ds_lvl4=16.0,
        moving_ds_lvl2=4.0,
        moving_ds_lvl4=16.0,
    )
    # 假设 4x 在 Ref L4 的变换是单位平移 (100, 200)
    mat_anchor_l4 = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 200.0]])
    graph.set_reference_anchor(mat_anchor_l4)

    # 假设 Moving 到 Ref L4 是纯旋转/平移
    mat_moving_to_ref = np.eye(3)
    graph.set_global_cross_stain(mat_moving_to_ref)

    # 局部残差设为平移 (5, -3)
    mat_local = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    graph.set_local_refinement(mat_local)

    # 验证 20x Level 0 复合矩阵计算
    m20_to_l0 = graph.get_crop20_to_moving_lvl0()
    assert m20_to_l0.shape == (3, 3)
    assert not np.isnan(m20_to_l0).any()


def test_transform_graph_evidence_views():
    from crossstainwsi.domain import EvidenceView
    view_anchor = EvidenceView(id="overview", width_px=2000, height_px=1000, nominal_magnification=4.0)
    view_detail = EvidenceView(id="detail", width_px=2000, height_px=1000, nominal_magnification=20.0)

    graph = TransformGraph(
        anchor_view=view_anchor,
        secondary_view=view_detail,
        ref_ds_lvl2=4.0,
        ref_ds_lvl4=16.0,
        moving_ds_lvl2=4.0,
        moving_ds_lvl4=16.0,
    )
    mat_anchor_l4 = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 80.0]])
    graph.set_reference_anchor(mat_anchor_l4)
    graph.set_global_cross_stain(np.eye(3))

    m_detail_l0 = graph.get_view_to_moving_lvl0(target_mag=20.0, base_mag=4.0)
    assert m_detail_l0.shape == (3, 3)
    assert not np.isnan(m_detail_l0).any()
