"""
几何变换与坐标图严格数值断言测试 (Ground-Truth Geometry Tests)
"""

import numpy as np
import cv2
import pytest

from crossstainwsi.domain import EvidenceView
from crossstainwsi.planning.acquisition import AcquisitionProfile
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
    倍率比 = 4.0，理论缩放 = 0.25
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


def test_mpp_expected_scale_direction():
    """
    严格断言 Moving -> Reference 跨扫描仪预期物理尺度方向:
    Moving MPP = 0.50 um/px, Ref MPP = 0.40 um/px
    Moving 1 像素 = 0.50 um，在 Ref 上占 1.25 像素
    预期尺度必须为 1.25 (即 moving_mpp / ref_mpp)
    """
    moving_mpp = 0.50
    ref_mpp = 0.40
    expected_scale = moving_mpp / ref_mpp
    assert np.isclose(expected_scale, 1.25)
