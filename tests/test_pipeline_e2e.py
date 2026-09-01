"""
端到端几何回归测试 (E2E Geometry & Scale Regression Tests)
涵盖 20x 证据输入 -> 多倍率视图提取、镜像双假设与匹配层有效 MPP 尺度计算
"""

import numpy as np
import pytest

from crossstainwsi.domain import EvidenceView, FailureCode
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.goal import ViewSpec
from crossstainwsi.transforms.geom import affine, apply_mat
from crossstainwsi.transforms.graph import TransformGraph


def test_20x_evidence_to_arbitrary_views_exact_scale_ratios():
    """
    真值断言：当用户输入 20x 证据截图 (2000x1000 px) 时，
    请求输出 4x, 10x, 20x, 40x 视图：
    - 4x 视图相比于 20x 证据，空间跨度放大 5.0 倍 (scale = 5.0)
    - 10x 视图相比于 20x 证据，空间跨度放大 2.0 倍 (scale = 2.0)
    - 20x 视图与 20x 证据完全等大 (scale = 1.0)
    - 40x 视图相比于 20x 证据，空间跨度缩小为 0.5 倍 (scale = 0.5)
    """
    view_anc_20x = EvidenceView(
        id="input_evidence_20x",
        width_px=2000,
        height_px=1000,
        nominal_magnification=20.0,
    )

    profile = AcquisitionProfile(same_center=True)

    # 1. 验证 4x 输出视图矩阵
    v4 = ViewSpec(name="4x", pixel_dimensions=(2000, 1000), magnification_approx=4.0)
    m4 = profile.derive_view_to_anchor_matrix(view_anc_20x, v4)
    # (0, 0) 与 (100, 0) 在 4x 视场中跨度映射到 20x 锚点像素坐标
    p1 = apply_mat(m4, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p2 = apply_mat(m4, np.array([[100.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p2 - p1), 500.0, atol=1e-3)

    # 2. 验证 10x 输出视图矩阵
    v10 = ViewSpec(name="10x", pixel_dimensions=(2000, 1000), magnification_approx=10.0)
    m10 = profile.derive_view_to_anchor_matrix(view_anc_20x, v10)
    p1 = apply_mat(m10, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p2 = apply_mat(m10, np.array([[100.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p2 - p1), 200.0, atol=1e-3)

    # 3. 验证 20x 输出视图矩阵 (恒等)
    v20 = ViewSpec(name="20x", pixel_dimensions=(2000, 1000), magnification_approx=20.0)
    m20 = profile.derive_view_to_anchor_matrix(view_anc_20x, v20)
    p1 = apply_mat(m20, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p2 = apply_mat(m20, np.array([[100.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p2 - p1), 100.0, atol=1e-3)

    # 4. 验证 40x 输出视图矩阵
    v40 = ViewSpec(name="40x", pixel_dimensions=(2000, 1000), magnification_approx=40.0)
    m40 = profile.derive_view_to_anchor_matrix(view_anc_20x, v40)
    p1 = apply_mat(m40, np.array([[0.0, 0.0]], dtype=np.float32))[0]
    p2 = apply_mat(m40, np.array([[100.0, 0.0]], dtype=np.float32))[0]
    assert np.isclose(np.linalg.norm(p2 - p1), 50.0, atol=1e-3)


def test_mirrored_parity_matrix_exact_mapping():
    """
    验证水平镜像反射 F_x 合成后的变换矩阵行列式 det < 0 且坐标映射精确对称
    """
    w, h = 2257, 1310
    f_x = np.array([
        [-1.0, 0.0, float(w - 1)],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    assert np.linalg.det(f_x[:2, :2]) < 0

    # 原始左上角 (0, 0) 映射到翻转后的右上角 (2256, 0)
    pt_orig = np.array([[0.0, 0.0]], dtype=np.float32)
    pt_mapped = apply_mat(affine(f_x), pt_orig)[0]
    assert np.allclose(pt_mapped, [2256.0, 0.0])

    # 原始中心点保持不变
    pt_center = np.array([[1128.0, 655.0]], dtype=np.float32)
    pt_center_mapped = apply_mat(affine(f_x), pt_center)[0]
    assert np.allclose(pt_center_mapped, [1128.0, 655.0])
