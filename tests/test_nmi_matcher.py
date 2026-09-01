"""
归一化互信息 (NMI) 匹配器单元测试 (tests/test_nmi_matcher.py)
涵盖跨模态反相、float32 核密度场与 initial_guess 先验复合验证
"""

import cv2
import numpy as np
import pytest

from crossstainwsi.matching.nmi import NormalizedMutualInformationMatcher, compute_nmi


def test_compute_nmi_identical_and_shifted():
    img_a = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_a, (50, 50), 20, 200, -1)

    # 1. 相同图像的 NMI 必然最高
    nmi_self = compute_nmi(img_a, img_a)
    assert nmi_self > 1.20

    # 2. 带有轻微平移的图像
    img_b = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_b, (54, 52), 20, 200, -1)

    nmi_shifted = compute_nmi(img_a, img_b)
    assert nmi_shifted < nmi_self


def test_nmi_matcher_local_convergence():
    fixed = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(fixed, (50, 50), (100, 100), 220, -1)

    # moving 向右平移 6 像素，向下平移 4 像素
    moving = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(moving, (56, 54), (106, 104), 220, -1)

    matcher = NormalizedMutualInformationMatcher(max_translation_px=20.0, min_nmi_score=1.05)
    res = matcher.match(moving, fixed)

    assert bool(res.is_valid) is True
    assert res.matrix is not None
    dx = res.details["dx"]
    dy = res.details["dy"]
    assert abs(dx - (-6.0)) < 1.5
    assert abs(dy - (-4.0)) < 1.5
    assert res.evidence is not None
    assert res.evidence.backend == "NMI"


def test_nmi_matcher_cross_modal_contrast_inversion_and_float32():
    """
    跨模态极性反转测试：
    fixed 为暗底亮核 (如 DAPI 细胞核场 float32: 0~1)
    moving 为明底深色核 (如 H&E 苏木精场 uint8: 0~255) 并施加轻微平移
    """
    # 构造 fixed (float32 DAPI 荧光场)
    fixed_dapi = np.zeros((160, 160), dtype=np.float32)
    cv2.circle(fixed_dapi, (80, 80), 25, 1.0, -1)
    cv2.circle(fixed_dapi, (50, 50), 15, 0.8, -1)

    # 构造 moving (明场苏木精核, 灰度反相 + 平移: dx=+5, dy=-3)
    moving_he = np.full((160, 160), 240, dtype=np.uint8)
    cv2.circle(moving_he, (85, 77), 25, 30, -1)
    cv2.circle(moving_he, (55, 47), 15, 50, -1)

    matcher = NormalizedMutualInformationMatcher(max_translation_px=25.0, min_nmi_score=1.05)
    res = matcher.match(moving_he, fixed_dapi)

    assert bool(res.is_valid) is True
    assert res.matrix is not None
    dx = res.details["dx"]
    dy = res.details["dy"]
    # 期望求解逆平移: dx ≈ -5, dy ≈ +3
    assert abs(dx - (-5.0)) < 2.0
    assert abs(dy - 3.0) < 2.0


def test_nmi_matcher_with_initial_guess():
    """
    测试 initial_guess 先验复合:
    真实位移为 (dx=25, dy=15), 初始猜测为 (dx=20, dy=10)
    NMI 优化器只需在 Basin 内求解残差 (dx=5, dy=5) 即可收敛
    """
    fixed = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(fixed, (80, 80), 30, 220, -1)

    moving = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(moving, (105, 95), 30, 220, -1) # 真实平移 (+25, +15)

    # 初始粗略先验矩阵 T_init: 预平移 (-20, -10)
    t_init = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    matcher = NormalizedMutualInformationMatcher(max_translation_px=15.0, min_nmi_score=1.05)
    res = matcher.match(moving, fixed, initial_guess_3x3=t_init)

    assert bool(res.is_valid) is True
    assert res.matrix is not None
    # 复合总变换应当精确逼近 (-25, -15)
    total_dx = res.matrix[0, 2]
    total_dy = res.matrix[1, 2]
    assert abs(total_dx - (-25.0)) < 2.0
    assert abs(total_dy - (-15.0)) < 2.0
