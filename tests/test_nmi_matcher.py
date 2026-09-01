"""
归一化互信息 (NMI) 匹配器单元测试 (tests/test_nmi_matcher.py)
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
    # 验证优化出的逆平移 (dx ≈ -6, dy ≈ -4)
    dx = res.details["dx"]
    dy = res.details["dy"]
    assert abs(dx - (-6.0)) < 1.5
    assert abs(dy - (-4.0)) < 1.5
    assert res.evidence is not None
    assert res.evidence.backend == "NMI"
