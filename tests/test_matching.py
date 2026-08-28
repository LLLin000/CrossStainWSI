import numpy as np
import cv2
import pytest

from crossstainwsi.matching.phase_correlation import PhaseCorrelationMatcher
from crossstainwsi.matching.template import TemplateMatcher
from crossstainwsi.matching.sift import SiftMatcher
from crossstainwsi.transforms.geom import affine, translation_matrix


def test_phase_correlation_matcher():
    # 创建一个具有明显边缘的合成图像
    fixed = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(fixed, (150, 150), 50, (30, 30, 30), -1)

    # moving 向右平移 12 像素，向下平移 8 像素
    moving = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(moving, (162, 158), 50, (30, 30, 30), -1)

    matcher = PhaseCorrelationMatcher(max_displacement=60.0)
    res = matcher.match(moving, fixed)

    assert res.is_valid
    assert res.matrix is not None
    # 期望移动图像逆向平移 (-12, -8)
    dx = res.details["dx"]
    dy = res.details["dy"]
    assert abs(dx - (-12.0)) < 1.0
    assert abs(dy - (-8.0)) < 1.0


def test_template_matcher_translation():
    fixed = np.full((400, 400, 3), 255, dtype=np.uint8)
    # 在 (150, 150) 绘制图案
    cv2.rectangle(fixed, (150, 150), (250, 250), (50, 50, 50), -1)

    crop = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.rectangle(crop, (0, 0), (100, 100), (50, 50, 50), -1)

    matcher = TemplateMatcher(physical_scale=1.0, angle_range=(-10, 10), angle_step=5)
    res = matcher.match(crop, fixed)

    assert res.is_valid
    assert res.metrics.ncc_score is not None
    assert res.metrics.ncc_score > 0.8
