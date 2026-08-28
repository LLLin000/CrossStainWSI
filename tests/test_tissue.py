import numpy as np
import cv2
from crossstainwsi.tissue.islands import TissueSegmenter


def test_tissue_segmenter():
    # 创建一个带有两个明显组织岛的合成图像
    canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)  # 白色玻片背景

    # 岛 1 (紫色 HE 组织色调)
    cv2.rectangle(canvas, (100, 100), (300, 300), (180, 50, 150), -1)
    # 岛 2 (蓝色 Masson 组织色调)
    cv2.rectangle(canvas, (600, 600), (900, 900), (200, 100, 20), -1)

    islands = TissueSegmenter.find_tissue_islands(canvas, min_area=1000)
    assert len(islands) == 2

    # 验证按面积排序
    assert islands[0].area >= islands[1].area

    # 验证坐标点命中选择
    selected = TissueSegmenter.select_island_by_coordinate(islands, (200, 200))
    x, y, w, h = selected.bbox
    assert x <= 200 <= x + w
    assert y <= 200 <= y + h
