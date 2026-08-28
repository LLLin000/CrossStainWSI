"""
组织区域分割与组织岛提取
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class TissueIsland:
    """
    单个独立的组织岛描述
    """
    island_id: int
    image: np.ndarray             # 组织岛 BGR 图像 (含 padding)
    bbox: Tuple[int, int, int, int] # (x, y, w, h)
    offset: Tuple[int, int]       # (x, y) 在原图中的左上角偏移
    centroid: Tuple[float, float] # (cx, cy) 质心坐标
    area: int                     # 组织像素面积

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


class TissueSegmenter:
    """
    基于颜色空间与形态学操作提取 WSI 中的各连通组织岛
    """
    @staticmethod
    def extract_tissue_mask(wsi_bgr: np.ndarray, gray_thresh: int = 240, sat_thresh: int = 15) -> np.ndarray:
        """
        提取前景组织二值掩模 (255 表示组织，0 表示背景空白玻片)
        """
        gray = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2HSV)
        mask = ((gray < gray_thresh) | (hsv[:, :, 1] > sat_thresh)).astype(np.uint8) * 255
        # 闭运算填补内部组织孔隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @classmethod
    def find_tissue_islands(
        cls,
        wsi_bgr: np.ndarray,
        min_area: int = 5000,
        pad_ratio: float = 0.10,
    ) -> List[TissueIsland]:
        """
        查找并分割出切片中所有大于 min_area 的组织岛
        """
        mask = cls.extract_tissue_mask(wsi_bgr)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        h_wsi, w_wsi = wsi_bgr.shape[:2]

        valid_indices = [
            i for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] >= min_area
        ]

        if not valid_indices:
            # 没有检测到显著独立组织岛时，将全切片作为一个组织岛
            return [
                TissueIsland(
                    island_id=0,
                    image=wsi_bgr,
                    bbox=(0, 0, w_wsi, h_wsi),
                    offset=(0, 0),
                    centroid=(w_wsi / 2.0, h_wsi / 2.0),
                    area=w_wsi * h_wsi,
                )
            ]

        islands = []
        for idx in valid_indices:
            bx = int(stats[idx, cv2.CC_STAT_LEFT])
            by = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[idx][0]), float(centroids[idx][1])

            pad_x = int(bw * pad_ratio)
            pad_y = int(bh * pad_ratio)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w_wsi, bx + bw + pad_x)
            y2 = min(h_wsi, by + bh + pad_y)

            island_crop = wsi_bgr[y1:y2, x1:x2].copy()
            islands.append(
                TissueIsland(
                    island_id=idx,
                    image=island_crop,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    offset=(x1, y1),
                    centroid=(cx, cy),
                    area=area,
                )
            )

        # 按面积从大到小排序
        islands.sort(key=lambda isl: isl.area, reverse=True)
        return islands

    @staticmethod
    def select_island_by_coordinate(
        islands: List[TissueIsland],
        coord: Tuple[float, float],
    ) -> TissueIsland:
        """
        根据指定的 (x, y) 坐标，查找包含该坐标的目标组织岛。如未命中则返回面积最大的组织岛。
        """
        cx, cy = coord
        for isl in islands:
            x, y, w, h = isl.bbox
            if x <= cx <= x + w and y <= cy <= y + h:
                return isl
        return islands[0]
