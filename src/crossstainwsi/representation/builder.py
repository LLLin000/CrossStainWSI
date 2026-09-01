"""
结构表征集合通用构建器 (RepresentationBuilder)
提供掩模分割、欧氏距离场 (EDT)、多尺度梯度金字塔与信息量计算
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class RepresentationBuilder:
    """
    负责从图像中提取几何轮廓、距离场、多尺度梯度与信息量指标
    """
    @staticmethod
    def compute_tissue_mask(
        img_bgr: np.ndarray,
        gray_thresh: int = 240,
        sat_thresh: int = 15,
    ) -> np.ndarray:
        """
        提取前景组织掩模 (255 表示组织前景, 0 表示玻片背景)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = ((gray < gray_thresh) | (hsv[:, :, 1] > sat_thresh)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def compute_coarse_contour(tissue_mask: np.ndarray) -> np.ndarray:
        """
        计算宏观组织外轮廓二值图
        """
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(tissue_mask, dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, 255, thickness=2)
        return contour_img

    @staticmethod
    def compute_distance_field(tissue_mask: np.ndarray) -> np.ndarray:
        """
        计算组织边缘欧氏距离变换场 (Euclidean Distance Transform, EDT)
        """
        # 对前景与背景分别计算 EDT 并合成双向距离场
        dist_inside = cv2.distanceTransform(tissue_mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - tissue_mask, cv2.DIST_L2, 5)
        signed_dist = dist_inside - dist_outside
        return signed_dist.astype(np.float32)

    @staticmethod
    def compute_gradient_pyramid(
        gray_img: np.ndarray,
        scales: Tuple[int, ...] = (1, 2, 4),
    ) -> Tuple[np.ndarray, ...]:
        """
        计算多尺度 Sobel 结构梯度幅值
        """
        gradients = []
        g_float = gray_img.astype(np.float32) / 255.0
        for s in scales:
            if s > 1:
                blurred = cv2.GaussianBlur(g_float, (0, 0), sigmaX=float(s))
            else:
                blurred = g_float
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            gradients.append(mag)
        return tuple(gradients)

    @staticmethod
    def compute_informativeness(
        img_gray: np.ndarray,
        tissue_mask: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算图像的结构信息量、组织占比与梯度能量
        """
        tissue_fraction = float((tissue_mask > 0).mean())

        # 梯度能量
        gx = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 0, 1)
        grad_energy = float(np.mean(gx * gx + gy * gy))

        # 灰度熵 (Entropy)
        hist = cv2.calcHist([img_gray], [0], tissue_mask if tissue_fraction > 0.05 else None, [256], [0, 256])
        hist_prob = hist.ravel() / max(1.0, float(hist.sum()))
        hist_prob = hist_prob[hist_prob > 1e-7]
        entropy = float(-np.sum(hist_prob * np.log2(hist_prob))) if len(hist_prob) > 0 else 0.0

        return {
            "tissue_fraction": tissue_fraction,
            "gradient_energy": grad_energy,
            "entropy": entropy,
        }
