"""
几何变换工具与齐次矩阵操作
"""

import math
from typing import Tuple, Union
import cv2
import numpy as np


def h(m: np.ndarray) -> np.ndarray:
    """
    将 2x3 仿射矩阵或 3x3 矩阵提升为标准 3x3 齐次矩阵 (float64)
    """
    arr = np.asarray(m, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (2, 3):
        return np.vstack([arr, [0.0, 0.0, 1.0]])
    raise ValueError(f"Expected shape (2, 3) or (3, 3), got {arr.shape}")


def affine(m3: np.ndarray) -> np.ndarray:
    """
    从 3x3 齐次矩阵截取前两行，返回 OpenCV 标准 2x3 仿射矩阵 (float32)
    """
    arr = np.asarray(m3, dtype=np.float32)
    return arr[:2, :3]


def apply_mat(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    使用 2x3 或 3x3 矩阵对点集进行坐标变换
    pts 形状为 (N, 2)
    返回 (N, 2) float32
    """
    pts_arr = np.asarray(pts, dtype=np.float32)
    if pts_arr.ndim == 1:
        pts_arr = pts_arr[None, :]
    m_2x3 = affine(m)
    transformed = cv2.transform(pts_arr[None, :, :], m_2x3)
    return transformed[0]


def invert_transform(m: np.ndarray) -> np.ndarray:
    """
    求变换矩阵的逆矩阵 (返回 3x3 float64)
    """
    m3 = h(m)
    return np.linalg.inv(m3)


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """
    构造平移 3x3 齐次矩阵
    """
    return np.array([
        [1.0, 0.0, float(dx)],
        [0.0, 1.0, float(dy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """
    构造缩放 3x3 齐次矩阵
    """
    return np.array([
        [float(sx), 0.0, 0.0],
        [0.0, float(sy), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotation_matrix_2d(center: Tuple[float, float], angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """
    围绕中心点旋转的 3x3 齐次矩阵
    """
    m_2x3 = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return h(m_2x3)


def standard_90deg_rotation(angle: int, width: int, height: int) -> np.ndarray:
    """
    获取与 cv2.rotate() 完全对齐的标准 0/90/180/270 度旋转 3x3 矩阵
    """
    ang = angle % 360
    if ang == 0:
        return np.eye(3, dtype=np.float64)
    elif ang == 90:
        return np.array([
            [0.0, -1.0, float(height - 1)],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    elif ang == 180:
        return np.array([
            [-1.0, 0.0, float(width - 1)],
            [0.0, -1.0, float(height - 1)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    elif ang == 270:
        return np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, float(width - 1)],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Standard orthogonal angle must be 0, 90, 180, or 270, got {angle}")


def extract_scale_and_angle(m: np.ndarray) -> Tuple[float, float]:
    """
    从仿射变换矩阵中分解出各向同性尺度 scale 与 旋转角度 angle (deg)
    """
    m3 = h(m)
    a = m3[0, 0]
    b = m3[1, 0]
    scale = math.sqrt(a * a + b * b)
    angle_rad = math.atan2(b, a)
    angle_deg = math.degrees(angle_rad)
    return float(scale), float(angle_deg)
