"""
合成几何扰动生成器 (Synthetic Ground Truth Perturbation Generator)
为配准算法提供具备精确解析解 (Ground Truth) 的几何基准测试用例
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.transforms.geom import affine, apply_mat, h, rotation_matrix_2d, scale_matrix, translation_matrix


@dataclass
class GroundTruthParams:
    """真实几何扰动参数"""
    angle_deg: float
    dx_px: float
    dy_px: float
    scale: float
    is_mirrored: bool
    center_px: Tuple[float, float]


@dataclass
class PerturbationCase:
    """单条合成基准用例"""
    case_id: str
    image_original: np.ndarray
    image_perturbed: np.ndarray
    matrix_gt_3x3: np.ndarray        # 真实映射矩阵: M(original -> perturbed)
    matrix_inverse_gt_3x3: np.ndarray # 真实逆映射: M(perturbed -> original)
    params: GroundTruthParams


class SyntheticPerturbationGenerator:
    """
    负责在真实切片/图像上施加精确可控的几何扰动
    """
    @staticmethod
    def generate(
        image_bgr: np.ndarray,
        angle_deg: float = 0.0,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
        scale: float = 1.0,
        is_mirrored: bool = False,
        case_id: str = "case_01",
        border_value: Tuple[int, int, int] = (255, 255, 255),
    ) -> PerturbationCase:
        h_img, w_img = image_bgr.shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0

        # 1. 基础几何矩阵分量
        # 旋转与中心平移
        r_mat = rotation_matrix_2d((cx, cy), angle_deg, scale=1.0)
        # 尺度缩放
        s_mat = translation_matrix(cx, cy) @ scale_matrix(scale, scale) @ translation_matrix(-cx, -cy)
        # 平移
        t_mat = translation_matrix(dx_px, dy_px)

        # 水平镜像反射矩阵 F_x: x' = w - 1 - x
        f_x = np.array([
            [-1.0, 0.0, float(w_img - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        if is_mirrored:
            # 扰动复合: T_gt = T_shift @ T_scale @ T_rot @ F_x
            m_gt_3x3 = t_mat @ s_mat @ r_mat @ f_x
        else:
            m_gt_3x3 = t_mat @ s_mat @ r_mat

        # 生成扰动图像 (对原始图像执行仿射变换)
        perturbed = cv2.warpAffine(
            image_bgr,
            affine(m_gt_3x3),
            (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        m_inv_gt_3x3 = np.linalg.inv(m_gt_3x3)

        params = GroundTruthParams(
            angle_deg=angle_deg,
            dx_px=dx_px,
            dy_px=dy_px,
            scale=scale,
            is_mirrored=is_mirrored,
            center_px=(cx, cy),
        )

        return PerturbationCase(
            case_id=case_id,
            image_original=image_bgr,
            image_perturbed=perturbed,
            matrix_gt_3x3=m_gt_3x3,
            matrix_inverse_gt_3x3=m_inv_gt_3x3,
            params=params,
        )

    @classmethod
    def generate_benchmark_suite(
        cls,
        image_bgr: np.ndarray,
        base_name: str = "synth",
    ) -> List[PerturbationCase]:
        """
        生成覆盖多角度、多平移、缩放与镜像的标准压力测试用例集
        """
        cases = []
        # Case 1: 纯微小位移
        cases.append(cls.generate(image_bgr, angle_deg=0.0, dx_px=15.0, dy_px=-20.0, case_id=f"{base_name}_small_shift"))
        # Case 2: 90 度正交旋转
        cases.append(cls.generate(image_bgr, angle_deg=90.0, dx_px=0.0, dy_px=0.0, case_id=f"{base_name}_rot_90"))
        # Case 3: 任意非正交旋转与尺度缩放
        cases.append(cls.generate(image_bgr, angle_deg=37.5, dx_px=30.0, dy_px=-15.0, scale=1.02, case_id=f"{base_name}_rot_37.5_scale"))
        # Case 4: 180 度反转 + 平移
        cases.append(cls.generate(image_bgr, angle_deg=180.0, dx_px=-25.0, dy_px=40.0, case_id=f"{base_name}_rot_180"))
        # Case 5: 纯水平镜像翻转
        cases.append(cls.generate(image_bgr, angle_deg=0.0, is_mirrored=True, case_id=f"{base_name}_pure_mirror"))
        # Case 6: 水平镜像 + 旋转 + 平移 (最强综合挑战)
        cases.append(cls.generate(image_bgr, angle_deg=45.0, dx_px=-20.0, dy_px=35.0, scale=0.98, is_mirrored=True, case_id=f"{base_name}_mirror_rot_45"))
        return cases
