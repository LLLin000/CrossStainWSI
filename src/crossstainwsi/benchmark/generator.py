"""
合成几何扰动生成器 (Synthetic Ground Truth Perturbation Generator)
为配准算法提供具备精确解析解的正例与不可配准的负例测试用例 (Positive & Negative Test Cases)
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
    expected_matchable: bool = True  # True 表示理论上存在真实几何对应，False 表示负例 (应被拒绝)


class SyntheticPerturbationGenerator:
    """
    负责在真实切片/图像上施加精确可控的几何扰动并构建正/负例评测套件
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

        r_mat = rotation_matrix_2d((cx, cy), angle_deg, scale=1.0)
        s_mat = translation_matrix(cx, cy) @ scale_matrix(scale, scale) @ translation_matrix(-cx, -cy)
        t_mat = translation_matrix(dx_px, dy_px)

        f_x = np.array([
            [-1.0, 0.0, float(w_img - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        if is_mirrored:
            m_gt_3x3 = t_mat @ s_mat @ r_mat @ f_x
        else:
            m_gt_3x3 = t_mat @ s_mat @ r_mat

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
            expected_matchable=True,
        )

    @staticmethod
    def generate_negative_case(
        image_original: np.ndarray,
        image_negative: np.ndarray,
        case_id: str = "negative_non_match",
    ) -> PerturbationCase:
        """
        生成非同源负例用例 (理论不存在几何重合，用于测试拒识与真实假阳性率 False Accept Rate)
        """
        h_img, w_img = image_original.shape[:2]
        return PerturbationCase(
            case_id=case_id,
            image_original=image_original,
            image_perturbed=image_negative,
            matrix_gt_3x3=np.eye(3),
            matrix_inverse_gt_3x3=np.eye(3),
            params=GroundTruthParams(0.0, 0.0, 0.0, 1.0, False, (w_img / 2.0, h_img / 2.0)),
            expected_matchable=False,
        )

    @staticmethod
    def generate_blank_case(
        image_original: np.ndarray,
        case_id: str = "negative_blank_background",
    ) -> PerturbationCase:
        """
        生成纯白空白玻片负例 (用于测试退化无组织信息下的安全拒识)
        """
        h_img, w_img = image_original.shape[:2]
        blank_img = np.full_like(image_original, 255)
        return PerturbationCase(
            case_id=case_id,
            image_original=image_original,
            image_perturbed=blank_img,
            matrix_gt_3x3=np.eye(3),
            matrix_inverse_gt_3x3=np.eye(3),
            params=GroundTruthParams(0.0, 0.0, 0.0, 1.0, False, (w_img / 2.0, h_img / 2.0)),
            expected_matchable=False,
        )

    @classmethod
    def generate_benchmark_suite(
        cls,
        image_bgr: np.ndarray,
        base_name: str = "synth",
        include_negatives: bool = True,
    ) -> List[PerturbationCase]:
        """
        生成覆盖多角度、多平移、缩放、镜像以及不可配准负例的标准压力测试套件
        """
        cases = []
        # 正例组 1: 纯微小位移
        cases.append(cls.generate(image_bgr, angle_deg=0.0, dx_px=15.0, dy_px=-20.0, case_id=f"{base_name}_pos_small_shift"))
        # 正例组 2: 90 度正交旋转
        cases.append(cls.generate(image_bgr, angle_deg=90.0, dx_px=0.0, dy_px=0.0, case_id=f"{base_name}_pos_rot_90"))
        # 正例组 3: 任意非正交旋转与尺度缩放
        cases.append(cls.generate(image_bgr, angle_deg=37.5, dx_px=30.0, dy_px=-15.0, scale=1.02, case_id=f"{base_name}_pos_rot_37.5_scale"))
        # 正例组 4: 180 度反转 + 平移
        cases.append(cls.generate(image_bgr, angle_deg=180.0, dx_px=-25.0, dy_px=40.0, case_id=f"{base_name}_pos_rot_180"))
        # 正例组 5: 纯水平镜像翻转
        cases.append(cls.generate(image_bgr, angle_deg=0.0, is_mirrored=True, case_id=f"{base_name}_pos_pure_mirror"))
        # 正例组 6: 水平镜像 + 旋转 + 平移
        cases.append(cls.generate(image_bgr, angle_deg=45.0, dx_px=-20.0, dy_px=35.0, scale=0.98, is_mirrored=True, case_id=f"{base_name}_pos_mirror_rot_45"))

        if include_negatives:
            # 负例 1: 纯空白背景玻片 (应被正确 ABSTAIN 拒绝)
            cases.append(cls.generate_blank_case(image_bgr, case_id=f"{base_name}_neg_blank"))
            # 负例 2: 随机高斯噪声无结构图像 (应被正确 ABSTAIN 拒绝)
            noise_img = np.random.randint(50, 200, image_bgr.shape, dtype=np.uint8)
            cases.append(cls.generate_negative_case(image_bgr, noise_img, case_id=f"{base_name}_neg_noise"))

        return cases
