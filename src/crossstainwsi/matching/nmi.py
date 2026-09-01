"""
归一化互信息局部优化救援匹配器 (NormalizedMutualInformationMatcher)
支持输入候选先验 (initial_guess_3x3)、跨模态对比度反相、动态动态范围归一化与有效重叠掩模过滤
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from scipy.optimize import minimize

from crossstainwsi.domain import EvidenceRecord, FailureCode, QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import affine, apply_mat, h, rotation_matrix_2d, translation_matrix


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    鲁棒动态范围归一化 (支持 float32 核密度场 0~1、距离场与 uint16 荧光 0~65535)
    """
    if img.dtype == np.uint8:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    g_float = img.astype(np.float32)
    img_min, img_max = float(g_float.min()), float(g_float.max())

    if img_max - img_min < 1e-4:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    norm_uint8 = np.clip((g_float - img_min) / (img_max - img_min) * 255.0, 0, 255).astype(np.uint8)
    return norm_uint8


def compute_nmi_masked(
    img_a_uint8: np.ndarray,
    img_b_uint8: np.ndarray,
    mask_overlap: Optional[np.ndarray] = None,
    bins: int = 32,
) -> float:
    """
    在有效重叠区域内计算归一化互信息 (NMI)
    """
    if mask_overlap is not None:
        valid = mask_overlap > 0
        if valid.sum() < 50:
            return 1.0
        a_vals = img_a_uint8[valid]
        b_vals = img_b_uint8[valid]
    else:
        a_vals = img_a_uint8.ravel()
        b_vals = img_b_uint8.ravel()

    if np.std(a_vals) < 1e-4 or np.std(b_vals) < 1e-4:
        return 1.0

    hist_2d, _, _ = np.histogram2d(a_vals, b_vals, bins=bins, range=[[0, 256], [0, 256]])
    pxy = hist_2d / max(1e-10, float(np.sum(hist_2d)))

    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px = px[px > 1e-10]
    py = py[py > 1e-10]
    pxy = pxy[pxy > 1e-10]

    hx = -np.sum(px * np.log2(px))
    hy = -np.sum(py * np.log2(py))
    hxy = -np.sum(pxy * np.log2(pxy))

    if hxy < 1e-10:
        return 1.0

    nmi = float((hx + hy) / hxy)
    return nmi


compute_nmi = compute_nmi_masked


class NormalizedMutualInformationMatcher(ImageMatcher):
    """
    基于归一化互信息的局部微调优化器 (严格基于候选先验 basin 优化，避免边界假相关)
    """
    def __init__(
        self,
        max_translation_px: float = 40.0,
        max_rotation_deg: float = 10.0,
        bins: int = 32,
        min_nmi_score: float = 1.05,
        min_overlap_fraction: float = 0.20,
    ):
        self.max_trans = max_translation_px
        self.max_rot = max_rotation_deg
        self.bins = bins
        self.min_nmi_score = min_nmi_score
        self.min_overlap_fraction = min_overlap_fraction

    def match(
        self,
        moving: np.ndarray,
        fixed: np.ndarray,
        initial_guess_3x3: Optional[np.ndarray] = None,
    ) -> MatchResult:
        h_f, w_f = fixed.shape[:2]
        cx, cy = w_f / 2.0, h_f / 2.0

        g_m_uint8 = normalize_to_uint8(moving)
        g_f_uint8 = normalize_to_uint8(fixed)

        # 高斯平滑构建连续吸引盆地
        g_m_smooth = cv2.GaussianBlur(g_m_uint8.astype(np.float32), (9, 9), 2.5).astype(np.uint8)
        g_f_smooth = cv2.GaussianBlur(g_f_uint8.astype(np.float32), (9, 9), 2.5).astype(np.uint8)

        # 初始几何先验
        m_init_3x3 = h(initial_guess_3x3) if initial_guess_3x3 is not None else np.eye(3, dtype=np.float64)

        # 图像有效视场掩模 (计算画布重叠比例)
        support_mask_m = np.full((g_m_uint8.shape[0], g_m_uint8.shape[1]), 255, dtype=np.uint8)
        total_pixels = float(w_f * h_f)

        def loss_fn(params):
            dx, dy, d_theta = params
            if abs(dx) > self.max_trans or abs(dy) > self.max_trans or abs(d_theta) > self.max_rot:
                return 100.0

            r_m = rotation_matrix_2d((cx, cy), d_theta, scale=1.0)
            t_m = translation_matrix(dx, dy)
            m_res_3x3 = t_m @ r_m

            # 复合初始矩阵与残差矩阵: T_total = T_res @ T_init
            m_total_3x3 = m_res_3x3 @ m_init_3x3
            m_total_2x3 = affine(m_total_3x3)

            warped_m = cv2.warpAffine(
                g_m_smooth, m_total_2x3, (w_f, h_f),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_support_m = cv2.warpAffine(
                support_mask_m, m_total_2x3, (w_f, h_f),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            overlap_frac = float((warped_support_m > 0).sum()) / total_pixels

            # 抑制重叠过小的无效位移
            if overlap_frac < self.min_overlap_fraction:
                return 10.0

            nmi_val = compute_nmi_masked(g_f_smooth, warped_m, mask_overlap=warped_support_m, bins=self.bins)
            return -nmi_val

        init_p = [0.0, 0.0, 0.0]
        opt_res = minimize(
            loss_fn,
            init_p,
            method="Powell",
            options={"maxiter": 40, "ftol": 1e-4},
        )

        best_dx, best_dy, best_theta = opt_res.x
        best_nmi = -float(opt_res.fun)

        r_opt = rotation_matrix_2d((cx, cy), best_theta, scale=1.0)
        t_opt = translation_matrix(best_dx, best_dy)
        m_best_res_3x3 = t_opt @ r_opt
        m_final_3x3 = m_best_res_3x3 @ m_init_3x3

        is_valid = bool(
            best_nmi >= self.min_nmi_score
            and abs(best_dx) <= self.max_trans
            and abs(best_dy) <= self.max_trans
        )

        evidence = EvidenceRecord(
            backend="NMI",
            support_score=float(min(1.0, max(0.0, (best_nmi - 1.0) / 0.5))),
            scale=1.0,
            rotation_deg=float(best_theta),
            residual_dispersion_px=float(np.hypot(best_dx, best_dy)),
            is_independent_evidence=True,
            diagnostics={
                "best_nmi": best_nmi,
                "residual_dx": float(best_dx),
                "residual_dy": float(best_dy),
                "residual_d_theta": float(best_theta),
                "initial_guess_used": initial_guess_3x3 is not None,
                "iterations": opt_res.nit,
            },
        )

        metrics = QCMetrics(
            scale=1.0,
            rotation_deg=float(best_theta),
            method="NMI_Local_Optimizer",
            details={"nmi_score": best_nmi, "dx": float(best_dx), "dy": float(best_dy)},
        )

        return MatchResult(
            matrix=affine(m_final_3x3) if is_valid else None,
            metrics=metrics,
            is_valid=is_valid,
            evidence=evidence,
            details={"nmi": best_nmi, "dx": float(best_dx), "dy": float(best_dy), "d_theta": float(best_theta)},
        )
