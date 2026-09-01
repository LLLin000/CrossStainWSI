"""
归一化互信息局部优化救援匹配器 (NormalizedMutualInformationMatcher)
在候选先验区域内基于信息论熵值优化刚性/相似性几何残差 (解决跨模态灰度剧烈反差)
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from scipy.optimize import minimize

from crossstainwsi.domain import EvidenceRecord, FailureCode, QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import affine, apply_mat, h, rotation_matrix_2d, translation_matrix


def compute_nmi(img_a: np.ndarray, img_b: np.ndarray, bins: int = 32) -> float:
    """
    计算两张单通道图像之间的归一化互信息 (NMI):
    NMI(X, Y) = (H(X) + H(Y)) / H(X, Y)
    范围在 1.0 ~ 2.0 之间，值越高表示统计相关性越强
    """
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)

    a_flat = img_a.ravel()
    b_flat = img_b.ravel()

    # 如果图像几乎全常数，返回 1.0
    if np.std(a_flat) < 1e-4 or np.std(b_flat) < 1e-4:
        return 1.0

    hist_2d, _, _ = np.histogram2d(a_flat, b_flat, bins=bins, range=[[0, 256], [0, 256]])
    pxy = hist_2d / max(1e-10, float(np.sum(hist_2d)))

    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # 计算香农熵
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


class NormalizedMutualInformationMatcher(ImageMatcher):
    """
    基于归一化互信息的局部微调优化器 (需要候选先验引导，禁止无先验盲跑)
    """
    def __init__(
        self,
        max_translation_px: float = 40.0,
        max_rotation_deg: float = 10.0,
        bins: int = 32,
        min_nmi_score: float = 1.05,
    ):
        self.max_trans = max_translation_px
        self.max_rot = max_rotation_deg
        self.bins = bins
        self.min_nmi_score = min_nmi_score

    def match(
        self,
        moving: np.ndarray,
        fixed: np.ndarray,
        initial_guess_3x3: Optional[np.ndarray] = None,
    ) -> MatchResult:
        h_f, w_f = fixed.shape[:2]
        cx, cy = w_f / 2.0, h_f / 2.0

        g_m = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving
        g_f = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY) if fixed.ndim == 3 else fixed

        # 使用高斯平滑构建平滑的吸引盆地 (Continuous Basin of Attraction)
        g_m_smooth = cv2.GaussianBlur(g_m.astype(np.float32), (9, 9), 2.5)
        g_f_smooth = cv2.GaussianBlur(g_f.astype(np.float32), (9, 9), 2.5)

        # 目标损失函数 (最小化 -NMI)
        def loss_fn(params):
            dx, dy, d_theta = params
            if abs(dx) > self.max_trans or abs(dy) > self.max_trans or abs(d_theta) > self.max_rot:
                return 100.0

            r_m = rotation_matrix_2d((cx, cy), d_theta, scale=1.0)
            t_m = translation_matrix(dx, dy)
            m_curr = affine(t_m @ r_m)

            warped_m = cv2.warpAffine(
                g_m_smooth, m_curr, (w_f, h_f),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            nmi_val = compute_nmi(g_f_smooth, warped_m, bins=self.bins)
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
        mat_opt_3x3 = t_opt @ r_opt

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
                "dx": float(best_dx),
                "dy": float(best_dy),
                "d_theta": float(best_theta),
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
            matrix=affine(mat_opt_3x3) if is_valid else None,
            metrics=metrics,
            is_valid=is_valid,
            evidence=evidence,
            details={"nmi": best_nmi, "dx": float(best_dx), "dy": float(best_dy), "d_theta": float(best_theta)},
        )
