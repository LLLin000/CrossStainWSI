"""
基于 SIFT 特征的多角度旋转与 RANSAC 几何解算器
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import affine, apply_mat, extract_scale_and_angle, h


class SiftMatcher(ImageMatcher):
    """
    SIFT 特征提取与多角度旋转搜索匹配器
    """
    def __init__(
        self,
        nfeatures: int = 4000,
        contrast_threshold: float = 0.01,
        ratio_threshold: float = 0.78,
        ransac_threshold: float = 5.0,
        angle_step: int = 15,
        angle_range: Tuple[int, int] = (-60, 60),
    ):
        self.nfeatures = nfeatures
        self.contrast_threshold = contrast_threshold
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.angle_step = angle_step
        self.angle_range = angle_range
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess_gray(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def match(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
    ) -> MatchResult:
        g_m = self._preprocess_gray(moving_bgr)
        g_f = self._preprocess_gray(fixed_bgr)
        h_m, w_m = moving_bgr.shape[:2]

        sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            contrastThreshold=self.contrast_threshold,
        )
        kp_f, des_f = sift.detectAndCompute(g_f, None)
        if des_f is None:
            return MatchResult(
                matrix=None,
                metrics=QCMetrics(method="SIFT_RANSAC"),
                is_valid=False,
                details={"reason": "No features in fixed image"},
            )

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        best_inliers = 0
        best_mat = None
        best_angle = 0
        best_good = 0
        best_median_err = 999.0

        for ang in range(self.angle_range[0], self.angle_range[1] + 1, self.angle_step):
            m_rot = cv2.getRotationMatrix2D((w_m / 2.0, h_m / 2.0), ang, 1.0)
            rot_m = cv2.warpAffine(g_m, m_rot, (w_m, h_m))
            kp_m_r, des_m_r = sift.detectAndCompute(rot_m, None)
            if des_m_r is None:
                continue

            raw_matches = matcher.knnMatch(des_m_r, des_f, k=2)
            good = [m for m, n in raw_matches if m.distance < self.ratio_threshold * n.distance]
            if len(good) < 4:
                continue

            src_pts = np.float32([kp_m_r[m.queryIdx].pt for m in good])
            dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good])

            mat_r, inliers_mask = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=10000,
            )
            n_in = int(inliers_mask.sum()) if inliers_mask is not None else 0
            if n_in > best_inliers:
                best_inliers = n_in
                best_good = len(good)
                best_angle = ang
                # 复合旋转与仿射矩阵
                total_mat = affine(h(mat_r) @ h(m_rot))
                best_mat = total_mat

                # 计算重投影误差
                inliers_bool = inliers_mask.ravel().astype(bool)
                src_orig_pts = apply_mat(cv2.invertAffineTransform(m_rot), src_pts[inliers_bool])
                proj = apply_mat(total_mat, src_orig_pts)
                errs = np.linalg.norm(dst_pts[inliers_bool] - proj, axis=1)
                best_median_err = float(np.median(errs)) if len(errs) > 0 else 999.0

        if best_mat is not None and best_inliers >= 4:
            scale, angle_deg = extract_scale_and_angle(best_mat)
            inlier_ratio = float(best_inliers / max(1, best_good))
            metrics = QCMetrics(
                inliers=best_inliers,
                matches=best_good,
                inlier_ratio=inlier_ratio,
                median_reproj_error=best_median_err,
                scale=scale,
                rotation_deg=angle_deg,
                method="SIFT_RANSAC",
                details={"best_search_angle": best_angle},
            )
            return MatchResult(
                matrix=best_mat,
                metrics=metrics,
                is_valid=True,
                details={"best_angle": best_angle},
            )

        return MatchResult(
            matrix=None,
            metrics=QCMetrics(inliers=best_inliers, method="SIFT_RANSAC"),
            is_valid=False,
            details={"reason": f"Insufficient inliers: {best_inliers}"},
        )
