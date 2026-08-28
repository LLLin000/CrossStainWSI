"""
基于 LoFTR 的跨染色深度形态学匹配器 (带 Letterbox 保持宽高比与四阶 QC 统计)
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch

try:
    import kornia as K
    import kornia.feature as KF
except ImportError:
    K = None
    KF = None

from crossstainwsi.domain import QCMetrics
from crossstainwsi.matching.base import ImageMatcher, MatchResult
from crossstainwsi.transforms.geom import affine, apply_mat, extract_scale_and_angle, h


def letterbox_image(
    img_bgr: np.ndarray,
    target_size: int = 640
) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    保持宽高比等比缩放并居中填充到 target_size x target_size
    返回: (canvas, scale, (pad_x, pad_y), (new_w, new_h))
    """
    h_orig, w_orig = img_bgr.shape[:2]
    scale = target_size / max(h_orig, w_orig)
    new_w = max(1, round(w_orig * scale))
    new_h = max(1, round(h_orig * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y), (new_w, new_h)


class LoFTRMatcher(ImageMatcher):
    """
    LoFTR 深度形态匹配器
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        target_size: int = 640,
        confidence_thresh: float = 0.38,
        ransac_thresh: float = 8.0,
    ):
        if KF is None:
            raise RuntimeError("kornia is required for LoFTRMatcher. Please install kornia.")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.confidence_thresh = confidence_thresh
        self.ransac_thresh = ransac_thresh
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def match(
        self,
        moving_bgr: np.ndarray,
        fixed_bgr: np.ndarray,
    ) -> MatchResult:
        box_m, scale_m, (pad_m_x, pad_m_y), (rw_m, rh_m) = letterbox_image(moving_bgr, self.target_size)
        box_f, scale_f, (pad_f_x, pad_f_y), (rw_f, rh_f) = letterbox_image(fixed_bgr, self.target_size)

        g_m = self.clahe.apply(cv2.cvtColor(box_m, cv2.COLOR_BGR2GRAY))
        g_f = self.clahe.apply(cv2.cvtColor(box_f, cv2.COLOR_BGR2GRAY))

        t1 = K.image.image_to_tensor(g_m, keepdim=False).float().to(self.device) / 255.0
        t2 = K.image.image_to_tensor(g_f, keepdim=False).float().to(self.device) / 255.0

        with torch.no_grad():
            out = self._model({"image0": t1, "image1": t2})

        pts0 = out["keypoints0"].cpu().numpy()
        pts1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

        # 过滤处于填充区域之外的点
        valid_m = (
            (pts0[:, 0] >= pad_m_x) & (pts0[:, 0] < pad_m_x + rw_m) &
            (pts0[:, 1] >= pad_m_y) & (pts0[:, 1] < pad_m_y + rh_m)
        )
        valid_f = (
            (pts1[:, 0] >= pad_f_x) & (pts1[:, 0] < pad_f_x + rw_f) &
            (pts1[:, 1] >= pad_f_y) & (pts1[:, 1] < pad_f_y + rh_f)
        )
        valid = valid_m & valid_f & (conf > self.confidence_thresh)

        pts0 = pts0[valid]
        pts1 = pts1[valid]

        if len(pts0) < 4:
            return MatchResult(
                matrix=None,
                metrics=QCMetrics(method="LoFTR"),
                is_valid=False,
                details={"reason": f"Too few raw valid points: {len(pts0)}"},
            )

        # 反算回原图坐标系
        pts0_orig = pts0.copy()
        pts1_orig = pts1.copy()
        pts0_orig[:, 0] = (pts0[:, 0] - pad_m_x) / scale_m
        pts0_orig[:, 1] = (pts0[:, 1] - pad_m_y) / scale_m
        pts1_orig[:, 0] = (pts1[:, 0] - pad_f_x) / scale_f
        pts1_orig[:, 1] = (pts1[:, 1] - pad_f_y) / scale_f

        # 严格使用 similarity (estimateAffinePartial2D: 旋转+等比缩放+平移，禁止剪切形变)
        mat, mask = cv2.estimateAffinePartial2D(
            pts0_orig,
            pts1_orig,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=10000,
            confidence=0.999,
        )

        inliers_mask = (mask.ravel() == 1) if mask is not None else np.zeros(len(pts0_orig), dtype=bool)
        n_in = int(inliers_mask.sum())
        inlier_ratio = float(n_in / len(pts0_orig)) if len(pts0_orig) > 0 else 0.0

        if n_in >= 4 and mat is not None:
            p1_in = pts1_orig[inliers_mask]
            # 计算 4x4 网格空间覆盖率 (0.0 ~ 1.0)
            grid_x = np.clip((p1_in[:, 0] / max(1, fixed_bgr.shape[1]) * 4).astype(int), 0, 3)
            grid_y = np.clip((p1_in[:, 1] / max(1, fixed_bgr.shape[0]) * 4).astype(int), 0, 3)
            occupied = len(set(zip(grid_x, grid_y)))
            spatial_coverage = float(occupied / 16.0)

            p0_in = pts0_orig[inliers_mask]
            p0_trans = apply_mat(mat, p0_in)
            reproj_errors = np.linalg.norm(p1_in - p0_trans, axis=1)
            median_reproj_error = float(np.median(reproj_errors))
            scale, angle_deg = extract_scale_and_angle(mat)

            metrics = QCMetrics(
                inliers=n_in,
                matches=len(pts0_orig),
                inlier_ratio=inlier_ratio,
                spatial_coverage=spatial_coverage,
                median_reproj_error=median_reproj_error,
                scale=scale,
                rotation_deg=angle_deg,
                method="LoFTR",
                details={"confidence_threshold": self.confidence_thresh},
            )
            return MatchResult(
                matrix=mat,
                metrics=metrics,
                is_valid=True,
                details={
                    "inliers": n_in,
                    "inlier_ratio": inlier_ratio,
                    "spatial_coverage": spatial_coverage,
                    "scale": scale,
                },
            )

        return MatchResult(
            matrix=None,
            metrics=QCMetrics(
                inliers=n_in,
                matches=len(pts0_orig),
                inlier_ratio=inlier_ratio,
                method="LoFTR",
            ),
            is_valid=False,
            details={"reason": f"RANSAC rejected with {n_in} inliers"},
        )
