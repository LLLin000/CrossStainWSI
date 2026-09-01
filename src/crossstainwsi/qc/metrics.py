"""
质量控制 (QC) 图像评估指标计算 (带严格 NaN/Inf 过滤与组织信息量防护)
"""

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics
from crossstainwsi.transforms.geom import apply_mat


def clahe_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def compute_same_image_metrics(
    reference_bgr: np.ndarray,
    extracted_bgr: Optional[np.ndarray],
) -> QCMetrics:
    """
    计算同源或同视角两张图像之间的全方位质量一致性指标 (严格防护 NaN 与空白常数图)
    """
    if extracted_bgr is None:
        return QCMetrics(
            inliers=0,
            matches=0,
            inlier_ratio=0.0,
            spatial_coverage=0.0,
            median_reproj_error=999.0,
            scale=1.0,
            ncc_score=-1.0,
            mask_iou=0.0,
            background_agreement=0.0,
            edge_corr=-1.0,
            method="None",
            details={"tissue_fraction": 0.0, "is_constant": True},
        )

    if reference_bgr.shape[:2] != extracted_bgr.shape[:2]:
        extracted_bgr = cv2.resize(
            extracted_bgr,
            (reference_bgr.shape[1], reference_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    g_ref = clahe_gray(reference_bgr)
    g_ext = clahe_gray(extracted_bgr)

    # 1. 组织掩模 IoU 与前景组织占比 (防范纯白玻片通过门限)
    hsv_ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)
    hsv_ext = cv2.cvtColor(extracted_bgr, cv2.COLOR_BGR2HSV)
    mask_ref = ((g_ref < 245) | (hsv_ref[:, :, 1] > 20)).astype(np.uint8)
    mask_ext = ((g_ext < 245) | (hsv_ext[:, :, 1] > 20)).astype(np.uint8)

    tissue_fraction_ref = float(mask_ref.mean())
    tissue_fraction_ext = float(mask_ext.mean())
    tissue_fraction = min(tissue_fraction_ref, tissue_fraction_ext)

    union = np.logical_or(mask_ref, mask_ext).sum()
    mask_iou = float(np.logical_and(mask_ref, mask_ext).sum() / union) if union > 0 else 0.0
    bg_agreement = float((mask_ref == mask_ext).mean())

    # 2. NCC 灰度归一化互相关 (严格防护常数图导致的 NaN)
    std_ref = np.std(g_ref)
    std_ext = np.std(g_ext)

    if std_ref < 1e-4 or std_ext < 1e-4:
        ncc_val = -1.0
    else:
        try:
            corr_mat = np.corrcoef(g_ref.reshape(-1).astype(np.float32), g_ext.reshape(-1).astype(np.float32))
            raw_ncc = float(corr_mat[0, 1])
            ncc_val = raw_ncc if np.isfinite(raw_ncc) else -1.0
        except Exception:
            ncc_val = -1.0

    # 3. Canny 边缘结构相关性
    edge_ref = cv2.Canny(g_ref, 30, 80).astype(np.float32)
    edge_ext = cv2.Canny(g_ext, 30, 80).astype(np.float32)

    std_e_ref = np.std(edge_ref)
    std_e_ext = np.std(edge_ext)
    if std_e_ref < 1e-4 or std_e_ext < 1e-4:
        edge_corr = -1.0
    else:
        try:
            corr_mat = np.corrcoef(edge_ref.reshape(-1), edge_ext.reshape(-1))
            raw_edge = float(corr_mat[0, 1])
            edge_corr = raw_edge if np.isfinite(raw_edge) else -1.0
        except Exception:
            edge_corr = -1.0

    # 4. SIFT 特征自对齐重投影误差与内点数
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.005)
    kp_ref, des_ref = sift.detectAndCompute(g_ref, None)
    kp_ext, des_ext = sift.detectAndCompute(g_ext, None)

    if des_ref is None or des_ext is None:
        return QCMetrics(
            inliers=0,
            matches=0,
            inlier_ratio=0.0,
            spatial_coverage=0.0,
            median_reproj_error=999.0,
            scale=1.0,
            ncc_score=ncc_val,
            mask_iou=mask_iou,
            background_agreement=bg_agreement,
            edge_corr=edge_corr,
            method="Same_Image_QC",
            details={
                "tissue_fraction": tissue_fraction,
                "tissue_fraction_ref": tissue_fraction_ref,
                "tissue_fraction_ext": tissue_fraction_ext,
            },
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(des_ref, des_ext, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return QCMetrics(
            inliers=0,
            matches=len(good),
            inlier_ratio=0.0,
            spatial_coverage=0.0,
            median_reproj_error=999.0,
            scale=1.0,
            ncc_score=ncc_val,
            mask_iou=mask_iou,
            background_agreement=bg_agreement,
            edge_corr=edge_corr,
            method="Same_Image_QC",
            details={
                "tissue_fraction": tissue_fraction,
                "good_matches": len(good),
            },
        )

    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_ext[m.trainIdx].pt for m in good])

    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000
    )

    if matrix is None or mask is None:
        return QCMetrics(
            inliers=0,
            matches=len(good),
            inlier_ratio=0.0,
            spatial_coverage=0.0,
            median_reproj_error=999.0,
            scale=1.0,
            ncc_score=ncc_val,
            mask_iou=mask_iou,
            background_agreement=bg_agreement,
            edge_corr=edge_corr,
            method="Same_Image_QC",
            details={
                "tissue_fraction": tissue_fraction,
                "good_matches": len(good),
            },
        )

    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = float(n_inliers / len(good))

    proj = apply_mat(matrix, src[inlier_mask])
    errors = np.linalg.norm(dst[inlier_mask] - proj, axis=1)
    median_err = float(np.median(errors)) if len(errors) > 0 else 999.0
    if not np.isfinite(median_err):
        median_err = 999.0

    return QCMetrics(
        inliers=n_inliers,
        matches=len(good),
        inlier_ratio=inlier_ratio,
        spatial_coverage=0.0,
        median_reproj_error=median_err,
        scale=1.0,
        ncc_score=ncc_val,
        mask_iou=mask_iou,
        background_agreement=bg_agreement,
        edge_corr=edge_corr,
        method="Same_Image_QC",
        details={
            "tissue_fraction": tissue_fraction,
            "tissue_fraction_ref": tissue_fraction_ref,
            "tissue_fraction_ext": tissue_fraction_ext,
        },
    )
