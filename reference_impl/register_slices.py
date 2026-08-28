"""
================================================================================
多模态骨科病理切片自动配准与出版级截图提取引擎 (Cross-Stain WSI Registration Engine)
版本：v6.0 (双层纯形态特征流 / 全局LoFTR+局部LoFTR / 组织岛隔离 / 全刚性保真 / 300 DPI)
================================================================================
"""

import gc
import json
import time
import warnings
from pathlib import Path

import cv2
import kfbslide
import numpy as np
from PIL import Image, ImageDraw
import SimpleITK as sitk
import torch
import kornia as K
import kornia.feature as KF

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== 全局配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(r"E:\研究数据\骨科\切片扫描\2026-08-21")
TIFF_DIR = Path(r"E:\研究数据\骨科\切片扫描\tiff")
OUT_DIR = Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi")


# ==================== 几何变换工具函数 ====================
def h(m: np.ndarray) -> np.ndarray:
    if m.shape == (3, 3):
        return m.astype(np.float64)
    return np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float64)


def affine(m3: np.ndarray) -> np.ndarray:
    return m3[:2].astype(np.float32)


def apply_mat(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.transform(np.asarray(pts, np.float32)[None, :, :], m.astype(np.float32))[0]


def find_file_case_insensitive(directory: Path, pattern_prefix: str, suffix: str) -> Path | None:
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    target = (pattern_prefix + suffix).lower()
    for p in dir_path.iterdir():
        if p.name.lower() == target:
            return p
    for p in dir_path.glob(f"*{suffix}"):
        if pattern_prefix.lower() in p.name.lower():
            return p
    return None


def letterbox_image(img_bgr: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    h_orig, w_orig = img_bgr.shape[:2]
    scale = target_size / max(h_orig, w_orig)
    new_w, new_h = max(1, round(w_orig * scale)), max(1, round(h_orig * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y), (new_w, new_h)


# ==================== 核心配准类 ====================
class CrossStainRegistrar:
    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[Init] 加载 LoFTR 深度形态匹配模型 (Device: {self.device})...")
        self.loftr = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def read_wsi_level(self, kfb_path: Path, level: int = 4):
        with kfbslide.OpenSlide(str(kfb_path)) as s:
            dims = s.level_dimensions[level]
            ds = float(s.level_downsamples[level])
            im = s.read_region((0, 0), level, dims).convert("RGB")
            l0_dims = s.dimensions
            mpp_raw_x = s.properties.get("openslide.mpp-x")
            mpp_raw_y = s.properties.get("openslide.mpp-y")
            if mpp_raw_x is not None and mpp_raw_y is not None:
                mpp_x, mpp_y = float(mpp_raw_x), float(mpp_raw_y)
                mpp_source = "metadata"
            else:
                mpp_x, mpp_y = 0.44243, 0.44243
                mpp_source = "configured_override"
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), dims, ds, l0_dims, (mpp_x, mpp_y), mpp_source

    def locate_crop_in_reference(self, ref_kfb_path: Path, crop4_path: Path) -> dict:
        crop_img = Image.open(crop4_path).convert("RGB")
        crop_bgr = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2BGR)
        crop_h, crop_w = crop_bgr.shape[:2]

        with kfbslide.OpenSlide(str(ref_kfb_path)) as slide:
            lvl4 = 4
            ds4 = float(slide.level_downsamples[lvl4])
            dims4 = slide.level_dimensions[lvl4]
            lvl4_img = slide.read_region((0, 0), lvl4, dims4).convert("RGB")
            lvl4_bgr = cv2.cvtColor(np.asarray(lvl4_img), cv2.COLOR_RGB2BGR)
            lvl0_dims = slide.dimensions
            ds2 = float(slide.level_downsamples[2])

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_c = clahe.apply(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        g_l4 = clahe.apply(cv2.cvtColor(lvl4_bgr, cv2.COLOR_BGR2GRAY))

        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02, edgeThreshold=15)
        kp_c, des_c = sift.detectAndCompute(g_c, None)
        kp_l4, des_l4 = sift.detectAndCompute(g_l4, None)

        best_mat = None
        best_inliers = 0
        localization_method = "SIFT_RANSAC"
        best_ncc = None

        for ang in range(-60, 61, 15):
            M_rot = cv2.getRotationMatrix2D((crop_w / 2.0, crop_h / 2.0), ang, 1.0)
            rot_c = cv2.warpAffine(g_c, M_rot, (crop_w, crop_h))
            kc_r, dc_r = sift.detectAndCompute(rot_c, None)
            if dc_r is None:
                continue
            raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dc_r, des_l4, k=2)
            good = [m for m, n in raw if m.distance < 0.78 * n.distance]
            if len(good) >= 4:
                src = np.float32([kc_r[m.queryIdx].pt for m in good])
                dst = np.float32([kp_l4[m.trainIdx].pt for m in good])
                mat_r, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                inl = int(mask.sum()) if mask is not None else 0
                if inl > best_inliers:
                    best_inliers = inl
                    best_mat = (np.vstack([mat_r, [0, 0, 1]]) @ h(M_rot))[:2]

        if best_mat is None or best_inliers < 20:
            physical_scale = 5.0 / ds4
            scaled_size = (round(crop_w * physical_scale), round(crop_h * physical_scale))
            scaled_crop = cv2.resize(crop_bgr, scaled_size, interpolation=cv2.INTER_AREA)
            best_template = (-1.0, None)
            for angle in range(-180, 180, 5):
                rotation = cv2.getRotationMatrix2D((scaled_size[0] / 2.0, scaled_size[1] / 2.0), angle, 1.0)
                rotated = cv2.warpAffine(scaled_crop, rotation, scaled_size, borderValue=(255, 255, 255))
                template = clahe.apply(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
                score_map = cv2.matchTemplate(g_l4, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(score_map)
                if score > best_template[0]:
                    scale_matrix = np.diag([physical_scale, physical_scale, 1.0])
                    translation = np.array([[1.0, 0.0, location[0]], [0.0, 1.0, location[1]], [0.0, 0.0, 1.0]])
                    best_template = (score, affine(translation @ h(rotation) @ scale_matrix))
            best_ncc, best_mat = best_template
            if best_mat is None or best_ncc < 0.30:
                raise RuntimeError(
                    f"Masson crop localization rejected: {best_inliers} SIFT inliers, NCC={best_ncc:.3f}"
                )
            localization_method = "PHYSICAL_SCALE_NCC"

        mat_crop_to_lvl4 = best_mat
        center_crop = np.float32([[crop_w / 2.0, crop_h / 2.0]])
        center_lvl4 = cv2.transform(center_crop[None, :, :], mat_crop_to_lvl4)[0][0]
        center_lvl0 = (float(center_lvl4[0] * ds4), float(center_lvl4[1] * ds4))

        mat_crop_to_lvl2 = mat_crop_to_lvl4.copy()
        mat_crop_to_lvl2[:2, :] *= (ds4 / ds2)

        return {
            "crop_size": (crop_w, crop_h),
            "crop_bgr": crop_bgr,
            "mat_crop_to_lvl2": mat_crop_to_lvl2,
            "center_lvl4": center_lvl4.tolist(),
            "center_lvl0": center_lvl0,
            "lvl2_ds": ds2,
            "lvl4_ds": ds4,
            "lvl0_dims": lvl0_dims,
            "localization_method": localization_method,
            "ncc_score": best_ncc,
            "inliers": best_inliers,
        }

    def match_crop_scales(self, crop4_path: Path, crop20_path: Path) -> tuple[np.ndarray, dict]:
        crop4 = cv2.cvtColor(np.asarray(Image.open(crop4_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        crop20 = cv2.cvtColor(np.asarray(Image.open(crop20_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g4 = clahe.apply(cv2.cvtColor(crop4, cv2.COLOR_BGR2GRAY))
        g20 = clahe.apply(cv2.cvtColor(crop20, cv2.COLOR_BGR2GRAY))

        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
        kp4, des4 = sift.detectAndCompute(g4, None)
        kp20, des20 = sift.detectAndCompute(g20, None)
        raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des4, des20, k=2)
        good = [m for m, n in raw if m.distance < 0.78 * n.distance]
        if len(good) < 8:
            raise RuntimeError(f"4x/20x 映射特征不足: {len(good)}")

        src = np.float32([kp4[m.queryIdx].pt for m in good])
        dst = np.float32([kp20[m.trainIdx].pt for m in good])
        matrix, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=10000, confidence=0.999
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if matrix is None or inliers < 8:
            raise RuntimeError(f"4x/20x 仿射解算失败: inliers={inliers}")

        return matrix.astype(np.float32), {"good_matches": len(good), "inliers": inliers}

    def match_loftr(self, img_moving: np.ndarray, img_fixed: np.ndarray) -> dict:
        box_m, scale_m, (pad_m_x, pad_m_y), (rw_m, rh_m) = letterbox_image(img_moving, 640)
        box_f, scale_f, (pad_f_x, pad_f_y), (rw_f, rh_f) = letterbox_image(img_fixed, 640)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g_m = clahe.apply(cv2.cvtColor(box_m, cv2.COLOR_BGR2GRAY))
        g_f = clahe.apply(cv2.cvtColor(box_f, cv2.COLOR_BGR2GRAY))

        t1 = K.image.image_to_tensor(g_m, keepdim=False).float().to(self.device) / 255.0
        t2 = K.image.image_to_tensor(g_f, keepdim=False).float().to(self.device) / 255.0
        with torch.no_grad():
            out = self.loftr({"image0": t1, "image1": t2})

        pts0 = out["keypoints0"].cpu().numpy()
        pts1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

        valid_m = (pts0[:, 0] >= pad_m_x) & (pts0[:, 0] < pad_m_x + rw_m) & (pts0[:, 1] >= pad_m_y) & (pts0[:, 1] < pad_m_y + rh_m)
        valid_f = (pts1[:, 0] >= pad_f_x) & (pts1[:, 0] < pad_f_x + rw_f) & (pts1[:, 1] >= pad_f_y) & (pts1[:, 1] < pad_f_y + rh_f)
        valid = valid_m & valid_f & (conf > 0.38)

        pts0 = pts0[valid]
        pts1 = pts1[valid]

        if len(pts0) < 4:
            return {
                "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
                "spatial_coverage": 0.0, "median_reproj_error": 999.0,
                "scale": 1.0, "matrix": None
            }

        pts0[:, 0] = (pts0[:, 0] - pad_m_x) / scale_m
        pts0[:, 1] = (pts0[:, 1] - pad_m_y) / scale_m
        pts1[:, 0] = (pts1[:, 0] - pad_f_x) / scale_f
        pts1[:, 1] = (pts1[:, 1] - pad_f_y) / scale_f

        mat, mask = cv2.estimateAffinePartial2D(
            pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=10000, confidence=0.999
        )
        inliers_mask = (mask.ravel() == 1) if mask is not None else np.zeros(len(pts0), dtype=bool)
        n_in = int(inliers_mask.sum())
        inlier_ratio = float(n_in / len(pts0)) if len(pts0) > 0 else 0.0

        if n_in >= 4 and mat is not None:
            p1_in = pts1[inliers_mask]
            grid_x = np.clip((p1_in[:, 0] / max(1, img_fixed.shape[1]) * 4).astype(int), 0, 3)
            grid_y = np.clip((p1_in[:, 1] / max(1, img_fixed.shape[0]) * 4).astype(int), 0, 3)
            occupied = len(set(zip(grid_x, grid_y)))
            spatial_coverage = float(occupied / 16.0)

            p0_in = pts0[inliers_mask]
            p0_trans = apply_mat(mat, p0_in)
            reproj_errors = np.linalg.norm(p1_in - p0_trans, axis=1)
            median_reproj_error = float(np.median(reproj_errors))
            scale = float(np.sqrt(mat[0, 0]**2 + mat[1, 0]**2))
        else:
            spatial_coverage = 0.0
            median_reproj_error = 999.0
            scale = 1.0
            mat = None

        return {
            "matches": len(pts0),
            "inliers": n_in,
            "inlier_ratio": inlier_ratio,
            "spatial_coverage": spatial_coverage,
            "median_reproj_error": median_reproj_error,
            "scale": scale,
            "matrix": mat
        }

    def global_align_multiangle(self, moving_lvl4: np.ndarray, fixed_lvl4: np.ndarray) -> dict:
        best = {"inliers": -1, "angle": 0, "matrix": None, "score": -1.0, "scale": 1.0, "inlier_ratio": 0.0, "spatial_coverage": 0.0}
        h_m, w_m = moving_lvl4.shape[:2]

        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rot = moving_lvl4
                rot_mat = np.eye(3, dtype=np.float64)
            elif angle == 90:
                rot = cv2.rotate(moving_lvl4, cv2.ROTATE_90_CLOCKWISE)
                rot_mat = np.array([[0.0, -1.0, float(h_m - 1)], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            elif angle == 180:
                rot = cv2.rotate(moving_lvl4, cv2.ROTATE_180)
                rot_mat = np.array([[-1.0, 0.0, float(w_m - 1)], [0.0, -1.0, float(h_m - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)
            elif angle == 270:
                rot = cv2.rotate(moving_lvl4, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rot_mat = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, float(w_m - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)

            res = self.match_loftr(rot, fixed_lvl4)
            if res["matrix"] is not None:
                score = res["inliers"] * res["inlier_ratio"] * (1.0 + res["spatial_coverage"])
                if 0.90 <= res["scale"] <= 1.10 and score > best["score"]:
                    total_mat = affine(h(res["matrix"]) @ rot_mat)
                    best = {
                        "inliers": res["inliers"],
                        "matches": res["matches"],
                        "inlier_ratio": res["inlier_ratio"],
                        "spatial_coverage": res["spatial_coverage"],
                        "median_reproj_error": res["median_reproj_error"],
                        "scale": res["scale"],
                        "angle": angle,
                        "matrix": total_mat,
                        "score": score,
                    }
        return best

    def find_all_tissue_islands(self, wsi_bgr: np.ndarray, min_area: int = 5000) -> list[dict]:
        gray = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2HSV)
        mask = ((gray < 240) | (hsv[:, :, 1] > 15)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        large_islands = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

        if not large_islands:
            return [{
                "image": wsi_bgr,
                "bbox": (0, 0, wsi_bgr.shape[1], wsi_bgr.shape[0]),
                "offset": (0, 0),
                "centroid": (wsi_bgr.shape[1] / 2.0, wsi_bgr.shape[0] / 2.0),
                "area": wsi_bgr.shape[1] * wsi_bgr.shape[0],
            }]

        islands = []
        for i in large_islands:
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            pad_x = int(bw * 0.1)
            pad_y = int(bh * 0.1)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(wsi_bgr.shape[1], bx + bw + pad_x)
            y2 = min(wsi_bgr.shape[0], by + bh + pad_y)

            islands.append({
                "image": wsi_bgr[y1:y2, x1:x2],
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "offset": (x1, y1),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                "area": int(stats[i, cv2.CC_STAT_AREA]),
            })
        return islands

    def local_morphology_refine(self, fixed_bgr: np.ndarray, moving_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        res = self.match_loftr(moving_bgr, fixed_bgr)
        if (
            res["matrix"] is not None
            and res["inliers"] >= 8
            and 0.96 <= res["scale"] <= 1.04
            and abs(float(res["matrix"][0, 2])) <= 60
            and abs(float(res["matrix"][1, 2])) <= 60
        ):
            mat_local = res["matrix"]
            aligned = cv2.warpAffine(
                moving_bgr, mat_local, (fixed_bgr.shape[1], fixed_bgr.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
            return aligned, h(mat_local), {
                "method": "Local_LoFTR",
                "inliers": res["inliers"],
                "scale": res["scale"],
                "inlier_ratio": res["inlier_ratio"],
                "dx": float(mat_local[0, 2]),
                "dy": float(mat_local[1, 2]),
            }

        g_f = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g_m = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx_f = cv2.Sobel(g_f, cv2.CV_32F, 1, 0)
        gy_f = cv2.Sobel(g_f, cv2.CV_32F, 0, 1)
        gx_m = cv2.Sobel(g_m, cv2.CV_32F, 1, 0)
        gy_m = cv2.Sobel(g_m, cv2.CV_32F, 0, 1)
        mag_f = np.sqrt(gx_f**2 + gy_f**2)
        mag_m = np.sqrt(gx_m**2 + gy_m**2)
        (dx, dy), resp = cv2.phaseCorrelate(mag_f, mag_m)
        if abs(dx) < 60 and abs(dy) < 60:
            mat_pc = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
            aligned = cv2.warpAffine(
                moving_bgr, mat_pc, (fixed_bgr.shape[1], fixed_bgr.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
            return aligned, h(mat_pc), {"method": "Phase_Correlation", "dx": -dx, "dy": -dy, "response": resp}

        return moving_bgr, np.eye(3, dtype=np.float64), {"method": "Identity_Fallback"}

    def process_sample(
        self,
        sample_id: str,
        base_dir: Path = BASE_DIR,
        tiff_dir: Path = TIFF_DIR,
        out_dir: Path = OUT_DIR,
        stains: tuple[str, ...] = ("HE", "Gram"),
    ) -> dict:
        start_t = time.time()
        sample_out = out_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)

        masson_kfb = find_file_case_insensitive(base_dir / "masson", f"{sample_id}-masson", ".kfb")
        crop4_file = find_file_case_insensitive(tiff_dir, f"{sample_id}-4x", ".tif")
        crop20_file = find_file_case_insensitive(tiff_dir, f"{sample_id}-20x", ".tif")

        if not masson_kfb or not crop4_file or not crop20_file:
            raise FileNotFoundError(
                f"Missing input files for sample {sample_id}: Masson={masson_kfb}, Crop4={crop4_file}, Crop20={crop20_file}"
            )

        print(f"\n================ Processing Sample [{sample_id}] ================")
        print("1. Localizing Masson 4x Crop in Masson WSI Level 2...")
        masson_info = self.locate_crop_in_reference(masson_kfb, crop4_file)

        crop4_w, crop4_h = masson_info["crop_size"]
        crop20_bgr = cv2.cvtColor(np.asarray(Image.open(crop20_file).convert("RGB")), cv2.COLOR_RGB2BGR)
        crop20_w, crop20_h = crop20_bgr.shape[1], crop20_bgr.shape[0]

        # KFSlicerOS keeps the 4x and 20x viewport centre fixed; 20x pixels are 5x finer.
        mat_crop20_to_crop4 = np.array(
            [
                [0.2, 0.0, crop4_w / 2.0 - 0.2 * crop20_w / 2.0],
                [0.0, 0.2, crop4_h / 2.0 - 0.2 * crop20_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        print(f"   Masson ROI Center Level 0: {masson_info['center_lvl0']} (SIFT Inliers={masson_info['inliers']})")

        corners_crop = np.float32([[0, 0], [crop4_w, 0], [0, crop4_h], [crop4_w, crop4_h]])
        with kfbslide.OpenSlide(str(masson_kfb)) as s_f:
            ds_f = float(s_f.level_downsamples[2])
            pts_lvl2 = apply_mat(masson_info["mat_crop_to_lvl2"], corners_crop)
            minx = int(max(0, pts_lvl2[:, 0].min() - 32))
            miny = int(max(0, pts_lvl2[:, 1].min() - 32))
            maxx = int(min(s_f.level_dimensions[2][0], pts_lvl2[:, 0].max() + 32))
            maxy = int(min(s_f.level_dimensions[2][1], pts_lvl2[:, 1].max() + 32))

            patch_masson_l2 = cv2.cvtColor(
                np.asarray(s_f.read_region((round(minx * ds_f), round(miny * ds_f)), 2, (maxx - minx, maxy - miny)).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )

            m_patch_masson = affine(np.array([[1, 0, -minx], [0, 1, -miny], [0, 0, 1]], dtype=np.float64) @ h(masson_info["mat_crop_to_lvl2"]))
            masson_4x_extracted = cv2.warpAffine(
                patch_masson_l2,
                m_patch_masson,
                (crop4_w, crop4_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            # Use the (mirrored) input crop directly as the Masson reference baseline.
            masson_4x_extracted = masson_info["crop_bgr"]

        Image.fromarray(cv2.cvtColor(masson_4x_extracted, cv2.COLOR_BGR2RGB)).save(
            sample_out / f"{sample_id}-Masson-4x-300dpi.tif", dpi=(300, 300), compression="tiff_lzw"
        )
        Image.fromarray(cv2.cvtColor(crop20_bgr, cv2.COLOR_BGR2RGB)).save(
            sample_out / f"{sample_id}-Masson-20x-300dpi.tif", dpi=(300, 300), compression="tiff_lzw"
        )

        masson_lvl4, _, ds_masson_lvl4, _, mpp_masson, mpp_src_m = self.read_wsi_level(masson_kfb, level=4)
        
        masson_islands = self.find_all_tissue_islands(masson_lvl4)
        cx_m_l4, cy_m_l4 = masson_info["center_lvl4"]
        target_masson_island = masson_islands[0]
        for isl in masson_islands:
            bx, by, bw, bh = isl["bbox"]
            if bx <= cx_m_l4 <= bx + bw and by <= cy_m_l4 <= by + bh:
                target_masson_island = isl
                break

        report = {
            "sample": sample_id,
            "status": "success",
            "elapsed_seconds": 0.0,
            "mpp": {
                "mpp_x": mpp_masson[0],
                "mpp_y": mpp_masson[1],
                "provenance": mpp_src_m,
            },
            "physical_fov": {
                "4x_width_um": crop4_w * mpp_masson[0] * 5.0,
                "4x_height_um": crop4_h * mpp_masson[1] * 5.0,
                "20x_width_um": crop20_w * mpp_masson[0],
                "20x_height_um": crop20_h * mpp_masson[1],
            },
            "masson_info": {
                "center_lvl0": masson_info["center_lvl0"],
                "inliers": masson_info["inliers"],
            },
            "stains": {},
        }

        for stain in stains:
            moving_kfb = find_file_case_insensitive(base_dir, f"{sample_id}-{stain}", ".kfb")
            if not moving_kfb:
                raise FileNotFoundError(f"Missing {stain} KFB for sample {sample_id}")

            print(f"2. Global Multi-Island Deep Alignment for {stain}...")
            moving_lvl4, _, ds_moving_lvl4, l0_dims_moving, mpp_moving, _ = self.read_wsi_level(moving_kfb, level=4)

            moving_islands = self.find_all_tissue_islands(moving_lvl4)
            best_candidate = None
            best_score = -1.0

            for isl_idx, mov_isl in enumerate(moving_islands):
                res_isl = self.global_align_multiangle(mov_isl["image"], target_masson_island["image"])
                if res_isl["matrix"] is not None and res_isl["score"] > best_score:
                    best_score = res_isl["score"]
                    best_candidate = {
                        "island_idx": isl_idx,
                        "align_res": res_isl,
                        "moving_island": mov_isl,
                    }

            if best_candidate is None or best_candidate["align_res"]["inliers"] < 20:
                print(f"   [Island Fallback] 组织岛匹配内点较低，自动启用全片高鲁棒回退匹配...")
                res_full = self.global_align_multiangle(moving_lvl4, masson_lvl4)
                if res_full["matrix"] is None or res_full["inliers"] < 12:
                    raise RuntimeError(f"{stain} 全局配准失败: 内点数不足 ({res_full.get('inliers', 0)})")
                align_res = res_full
                mat_moving_to_fixed_lvl4 = align_res["matrix"]
                mat_fixed_to_moving_lvl4 = cv2.invertAffineTransform(mat_moving_to_fixed_lvl4)
                print(f"   {stain} Global Alignment (Full WSI): Angle={align_res['angle']} deg, Inliers={align_res['inliers']}")
            else:
                align_res = best_candidate["align_res"]
                mov_isl = best_candidate["moving_island"]
                off_m_x, off_m_y = target_masson_island["offset"]
                off_mov_x, off_mov_y = mov_isl["offset"]
                print(f"   {stain} Global Alignment (Island #{best_candidate['island_idx']}): Angle={align_res['angle']} deg, Inliers={align_res['inliers']}")
                m_off_m = np.array([[1.0, 0.0, float(off_m_x)], [0.0, 1.0, float(off_m_y)], [0.0, 0.0, 1.0]], dtype=np.float64)
                m_off_mov = np.array([[1.0, 0.0, -float(off_mov_x)], [0.0, 1.0, -float(off_mov_y)], [0.0, 0.0, 1.0]], dtype=np.float64)
                mat_moving_to_fixed_lvl4 = affine(m_off_m @ h(align_res["matrix"]) @ m_off_mov)
                mat_fixed_to_moving_lvl4 = cv2.invertAffineTransform(mat_moving_to_fixed_lvl4)
            with kfbslide.OpenSlide(str(moving_kfb)) as s_m:
                ds_m_l2 = float(s_m.level_downsamples[2])
                scale_m = np.diag([ds_moving_lvl4 / ds_m_l2, ds_moving_lvl4 / ds_m_l2, 1.0]).astype(np.float64)
                scale_f = np.diag([ds_f / ds_masson_lvl4, ds_f / ds_masson_lvl4, 1.0]).astype(np.float64)
                mat_crop_to_m_lvl2 = affine(scale_m @ h(mat_fixed_to_moving_lvl4) @ scale_f @ h(masson_info["mat_crop_to_lvl2"]))

                pts_m_l2 = apply_mat(mat_crop_to_m_lvl2, corners_crop)
                minx_m = int(max(0, pts_m_l2[:, 0].min() - 64))
                miny_m = int(max(0, pts_m_l2[:, 1].min() - 64))
                maxx_m = int(min(s_m.level_dimensions[2][0], pts_m_l2[:, 0].max() + 64))
                maxy_m = int(min(s_m.level_dimensions[2][1], pts_m_l2[:, 1].max() + 64))

                if maxx_m <= minx_m or maxy_m <= miny_m:
                    raise RuntimeError(f"Mapped ROI outside {stain} slide bounds: {(minx_m, miny_m, maxx_m, maxy_m)}")

                patch_m_l2 = cv2.cvtColor(
                    np.asarray(s_m.read_region((round(minx_m * ds_m_l2), round(miny_m * ds_m_l2)), 2, (maxx_m - minx_m, maxy_m - miny_m)).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )

                m_patch_moving = affine(np.array([[1, 0, -minx_m], [0, 1, -miny_m], [0, 0, 1]], dtype=np.float64) @ h(mat_crop_to_m_lvl2))
                moving_4x_initial = cv2.warpAffine(
                    patch_m_l2,
                    m_patch_moving,
                    (crop4_w, crop4_h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

            # 3. 局部纯形态深度匹配 (Local LoFTR 直接吸合解剖轮廓)
            print(f"3. Local Pure Morphology (LoFTR) Refinement for {stain}...")
            aligned_4x, m_local_3x3, local_metrics = self.local_morphology_refine(masson_4x_extracted, moving_4x_initial)
            print(f"   {stain} Local Refinement: {local_metrics}")

            Image.fromarray(cv2.cvtColor(aligned_4x, cv2.COLOR_BGR2RGB)).save(
                sample_out / f"{sample_id}-{stain}-4x-aligned-300dpi.tif", dpi=(300, 300), compression="tiff_lzw"
            )
            ov_4x = cv2.addWeighted(masson_4x_extracted, 0.5, aligned_4x, 0.5, 0)
            Image.fromarray(cv2.cvtColor(ov_4x, cv2.COLOR_BGR2RGB)).save(
                sample_out / f"overlay-{stain}-4x-aligned.png", dpi=(300, 300)
            )

            # 4. 从 Level 0 原始层直接矩阵复合重采样 20x
            print(f"4. Sampling 20x High-Resolution ROI at 300 DPI for {stain}...")
            scale_l2_to_l0 = np.diag([ds_m_l2, ds_m_l2, 1.0]).astype(np.float64)

            # 复合变换矩阵：直接将 20x Crop 像素坐标映射到 Moving Slide Level 0 坐标
            m_local_inv = np.linalg.inv(m_local_3x3)
            m_total_20x_to_l0 = scale_l2_to_l0 @ h(mat_crop_to_m_lvl2) @ m_local_inv @ h(mat_crop20_to_crop4)

            corners_20 = np.float32([[0, 0], [crop20_w, 0], [0, crop20_h], [crop20_w, crop20_h]])
            pts_l0_20 = apply_mat(affine(m_total_20x_to_l0), corners_20)

            minx_20 = int(max(0, pts_l0_20[:, 0].min() - 64))
            miny_20 = int(max(0, pts_l0_20[:, 1].min() - 64))
            maxx_20 = int(min(l0_dims_moving[0], pts_l0_20[:, 0].max() + 64))
            maxy_20 = int(min(l0_dims_moving[1], pts_l0_20[:, 1].max() + 64))

            with kfbslide.OpenSlide(str(moving_kfb)) as s_m:
                patch_m_l0 = cv2.cvtColor(
                    np.asarray(s_m.read_region((minx_20, miny_20), 0, (maxx_20 - minx_20, maxy_20 - miny_20)).convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )

            m_patch_20 = affine(np.array([[1, 0, -minx_20], [0, 1, -miny_20], [0, 0, 1]], dtype=np.float64) @ m_total_20x_to_l0)
            aligned_20x = cv2.warpAffine(
                patch_m_l0,
                m_patch_20,
                (crop20_w, crop20_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            Image.fromarray(cv2.cvtColor(aligned_20x, cv2.COLOR_BGR2RGB)).save(
                sample_out / f"{sample_id}-{stain}-20x-aligned-300dpi.tif", dpi=(300, 300), compression="tiff_lzw"
            )
            ov_20x = cv2.addWeighted(crop20_bgr, 0.5, aligned_20x, 0.5, 0)
            Image.fromarray(cv2.cvtColor(ov_20x, cv2.COLOR_BGR2RGB)).save(
                sample_out / f"overlay-{stain}-20x-aligned.png", dpi=(300, 300)
            )

            passed = (
                align_res["inliers"] >= 40 and
                align_res["inlier_ratio"] >= 0.15 and
                align_res["spatial_coverage"] >= 0.25 and
                0.97 <= align_res["scale"] <= 1.03
            )
            qc_verdict = "PASS" if passed else ("WARN" if align_res["inliers"] >= 15 else "FAIL")

            report["stains"][stain] = {
                "qc_verdict": qc_verdict,
                "global_angle": align_res["angle"],
                "global_inliers": align_res["inliers"],
                "inlier_ratio": align_res["inlier_ratio"],
                "spatial_coverage": align_res["spatial_coverage"],
                "median_reproj_error": align_res["median_reproj_error"],
                "scale": align_res["scale"],
                "local_refinement": local_metrics,
            }

        self.create_contact_sheet(sample_out, sample_id, masson_4x_extracted, crop20_bgr)
        report["elapsed_seconds"] = round(time.time() - start_t, 2)
        (sample_out / "registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Sample [{sample_id}] finished in {report['elapsed_seconds']}s! Report saved to: {sample_out}\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return report

    def create_contact_sheet(self, sample_out: Path, sample_id: str, crop4_ref: np.ndarray, crop20_ref: np.ndarray):
        for mag, ref_img in [("4x", crop4_ref), ("20x", crop20_ref)]:
            images = [("Masson", ref_img)]
            for stain in ["HE", "Gram"]:
                tif_path = sample_out / f"{sample_id}-{stain}-{mag}-aligned-300dpi.tif"
                if tif_path.exists():
                    bgr = cv2.cvtColor(np.asarray(Image.open(tif_path).convert("RGB")), cv2.COLOR_RGB2BGR)
                    images.append((stain, bgr))

            thumbs = []
            for label, bgr in images:
                im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                im.thumbnail((700, 420), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (700, 470), "white")
                canvas.paste(im, ((700 - im.width) // 2, 35))
                ImageDraw.Draw(canvas).text((20, 10), f"{sample_id} {label} ({mag})", fill="black")
                thumbs.append(canvas)

            sheet = Image.new("RGB", (700 * len(thumbs), 470), "white")
            for i, canvas in enumerate(thumbs):
                sheet.paste(canvas, (700 * i, 0))
            sheet.save(sample_out / f"contact_sheet_{mag}.png", dpi=(300, 300))


# ==================== 批量执行主入口 ====================
def run_batch():
    registrar = CrossStainRegistrar()
    tiff_files = sorted(TIFF_DIR.glob("*-4x.tif"))
    sample_ids = [f.name[:-7] for f in tiff_files]

    print("================ STARTING BATCH REGISTRATION (v6.0 Final) ================")
    print(f"Found {len(sample_ids)} samples to process: {sample_ids}\n")

    summary_records = []
    batch_start = time.time()

    for idx, sid in enumerate(sample_ids, 1):
        print(f"\n>>> [{idx}/{len(sample_ids)}] Processing: {sid}")
        try:
            res = registrar.process_sample(sid)
            summary_records.append({
                "sample": sid,
                "status": "success",
                "elapsed_seconds": res["elapsed_seconds"],
                "he_verdict": res["stains"].get("HE", {}).get("qc_verdict", "UNKNOWN"),
                "he_inliers": res["stains"].get("HE", {}).get("global_inliers", 0),
                "gram_verdict": res["stains"].get("Gram", {}).get("qc_verdict", "UNKNOWN"),
                "gram_inliers": res["stains"].get("Gram", {}).get("global_inliers", 0),
            })
        except Exception as exc:
            print(f"❌ Error processing {sid}: {exc}")
            summary_records.append({
                "sample": sid,
                "status": "failed",
                "error": str(exc),
            })

    total_time = round(time.time() - batch_start, 2)
    success_count = sum(1 for r in summary_records if r.get("status") == "success")

    master_report = {
        "total_samples": len(sample_ids),
        "success_count": success_count,
        "failed_count": len(sample_ids) - success_count,
        "total_elapsed_seconds": total_time,
        "results": summary_records,
    }

    (OUT_DIR / "batch_summary.json").write_text(json.dumps(master_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n================ BATCH PROCESSING FINISHED ================")
    print(f"Processed: {len(sample_ids)}, Success: {success_count}, Failed: {len(sample_ids) - success_count}")
    print(f"Total Time: {total_time}s")
    print(f"Output Directory: {OUT_DIR}")


if __name__ == "__main__":
    run_batch()
