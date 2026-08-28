import json
import math
import os
import sys
from pathlib import Path

import cv2
import kfbslide
import numpy as np
from PIL import Image, ImageDraw

from register_slices import affine, apply_mat, h

BASE_DIR = Path(r"E:\研究数据\骨科\切片扫描\2026-08-21")
TIFF_DIR = Path(os.environ.get("ANCHOR_TIFF_DIR", r"E:\研究数据\骨科\切片扫描\tiff"))
OUT_DIR = Path(os.environ.get("ANCHOR_OUT_DIR", r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"))
SAMPLES = sys.argv[1:] or ["4-4w-1", "2-2w-1", "5-4w-2"]


def clahe_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


def physical_anchor_candidates(crop, wsi_l4, ds4, max_candidates=8):
    # Manual 4x pixels correspond to 5x level-0 pixels; level 4 is ds4.
    scale = 5.0 / ds4
    scaled_size = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
    scaled = cv2.resize(crop, scaled_size, interpolation=cv2.INTER_AREA)
    search = clahe_gray(wsi_l4)
    candidates = []
    for angle in range(-180, 180, 5):
        rotation = cv2.getRotationMatrix2D((scaled_size[0] / 2.0, scaled_size[1] / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(scaled, rotation, scaled_size, borderValue=(255, 255, 255))
        template = clahe_gray(rotated)
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        for _ in range(3):
            _, score, _, location = cv2.minMaxLoc(response)
            if score < 0.05:
                break
            center = (location[0] + scaled_size[0] / 2.0, location[1] + scaled_size[1] / 2.0)
            matrix = np.array([
                [1.0, 0.0, location[0]],
                [0.0, 1.0, location[1]],
                [0.0, 0.0, 1.0],
            ]) @ h(rotation) @ np.diag([scale, scale, 1.0])
            candidates.append({
                "ncc_score": float(score),
                "angle_search": angle,
                "center_l4": center,
                "crop_to_l4": matrix,
            })
            # Non-maximum suppression in the response map for this angle.
            x0, y0 = location
            rx = max(8, scaled_size[0] // 3)
            ry = max(8, scaled_size[1] // 3)
            response[max(0, y0 - ry):min(response.shape[0], y0 + ry + 1), max(0, x0 - rx):min(response.shape[1], x0 + rx + 1)] = -1.0
    candidates.sort(key=lambda item: item["ncc_score"], reverse=True)
    unique = []
    for candidate in candidates:
        if any(np.hypot(candidate["center_l4"][0] - old["center_l4"][0], candidate["center_l4"][1] - old["center_l4"][1]) < 120 for old in unique):
            continue
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique


def sample_direct(kfb_path, crop_to_l0, size):
    width, height = size
    corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    with kfbslide.OpenSlide(str(kfb_path)) as slide:
        points = apply_mat(affine(crop_to_l0), corners)
        x1 = max(0, int(points[:, 0].min() - 64))
        y1 = max(0, int(points[:, 1].min() - 64))
        x2 = min(slide.dimensions[0], int(points[:, 0].max() + 64))
        y2 = min(slide.dimensions[1], int(points[:, 1].max() + 64))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = cv2.cvtColor(np.asarray(slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1)).convert("RGB")), cv2.COLOR_RGB2BGR)
    matrix = np.array([[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]]) @ crop_to_l0
    return cv2.warpAffine(patch, affine(matrix), size, flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderValue=(255, 255, 255))


def same_image_metrics(reference, extracted):
    if extracted is None:
        return {
            "ncc": -1.0,
            "mask_iou": 0.0,
            "background_agreement": 0.0,
            "edge_corr": -1.0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error": math.inf,
        }
    a = clahe_gray(reference)
    b = clahe_gray(extracted)
    ncc = float(np.corrcoef(a.reshape(-1).astype(np.float32), b.reshape(-1).astype(np.float32))[0, 1])
    hsv_a = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV)
    mask_a = ((a < 245) | (hsv_a[:, :, 1] > 20)).astype(np.uint8)
    mask_b = ((b < 245) | (hsv_b[:, :, 1] > 20)).astype(np.uint8)
    union = np.logical_or(mask_a, mask_b).sum()
    mask_iou = float(np.logical_and(mask_a, mask_b).sum() / union) if union else 1.0
    background_agreement = float((mask_a == mask_b).mean())
    edge_a = cv2.Canny(a, 30, 80).astype(np.float32)
    edge_b = cv2.Canny(b, 30, 80).astype(np.float32)
    edge_corr = float(np.corrcoef(edge_a.reshape(-1), edge_b.reshape(-1))[0, 1])
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.003)
    kp_a, des_a = sift.detectAndCompute(a, None)
    kp_b, des_b = sift.detectAndCompute(b, None)
    if des_a is None or des_b is None:
        return {
            "ncc": ncc,
            "mask_iou": mask_iou,
            "background_agreement": background_agreement,
            "edge_corr": edge_corr,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error": math.inf,
        }
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return {
            "ncc": ncc,
            "mask_iou": mask_iou,
            "background_agreement": background_agreement,
            "edge_corr": edge_corr,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error": math.inf,
        }
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000)
    if matrix is None or mask is None:
        return {
            "ncc": ncc,
            "mask_iou": mask_iou,
            "background_agreement": background_agreement,
            "edge_corr": edge_corr,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_error": math.inf,
        }
    inlier_mask = mask.ravel().astype(bool)
    projected = apply_mat(matrix, src[inlier_mask])
    errors = np.linalg.norm(dst[inlier_mask] - projected, axis=1)
    return {
        "ncc": ncc,
        "mask_iou": mask_iou,
        "background_agreement": background_agreement,
        "edge_corr": edge_corr,
        "inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "median_error": float(np.median(errors)) if len(errors) else math.inf,
    }


def render_diagnostic(sample_out, sample_id, rows):
    cards = []
    for row in rows:
        cards.append((f"#{row['rank']} score={row['ncc_score']:.3f}\n4x ncc={row['self4']['ncc']:.3f}, inliers={row['self4']['inliers']}\n20x ncc={row['self20']['ncc']:.3f}, inliers={row['self20']['inliers']}", row["extract4"], row["extract20"]))
    card_w, card_h = 700, 600
    sheet = Image.new("RGB", (card_w * max(1, len(cards)), card_h * 2), "white")
    for index, (label, e4, e20) in enumerate(cards):
        for line, image in enumerate((e4, e20)):
            card = Image.new("RGB", (card_w, card_h), "white")
            ImageDraw.Draw(card).text((15, 10), f"{sample_id} {label} {'4x' if line == 0 else '20x'}", fill="black")
            if image is not None:
                pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil.thumbnail((card_w, card_h - 45), Image.Resampling.LANCZOS)
                card.paste(pil, ((card_w - pil.width) // 2, 40))
            sheet.paste(card, (index * card_w, line * card_h))
    sheet.save(sample_out / "reference_anchor_diagnostic.png", dpi=(300, 300))


def run(sample_id):
    print(f"\n=== Reference anchor self-check: {sample_id} ===")
    sample_out = OUT_DIR / sample_id / "reference_anchor_diagnostic"
    sample_out.mkdir(parents=True, exist_ok=True)
    crop4_path = TIFF_DIR / f"{sample_id}-4x.tif"
    crop20_path = TIFF_DIR / f"{sample_id}-20x.tif"
    masson_path = BASE_DIR / "masson" / f"{sample_id.replace('w', 'W')}-masson.kfb"
    crop4 = cv2.cvtColor(np.asarray(Image.open(crop4_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    crop20 = cv2.cvtColor(np.asarray(Image.open(crop20_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    with kfbslide.OpenSlide(str(masson_path)) as slide:
        masson_l4 = cv2.cvtColor(np.asarray(slide.read_region((0, 0), 4, slide.level_dimensions[4]).convert("RGB")), cv2.COLOR_RGB2BGR)
        ds4 = float(slide.level_downsamples[4])
        ds2 = float(slide.level_downsamples[2])
    crop4_to_crop20 = np.array([
        [5.0, 0.0, crop20.shape[1] / 2.0 - 5.0 * crop4.shape[1] / 2.0],
        [0.0, 5.0, crop20.shape[0] / 2.0 - 5.0 * crop4.shape[0] / 2.0],
    ], dtype=np.float32)
    scale_metrics = {"method": "fixed_5x_center"}
    crop20_to_crop4 = h(cv2.invertAffineTransform(crop4_to_crop20))
    candidates = physical_anchor_candidates(crop4, masson_l4, ds4)
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        crop4_to_l2 = np.diag([ds4 / ds2, ds4 / ds2, 1.0]) @ candidate["crop_to_l4"]
        crop4_to_l0 = np.diag([ds2, ds2, 1.0]) @ crop4_to_l2
        crop20_to_l0 = crop4_to_l0 @ crop20_to_crop4
        extract4 = sample_direct(masson_path, crop4_to_l0, (crop4.shape[1], crop4.shape[0]))
        extract20 = sample_direct(masson_path, crop20_to_l0, (crop20.shape[1], crop20.shape[0]))
        self4 = same_image_metrics(crop4, extract4)
        self20 = same_image_metrics(crop20, extract20)
        row = {
            "rank": rank,
            "ncc_score": candidate["ncc_score"],
            "angle_search": candidate["angle_search"],
            "center_l4": candidate["center_l4"],
            "self4": self4,
            "self20": self20,
            "extract4": extract4,
            "extract20": extract20,
        }
        rows.append(row)
        print(f"candidate #{rank}: template_ncc={candidate['ncc_score']:.3f}, angle={candidate['angle_search']}, center={np.round(candidate['center_l4'],1)}, 4x ncc={self4['ncc']:.3f}/inliers={self4['inliers']}, 20x ncc={self20['ncc']:.3f}/inliers={self20['inliers']}")
    serializable = {
        "sample": sample_id,
        "crop_scale_metrics": scale_metrics,
        "candidates": [{k: v for k, v in row.items() if k not in ("extract4", "extract20")} for row in rows],
    }
    (sample_out / "reference_anchor_report.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    render_diagnostic(sample_out, sample_id, rows[:4])
    print(f"diagnostic saved to {sample_out}")


# Imported lazily after helper definitions to keep this file runnable from the scan directory.
from register_slices import CrossStainRegistrar
self_registrar = CrossStainRegistrar()
for sample in SAMPLES:
    run(sample)
