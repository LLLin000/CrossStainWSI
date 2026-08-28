import json
import sys
import time
from pathlib import Path

import cv2
import kfbslide
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from register_slices import BASE_DIR, OUT_DIR, TIFF_DIR, CrossStainRegistrar, affine, apply_mat, h


class DifficultCaseRegistrar:
    def __init__(self):
        self.registrar = CrossStainRegistrar()

    @staticmethod
    def _components(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = ((hsv[:, :, 1] > 25) & (gray < 248)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        components = []
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < 5000:
                continue
            component_mask = (labels == i).astype(np.uint8)
            bx, by, bw, bh = map(int, stats[i, :4])
            pad = 32
            ox, oy = max(0, bx - pad), max(0, by - pad)
            ex = min(component_mask.shape[1], bx + bw + pad)
            ey = min(component_mask.shape[0], by + bh + pad)
            local_mask = component_mask[oy:ey, ox:ex]
            contours = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            contour += np.array([ox, oy], dtype=float)
            contour = contour[np.linspace(0, len(contour) - 1, min(800, len(contour))).astype(int)]
            components.append({
                "mask": local_mask,
                "contour": contour,
                "centroid": centroids[i].astype(float),
                "area": int(stats[i, cv2.CC_STAT_AREA]),
                "bbox": (bx, by, bw, bh),
                "distance": cv2.distanceTransform((1 - local_mask).astype(np.uint8), cv2.DIST_L2, 3),
                "distance_origin": (ox, oy),
            })
        return components

    @staticmethod
    def _similarity(theta, scale, source_center, target_center):
        angle = np.radians(theta)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]) * scale
        translation = target_center - rotation @ source_center
        return np.array([
            [rotation[0, 0], rotation[0, 1], translation[0]],
            [rotation[1, 0], rotation[1, 1], translation[1]],
        ], dtype=np.float64)

    @staticmethod
    def _distance_at(distance, points, origin):
        x = np.rint(points[:, 0] - origin[0]).astype(int)
        y = np.rint(points[:, 1] - origin[1]).astype(int)
        valid = (x >= 0) & (x < distance.shape[1]) & (y >= 0) & (y < distance.shape[0])
        values = np.full(len(points), 200.0)
        values[valid] = distance[y[valid], x[valid]]
        return float(np.median(values) + 0.2 * np.mean(values))

    def _shape_score(self, source, target, matrix):
        source_to_target = apply_mat(matrix, source["contour"])
        target_to_source = apply_mat(cv2.invertAffineTransform(affine(matrix)), target["contour"])
        return self._distance_at(target["distance"], source_to_target, target["distance_origin"]) + self._distance_at(source["distance"], target_to_source, source["distance_origin"])

    def _match_components(self, moving_components, target):
        matches = []
        for index, component in enumerate(moving_components):
            best = (float("inf"), None, None)
            for theta in range(-180, 180, 2):
                for scale in np.arange(0.90, 1.101, 0.025):
                    matrix = self._similarity(theta, float(scale), component["centroid"], target["centroid"])
                    score = self._shape_score(component, target, matrix)
                    if score < best[0]:
                        best = (score, matrix, (theta, float(scale)))
            matches.append({
                "component_index": index,
                "component": component,
                "shape_score": best[0],
                "moving_to_fixed_l4": best[1],
                "theta": best[2][0],
                "scale": best[2][1],
            })
        return sorted(matches, key=lambda item: item["shape_score"])

    @staticmethod
    def _sample_level0(kfb_path, transform, size):
        width, height = size
        corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        with kfbslide.OpenSlide(str(kfb_path)) as slide:
            dimensions = slide.dimensions
            points = apply_mat(affine(transform), corners)
            x1 = max(0, int(points[:, 0].min() - 64))
            y1 = max(0, int(points[:, 1].min() - 64))
            x2 = min(dimensions[0], int(points[:, 0].max() + 64))
            y2 = min(dimensions[1], int(points[:, 1].max() + 64))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("Mapped output lies outside the source slide")
            patch = cv2.cvtColor(
                np.asarray(slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1)).convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
        patch_transform = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1.0]]) @ transform
        return cv2.warpAffine(
            patch,
            affine(patch_transform),
            size,
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    @staticmethod
    def _anchor_metrics(reference, extracted):
        if extracted is None:
            return {"ncc": -1.0, "mask_iou": 0.0, "background_agreement": 0.0, "inliers": 0, "inlier_ratio": 0.0, "median_error": float("inf")}
        ref_gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
        ext_gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY))
        ncc = float(np.corrcoef(ref_gray.ravel().astype(np.float32), ext_gray.ravel().astype(np.float32))[0, 1])
        ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
        ext_hsv = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV)
        ref_mask = ((ref_gray < 245) | (ref_hsv[:, :, 1] > 20)).astype(np.uint8)
        ext_mask = ((ext_gray < 245) | (ext_hsv[:, :, 1] > 20)).astype(np.uint8)
        union = np.logical_or(ref_mask, ext_mask).sum()
        mask_iou = float(np.logical_and(ref_mask, ext_mask).sum() / union) if union else 1.0
        background_agreement = float((ref_mask == ext_mask).mean())
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.003)
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        kp_ext, des_ext = sift.detectAndCompute(ext_gray, None)
        if des_ref is None or des_ext is None:
            return {"ncc": ncc, "mask_iou": mask_iou, "background_agreement": background_agreement, "inliers": 0, "inlier_ratio": 0.0, "median_error": float("inf")}
        good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_ref, des_ext, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return {"ncc": ncc, "mask_iou": mask_iou, "background_agreement": background_agreement, "inliers": 0, "inlier_ratio": 0.0, "median_error": float("inf")}
        src = np.float32([kp_ref[m.queryIdx].pt for m in good])
        dst = np.float32([kp_ext[m.trainIdx].pt for m in good])
        matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000)
        if matrix is None or mask is None:
            return {"ncc": ncc, "mask_iou": mask_iou, "background_agreement": background_agreement, "inliers": 0, "inlier_ratio": 0.0, "median_error": float("inf")}
        inlier_mask = mask.ravel().astype(bool)
        projected = apply_mat(matrix, src[inlier_mask])
        errors = np.linalg.norm(dst[inlier_mask] - projected, axis=1)
        return {
            "ncc": ncc,
            "mask_iou": mask_iou,
            "background_agreement": background_agreement,
            "inliers": int(inlier_mask.sum()),
            "inlier_ratio": float(inlier_mask.mean()),
            "median_error": float(np.median(errors)) if len(errors) else float("inf"),
        }

    @staticmethod
    def _remove_stale(sample_out, sample_id, stain):
        for path in (
            sample_out / f"{sample_id}-{stain}-4x-aligned-300dpi.tif",
            sample_out / f"{sample_id}-{stain}-20x-aligned-300dpi.tif",
            sample_out / f"overlay-{stain}-4x-aligned.png",
            sample_out / f"overlay-{stain}-20x-aligned.png",
        ):
            path.unlink(missing_ok=True)

    @staticmethod
    def _save_tiff(image, path):
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, dpi=(300, 300), compression="tiff_lzw")

    @staticmethod
    def _contact_sheet(sample_out, sample_id, magnitude, images, verdicts):
        cards = []
        for stain in ("Masson", "HE", "Gram"):
            card = Image.new("RGB", (700, 470), "white")
            if images.get(stain) is None:
                draw = ImageDraw.Draw(card)
                draw.text((30, 190), f"{sample_id} {stain} ({magnitude})\nMANUAL_REVIEW: {verdicts[stain]}", fill="#b00020")
            else:
                image = Image.fromarray(cv2.cvtColor(images[stain], cv2.COLOR_BGR2RGB))
                image.thumbnail((700, 420), Image.Resampling.LANCZOS)
                card.paste(image, ((700 - image.width) // 2, 35))
                ImageDraw.Draw(card).text((20, 10), f"{sample_id} {stain} ({magnitude})", fill="black")
            cards.append(card)
        sheet = Image.new("RGB", (2100, 470), "white")
        for index, card in enumerate(cards):
            sheet.paste(card, (700 * index, 0))
        sheet.save(sample_out / f"contact_sheet_{magnitude}.png", dpi=(300, 300))

    def process(self, sample_id, mask_only_stains=frozenset()):
        started = time.time()
        sample_out = OUT_DIR / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        crop4_path = TIFF_DIR / f"{sample_id}-4x.tif"
        crop20_path = TIFF_DIR / f"{sample_id}-20x.tif"
        reference_path = BASE_DIR / "masson" / f"{sample_id.replace('w', 'W')}-masson.kfb"
        crop4 = cv2.cvtColor(np.asarray(Image.open(crop4_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        crop20 = cv2.cvtColor(np.asarray(Image.open(crop20_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        crop4_h, crop4_w = crop4.shape[:2]
        crop20_h, crop20_w = crop20.shape[:2]
        self._save_tiff(crop4, sample_out / f"{sample_id}-Masson-4x-300dpi.tif")
        self._save_tiff(crop20, sample_out / f"{sample_id}-Masson-20x-300dpi.tif")

        reference_info = self.registrar.locate_crop_in_reference(reference_path, crop4_path)
        fixed_l4, _, fixed_ds4, _, _, _ = self.registrar.read_wsi_level(reference_path, 4)
        fixed_components = self._components(fixed_l4)
        center = np.array(reference_info["center_lvl4"])
        target = next(component for component in fixed_components if (
            component["bbox"][0] <= center[0] <= component["bbox"][0] + component["bbox"][2]
            and component["bbox"][1] <= center[1] <= component["bbox"][1] + component["bbox"][3]
        ))
        fixed_l2_to_l4 = np.diag([
            reference_info["lvl2_ds"] / reference_info["lvl4_ds"],
            reference_info["lvl2_ds"] / reference_info["lvl4_ds"],
            1.0,
        ])
        crop4_to_crop20, crop_scale_metrics = self.registrar.match_crop_scales(crop4_path, crop20_path)
        crop20_to_crop4 = h(cv2.invertAffineTransform(crop4_to_crop20))
        anchor4_to_l0 = np.diag([reference_info["lvl2_ds"], reference_info["lvl2_ds"], 1.0]) @ h(reference_info["mat_crop_to_lvl2"])
        anchor20_to_l0 = anchor4_to_l0 @ crop20_to_crop4
        anchor4 = self._sample_level0(reference_path, anchor4_to_l0, (crop4_w, crop4_h))
        anchor20 = self._sample_level0(reference_path, anchor20_to_l0, (crop20_w, crop20_h))
        anchor4_metrics = self._anchor_metrics(crop4, anchor4)
        anchor20_metrics = self._anchor_metrics(crop20, anchor20)
        anchor_pass = (
            anchor4_metrics["inliers"] >= 100
            and anchor20_metrics["inliers"] >= 100
            and anchor4_metrics["ncc"] >= 0.45
            and anchor20_metrics["ncc"] >= 0.25
            and anchor4_metrics["mask_iou"] >= 0.65
            and anchor20_metrics["mask_iou"] >= 0.75
        )
        report = {
            "sample": sample_id,
            "method": "tissue_mask_plus_20x_residual",
            "crop_scale_metrics": crop_scale_metrics,
            "reference_anchor": {
                "status": "PASS" if anchor_pass else "ABSTAIN",
                "4x": anchor4_metrics,
                "20x": anchor20_metrics,
            },
            "stains": {},
        }
        images4 = {"Masson": crop4}
        images20 = {"Masson": crop20}
        verdicts = {"Masson": "PASS"}
        if not anchor_pass:
            for stain in ("HE", "Gram"):
                self._remove_stale(sample_out, sample_id, stain)
                images4[stain] = None
                images20[stain] = None
                verdicts[stain] = "ABSTAIN: reference anchor self-check failed"
                report["stains"][stain] = {"status": "ABSTAIN", "reason": "reference_anchor_self_check_failed"}
            self._contact_sheet(sample_out, sample_id, "4x", images4, verdicts)
            self._contact_sheet(sample_out, sample_id, "20x", images20, verdicts)
            report["status"] = "REFERENCE_ANCHOR_ABSTAIN"
            report["elapsed_seconds"] = round(time.time() - started, 2)
            serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
            (sample_out / "difficult_registration_report.json").write_text(serialized, encoding="utf-8")
            (sample_out / "registration_report.json").write_text(serialized, encoding="utf-8")
            return report

        for stain in ("HE", "Gram"):
            moving_path = BASE_DIR / f"{sample_id.replace('w', 'W')}-{stain}.kfb"
            moving_l4, _, moving_ds4, _, _, _ = self.registrar.read_wsi_level(moving_path, 4)
            hypotheses = self._match_components(self._components(moving_l4), target)
            evaluations = []
            for hypothesis in hypotheses[:2]:
                crop4_to_moving_l0 = (
                    np.diag([moving_ds4, moving_ds4, 1.0])
                    @ h(cv2.invertAffineTransform(affine(hypothesis["moving_to_fixed_l4"])))
                    @ fixed_l2_to_l4
                    @ h(reference_info["mat_crop_to_lvl2"])
                )
                crop20_to_moving_l0 = crop4_to_moving_l0 @ crop20_to_crop4
                predicted20 = self._sample_level0(moving_path, crop20_to_moving_l0, (crop20_w, crop20_h))
                local = self.registrar.match_loftr(predicted20, crop20)
                matrix = local["matrix"]
                residual = float("inf") if matrix is None else float(np.hypot(matrix[0, 2], matrix[1, 2]))
                supported = (
                    matrix is not None
                    and local["inliers"] >= 10
                    and local["spatial_coverage"] >= 0.15
                    and 0.97 <= local["scale"] <= 1.03
                    and residual <= 150
                )
                evaluations.append({
                    "hypothesis": hypothesis,
                    "crop4_to_l0": crop4_to_moving_l0,
                    "crop20_to_l0": crop20_to_moving_l0,
                    "local": local,
                    "residual_px": residual,
                    "supported": supported,
                })
            supported = [item for item in evaluations if item["supported"]]
            supported.sort(key=lambda item: (-item["local"]["inliers"], item["hypothesis"]["shape_score"]))
            manual_accept = not supported and stain in mask_only_stains and evaluations
            if not supported and not manual_accept:
                self._remove_stale(sample_out, sample_id, stain)
                verdicts[stain] = "ABSTAIN: no 20x-supported tissue hypothesis"
                images4[stain] = None
                images20[stain] = None
                report["stains"][stain] = {
                    "status": "ABSTAIN",
                    "hypotheses": [{
                        "component": item["hypothesis"]["component_index"],
                        "shape_score": item["hypothesis"]["shape_score"],
                        "theta": item["hypothesis"]["theta"],
                        "shape_scale": item["hypothesis"]["scale"],
                        "20x_inliers": item["local"]["inliers"],
                        "20x_scale": item["local"]["scale"],
                        "20x_coverage": item["local"]["spatial_coverage"],
                        "20x_residual_px": item["residual_px"] if np.isfinite(item["residual_px"]) else None,
                    } for item in evaluations],
                }
                continue

            if manual_accept:
                winner = min(evaluations, key=lambda item: item["hypothesis"]["shape_score"])
                inverse_local = np.eye(3)
                status = "MANUAL_ACCEPTED_MASK_ONLY"
            else:
                winner = supported[0]
                inverse_local = h(cv2.invertAffineTransform(winner["local"]["matrix"]))
                status = "PASS"
            refined20 = winner["crop20_to_l0"] @ inverse_local
            refined4 = winner["crop4_to_l0"] @ crop20_to_crop4 @ inverse_local @ np.linalg.inv(crop20_to_crop4)
            aligned4 = self._sample_level0(moving_path, refined4, (crop4_w, crop4_h))
            aligned20 = self._sample_level0(moving_path, refined20, (crop20_w, crop20_h))
            self._save_tiff(aligned4, sample_out / f"{sample_id}-{stain}-4x-aligned-300dpi.tif")
            self._save_tiff(aligned20, sample_out / f"{sample_id}-{stain}-20x-aligned-300dpi.tif")
            Image.fromarray(cv2.cvtColor(cv2.addWeighted(crop4, 0.5, aligned4, 0.5, 0), cv2.COLOR_BGR2RGB)).save(sample_out / f"overlay-{stain}-4x-aligned.png")
            Image.fromarray(cv2.cvtColor(cv2.addWeighted(crop20, 0.5, aligned20, 0.5, 0), cv2.COLOR_BGR2RGB)).save(sample_out / f"overlay-{stain}-20x-aligned.png")
            verdicts[stain] = status
            images4[stain] = aligned4
            images20[stain] = aligned20
            report["stains"][stain] = {
                "status": status,
                "component": winner["hypothesis"]["component_index"],
                "shape_score": winner["hypothesis"]["shape_score"],
                "theta": winner["hypothesis"]["theta"],
                "shape_scale": winner["hypothesis"]["scale"],
                "20x_inliers": winner["local"]["inliers"],
                "20x_scale": winner["local"]["scale"],
                "20x_coverage": winner["local"]["spatial_coverage"],
                "20x_residual_px": winner["residual_px"] if np.isfinite(winner["residual_px"]) else None,
            }

        self._contact_sheet(sample_out, sample_id, "4x", images4, verdicts)
        self._contact_sheet(sample_out, sample_id, "20x", images20, verdicts)
        report["elapsed_seconds"] = round(time.time() - started, 2)
        report["status"] = "PASS" if all(item["status"] == "PASS" for item in report["stains"].values()) else "REVIEWED_PARTIAL"
        for stale_report in ("rigorous_rescue_report.json", "registration_report.json"):
            (sample_out / stale_report).unlink(missing_ok=True)
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        (sample_out / "difficult_registration_report.json").write_text(serialized, encoding="utf-8")
        (sample_out / "registration_report.json").write_text(serialized, encoding="utf-8")
        return report


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "4-4w-1"
    mask_only = {argument.split("=", 1)[1] for argument in sys.argv[2:] if argument.startswith("--accept-mask-only=")}
    result = DifficultCaseRegistrar().process(sample, mask_only)
    print(json.dumps(result, indent=2, ensure_ascii=False))
