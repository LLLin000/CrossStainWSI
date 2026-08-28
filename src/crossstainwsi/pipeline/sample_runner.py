"""
单样本全流程配准执行器 (SampleRunner)
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics, RegistrationResult, RegistrationStatus
from crossstainwsi.io.image import ImageCropReader
from crossstainwsi.io.kfb import KFBReader
from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.qc.rules import QCRuleEngine
from crossstainwsi.registration.global_reg import GlobalRegistrar
from crossstainwsi.registration.local_reg import LocalRefiner
from crossstainwsi.registration.reference_anchor import ReferenceAnchorLocator
from crossstainwsi.reporting.contact_sheet import ContactSheetGenerator
from crossstainwsi.reporting.report import ReportGenerator
from crossstainwsi.sampling.sampler import WSISampler
from crossstainwsi.tissue.islands import TissueSegmenter
from crossstainwsi.transforms.geom import affine, apply_mat, h
from crossstainwsi.transforms.graph import TransformGraph


def find_file_case_insensitive(directory: Path, pattern_prefix: str, suffix: str) -> Optional[Path]:
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


class SampleRunner:
    """
    负责执行单个样本从参考定位、组织岛隔离、全局深度形态匹配、局部微调、Level 0 采样到报告输出的全流程
    """
    def __init__(self, config: Optional[PipelineConfig] = None, loftr_matcher: Optional[LoFTRMatcher] = None):
        self.cfg = config or PipelineConfig()
        self.device = self.cfg.get_torch_device()
        self.loftr = loftr_matcher or LoFTRMatcher(device=self.device)
        self.anchor_locator = ReferenceAnchorLocator()
        self.global_registrar = GlobalRegistrar(loftr_matcher=self.loftr)
        self.local_refiner = LocalRefiner(loftr_matcher=self.loftr)
        self.qc_engine = QCRuleEngine()

    def process(self, sample_id: str) -> Dict[str, Any]:
        start_time = time.time()
        sample_out = self.cfg.output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)

        is_mirrored = sample_id in self.cfg.mirrored_samples
        print(f"\n================ [CrossStainWSI] Processing [{sample_id}] ================")
        if is_mirrored:
            print(f"[*] Sample {sample_id} is marked as MIRRORED; automatically flipping horizontal inputs.")

        # 1. 查找输入文件
        ref_kfb_path = find_file_case_insensitive(
            self.cfg.base_dir / self.cfg.reference_stain,
            f"{sample_id}-{self.cfg.reference_stain}",
            ".kfb"
        )
        crop4_path = find_file_case_insensitive(self.cfg.tiff_dir, f"{sample_id}-4x", ".tif")
        crop20_path = find_file_case_insensitive(self.cfg.tiff_dir, f"{sample_id}-20x", ".tif")

        if not ref_kfb_path or not crop4_path or not crop20_path:
            raise FileNotFoundError(
                f"Missing inputs for {sample_id}: "
                f"RefKFB={ref_kfb_path}, Crop4={crop4_path}, Crop20={crop20_path}"
            )

        # 2. 读取手工截图
        crop4_bgr, (crop4_w, crop4_h) = ImageCropReader.load_crop_bgr(crop4_path, flip_horizontal=is_mirrored)
        crop20_bgr, (crop20_w, crop20_h) = ImageCropReader.load_crop_bgr(crop20_path, flip_horizontal=is_mirrored)

        # 3. 参考切片锚点定位
        print("1. Localizing Reference (Masson) Anchor in WSI...")
        with KFBReader(ref_kfb_path, default_mpp=self.cfg.default_mpp) as ref_reader:
            ref_spec = ref_reader.read_metadata()
            anchor_res = self.anchor_locator.locate(crop4_bgr, ref_reader)
            if not anchor_res.is_valid or anchor_res.mat_crop4_to_lvl4 is None:
                raise RuntimeError(f"Reference anchor localization failed for sample {sample_id}")

            print(f"   Anchor found ({anchor_res.localization_method}): inliers={anchor_res.metrics.inliers}, center_l0={anchor_res.center_lvl0}")

            # 4. 保存 Masson 结果 (直接采用翻转后的高质量原始截图作为绝对基准)
            ImageCropReader.save_publication_tiff(
                crop4_bgr,
                sample_out / f"{sample_id}-Masson-4x-300dpi.tif",
            )
            ImageCropReader.save_publication_tiff(
                crop20_bgr,
                sample_out / f"{sample_id}-Masson-20x-300dpi.tif",
            )

            # 读取 Level 4 图像进行组织岛提取
            ref_lvl4_bgr, ds_ref_l4, _ = ref_reader.read_level_image(4)
            ds_ref_l2 = ref_reader.read_metadata().get_level_downsample(2)
            ref_islands = TissueSegmenter.find_tissue_islands(ref_lvl4_bgr)
            target_ref_island = TissueSegmenter.select_island_by_coordinate(ref_islands, anchor_res.center_lvl4)

        # 5. 跨染色循环处理 (HE, Gram)
        results_by_stain: Dict[str, RegistrationResult] = {}
        all_4x_images = {"Masson": crop4_bgr}
        all_20x_images = {"Masson": crop20_bgr}

        for stain in self.cfg.moving_stains:
            print(f"\n2. Registering moving stain [{stain}]...")
            moving_kfb_path = find_file_case_insensitive(self.cfg.base_dir, f"{sample_id}-{stain}", ".kfb")
            if not moving_kfb_path:
                print(f"   [WARN] Moving slide for {stain} not found, skipping.")
                continue

            with KFBReader(moving_kfb_path, default_mpp=self.cfg.default_mpp) as moving_reader:
                moving_lvl4_bgr, ds_mov_l4, _ = moving_reader.read_level_image(4)
                ds_mov_l2 = moving_reader.read_metadata().get_level_downsample(2)

                # 初始化变换拓扑图
                graph = TransformGraph(
                    crop4_size=(crop4_w, crop4_h),
                    crop20_size=(crop20_w, crop20_h),
                    ref_ds_lvl2=ds_ref_l2,
                    ref_ds_lvl4=ds_ref_l4,
                    moving_ds_lvl2=ds_mov_l2,
                    moving_ds_lvl4=ds_mov_l4,
                )
                graph.set_reference_anchor(anchor_res.mat_crop4_to_lvl4)

                # 全局多角度形态匹配
                global_res = self.global_registrar.register_stain(
                    moving_lvl4_bgr,
                    ref_lvl4_bgr,
                    target_ref_island,
                )

                if not global_res.is_valid or global_res.mat_moving_to_ref_lvl4 is None:
                    print(f"   [ABSTAIN] Global alignment failed for {stain}: {global_res.details.get('reason')}")
                    results_by_stain[stain] = RegistrationResult(
                        sample_id=sample_id,
                        moving_stain=stain,
                        reference_stain=self.cfg.reference_stain,
                        status=RegistrationStatus.ABSTAIN,
                        reason=global_res.details.get("reason", "Global alignment failed"),
                        metrics=global_res.metrics,
                    )
                    continue

                graph.set_global_cross_stain(global_res.mat_moving_to_ref_lvl4)
                mat_crop4_to_mov_l2 = graph.get_crop4_to_moving_lvl2()

                # 从 Moving Slide Level 2 单次采样 4x 初始切片
                initial_moving_4x = WSISampler.sample_patch(
                    moving_reader,
                    mat_crop4_to_mov_l2,
                    (crop4_w, crop4_h),
                    level=2,
                )

                # 局部形态微调
                print(f"3. Local morphology refinement for {stain}...")
                local_res = self.local_refiner.refine(initial_moving_4x, crop4_bgr)
                graph.set_local_refinement(local_res.mat_local_3x3)
                aligned_4x = local_res.aligned_image_bgr
                print(f"   Local refinement applied: {local_res.method} (details: {local_res.details})")

                # Level 0 单次无畸变重采样 20x
                print(f"4. Direct Level-0 sampling for 20x [{stain}]...")
                mat_crop20_to_mov_l0 = graph.get_crop20_to_moving_lvl0()
                aligned_20x = WSISampler.sample_patch(
                    moving_reader,
                    mat_crop20_to_mov_l0,
                    (crop20_w, crop20_h),
                    level=0,
                )

                # QC 评估
                qc_status, qc_reason = self.qc_engine.evaluate_cross_stain(global_res.metrics)
                print(f"   QC Status for {stain}: {qc_status.value} - {qc_reason}")

                # 保存 4x 和 20x 出版级 TIFF
                ImageCropReader.save_publication_tiff(
                    aligned_4x,
                    sample_out / f"{sample_id}-{stain}-4x-aligned-300dpi.tif",
                )
                ImageCropReader.save_publication_tiff(
                    aligned_20x,
                    sample_out / f"{sample_id}-{stain}-20x-aligned-300dpi.tif",
                )

                # 保存 Overlays
                if self.cfg.save_overlays:
                    ImageCropReader.save_overlay_png(
                        crop4_bgr,
                        aligned_4x,
                        sample_out / f"overlay-{stain}-4x-aligned.png",
                    )
                    ImageCropReader.save_overlay_png(
                        crop20_bgr,
                        aligned_20x,
                        sample_out / f"overlay-{stain}-20x-aligned.png",
                    )

                all_4x_images[stain] = aligned_4x
                all_20x_images[stain] = aligned_20x

                results_by_stain[stain] = RegistrationResult(
                    sample_id=sample_id,
                    moving_stain=stain,
                    reference_stain=self.cfg.reference_stain,
                    status=qc_status,
                    reason=qc_reason,
                    transform_matrix_3x3=global_res.mat_moving_to_ref_lvl4.tolist(),
                    metrics=global_res.metrics,
                    qc_details={
                        "global": global_res.details,
                        "local": local_res.details,
                    },
                )

        # 6. 生成 Contact Sheets 拼图
        if self.cfg.save_contact_sheets:
            print("5. Generating 4x and 20x Contact Sheets...")
            ContactSheetGenerator.create_contact_sheet(
                all_4x_images,
                f"Sample {sample_id} - 4x Aligned Cross-Stain Comparison (300 DPI)",
                sample_out / f"contact_sheet_4x.png",
            )
            ContactSheetGenerator.create_contact_sheet(
                all_20x_images,
                f"Sample {sample_id} - 20x Aligned Cross-Stain Comparison (300 DPI)",
                sample_out / f"contact_sheet_20x.png",
            )

        # 7. 保存结构化 JSON 报告
        elapsed = time.time() - start_time
        report = ReportGenerator.save_sample_report(
            sample_id=sample_id,
            results=results_by_stain,
            anchor_info={
                "center_lvl0": anchor_res.center_lvl0,
                "inliers": anchor_res.metrics.inliers,
                "method": anchor_res.localization_method,
            },
            mpp_info={
                "mpp_x": ref_spec.mpp_x,
                "mpp_y": ref_spec.mpp_y,
                "provenance": ref_spec.mpp_source,
            },
            fov_info={
                "4x_width_um": crop4_w * ref_spec.mpp_x * 5.0,
                "4x_height_um": crop4_h * ref_spec.mpp_y * 5.0,
                "20x_width_um": crop20_w * ref_spec.mpp_x,
                "20x_height_um": crop20_h * ref_spec.mpp_y,
            },
            out_path=sample_out / "registration_report.json",
            elapsed_seconds=elapsed,
        )

        print(f"[Done] Sample {sample_id} completed in {elapsed:.2f}s, Overall Status: {report['overall_status']}")
        return report
