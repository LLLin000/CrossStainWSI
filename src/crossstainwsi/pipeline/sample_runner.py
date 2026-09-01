"""
自适应工作流执行器 (SampleRunner)
根据 ExecutionPlan 智能调度不同任务路径，实现严格产物安全隔离 (final / review / debug)
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import QCMetrics, RegistrationResult, RegistrationStatus
from crossstainwsi.inventory.assets import SampleAssets
from crossstainwsi.inventory.discover import AssetDiscoverer
from crossstainwsi.io.image import ImageCropReader
from crossstainwsi.io.kfb import KFBReader
from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.execution_plan import ExecutionPlan, TaskType
from crossstainwsi.planning.goal import UserGoal
from crossstainwsi.planning.planner import WorkflowPlanner
from crossstainwsi.qc.rules import QCRuleEngine
from crossstainwsi.registration.global_reg import GlobalRegistrar
from crossstainwsi.registration.local_reg import LocalRefiner
from crossstainwsi.registration.reference_anchor import ReferenceAnchorLocator
from crossstainwsi.reporting.contact_sheet import ContactSheetGenerator
from crossstainwsi.reporting.report import ReportGenerator
from crossstainwsi.review.states import ArtifactTier, ConfidenceTier, RunVerdict, resolve_artifact_dir
from crossstainwsi.sampling.sampler import WSISampler
from crossstainwsi.tissue.islands import TissueSegmenter
from crossstainwsi.transforms.geom import affine, apply_mat, h
from crossstainwsi.transforms.graph import TransformGraph


class SampleRunner:
    """
    自适应跨染色 WSI 配准与提取执行器
    """
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        loftr_matcher: Optional[LoFTRMatcher] = None,
        goal: Optional[UserGoal] = None,
        acquisition_profile: Optional[AcquisitionProfile] = None,
    ):
        self.cfg = config or PipelineConfig()
        self.goal = goal or UserGoal()
        self.profile = acquisition_profile or AcquisitionProfile()
        self.device = self.cfg.get_torch_device()
        self.loftr = loftr_matcher or LoFTRMatcher(device=self.device)
        self.anchor_locator = ReferenceAnchorLocator()
        self.global_registrar = GlobalRegistrar(loftr_matcher=self.loftr)
        self.local_refiner = LocalRefiner(loftr_matcher=self.loftr)
        self.qc_engine = QCRuleEngine()
        self.planner = WorkflowPlanner(goal=self.goal, acquisition_profile=self.profile)

    def process(
        self,
        sample_id: str,
        assets: Optional[SampleAssets] = None,
        plan: Optional[ExecutionPlan] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.cfg.notify_progress(sample_id, 5, "Initializing asset discovery & planning")
        sample_base_out = self.cfg.output_dir / sample_id
        sample_base_out.mkdir(parents=True, exist_ok=True)
        # 1. 资产发现与规划 (如果没有直接提供)
        if assets is None:
            discoverer = AssetDiscoverer(
                base_dir=self.cfg.base_dir,
                tiff_dir=self.cfg.tiff_dir,
                mirrored_sample_ids=self.cfg.mirrored_samples,
            )
            inventory = discoverer.discover()
            assets = inventory.get_sample(sample_id)
            if not assets:
                raise FileNotFoundError(f"No assets discovered for sample [{sample_id}] in {self.cfg.base_dir}")

        if plan is None:
            plan = self.planner.plan(assets)

        print(f"\n{plan.describe()}\n")

        # 2. 前置检查拦截 (如缺少必选染色)
        if not plan.is_executable:
            print(f"[BLOCKED] Execution plan cannot run: {plan.block_reason}")
            debug_out = resolve_artifact_dir(sample_base_out, RunVerdict.INCOMPLETE)
            report = ReportGenerator.save_sample_report(
                sample_id=sample_id,
                results={},
                anchor_info={"error": plan.block_reason},
                mpp_info={},
                fov_info={},
                out_path=debug_out / "registration_report.json",
                elapsed_seconds=time.time() - start_time,
            )
            report["overall_status"] = RunVerdict.INCOMPLETE.value
            return report

        # 3. 获取参考切片
        ref_slide_asset = assets.get_slide(plan.reference_stain)
        if not ref_slide_asset:
            raise FileNotFoundError(f"Reference stain '{plan.reference_stain}' missing")

        # 4. 根据任务类型获取或构建 Reference ROI
        is_mirrored = plan.plan_details.get("is_mirrored", False)
        target_views = plan.requested_views
        # 默认使用首个 view 作为主 view (例如 4x), 次个 view 为高倍 view (例如 20x)
        view_4x = target_views[0] if len(target_views) > 0 else None
        view_20x = target_views[1] if len(target_views) > 1 else None

        crop4_w, crop4_h = view_4x.pixel_dimensions if view_4x else (2257, 1310)
        crop20_w, crop20_h = view_20x.pixel_dimensions if view_20x else (2257, 1310)

        with KFBReader(ref_slide_asset.path, default_mpp=self.cfg.default_mpp) as ref_reader:
            ref_spec = ref_reader.read_metadata()
            ref_lvl4_bgr, ds_ref_l4, _ = ref_reader.read_level_image(4)
            ds_ref_l2 = ref_reader.read_metadata().get_level_downsample(2)
            ref_islands = TissueSegmenter.find_tissue_islands(ref_lvl4_bgr)

            # 初始化拓扑变换图
            graph = TransformGraph(
                crop4_size=(crop4_w, crop4_h),
                crop20_size=(crop20_w, crop20_h),
                ref_ds_lvl2=ds_ref_l2,
                ref_ds_lvl4=ds_ref_l4,
                moving_ds_lvl2=4.0,  # 稍后根据 moving 切片动态重置
                moving_ds_lvl4=16.0,
                acquisition_profile=self.profile,
            )

            # 分支 A: Native ROI 模式 (零反查误差)
            if plan.task_type == TaskType.NATIVE_ROI_MATCH:
                ev = assets.roi_evidence
                center_l0 = ev.native_center_l0
                size_l0 = ev.native_size_l0 or (crop4_w * 5.0, crop4_h * 5.0)
                graph.set_native_reference_roi(center_l0, size_l0)

                # 直接从 Level 0 采样参考切片
                mat_ref4_to_l0 = graph.get_crop4_to_moving_lvl0() # 在参考图上
                ref_4x_extracted = WSISampler.sample_patch(ref_reader, mat_ref4_to_l0, (crop4_w, crop4_h), level=0)
                mat_ref20_to_l0 = graph.get_crop20_to_moving_lvl0()
                ref_20x_extracted = WSISampler.sample_patch(ref_reader, mat_ref20_to_l0, (crop20_w, crop20_h), level=0)

                anchor_info = {"method": "NATIVE_WSI_COORDINATES", "center_l0": center_l0, "inliers": 9999}
                target_ref_island = TissueSegmenter.select_island_by_coordinate(
                    ref_islands, (center_l0[0] / ds_ref_l4, center_l0[1] / ds_ref_l4)
                )

            # 分支 B & C: 截图反查模式 (4x / 20x)
            elif plan.task_type in (TaskType.DUAL_SCALE_REPRODUCE, TaskType.SINGLE_CROP_REPRODUCE, TaskType.HIGH_MAG_ASSISTED):
                ev = assets.roi_evidence
                crop4_path = ev.crop_4x_path or ev.crop_20x_path
                crop4_bgr, (crop4_w, crop4_h) = ImageCropReader.load_crop_bgr(crop4_path, flip_horizontal=is_mirrored)

                anchor_res = self.anchor_locator.locate(crop4_bgr, ref_reader)
                if not anchor_res.is_valid:
                    raise RuntimeError(f"Reference anchor localization rejected for sample {sample_id}")

                graph.set_reference_anchor(anchor_res.mat_crop4_to_lvl4)
                ref_4x_extracted = crop4_bgr

                if ev.has_20x:
                    ref_20x_extracted, (crop20_w, crop20_h) = ImageCropReader.load_crop_bgr(
                        ev.crop_20x_path, flip_horizontal=is_mirrored
                    )
                else:
                    # 仅有 4x 截图时，20x 从原参考 WSI Level 0 中按先验重采样
                    mat_ref20_to_l0 = graph.get_crop20_to_moving_lvl0()
                    ref_20x_extracted = WSISampler.sample_patch(ref_reader, mat_ref20_to_l0, (crop20_w, crop20_h), level=0)

                anchor_info = {
                    "method": anchor_res.localization_method,
                    "center_l0": anchor_res.center_lvl0,
                    "inliers": anchor_res.metrics.inliers,
                }
                target_ref_island = TissueSegmenter.select_island_by_coordinate(ref_islands, anchor_res.center_lvl4)

            # 分支 E: 全片配准模式 (无 ROI 截图)
            else:
                ref_4x_extracted = ref_lvl4_bgr
                ref_20x_extracted = ref_lvl4_bgr
                anchor_info = {"method": "WHOLE_SLIDE_OVERVIEW"}
                target_ref_island = ref_islands[0]

        # 5. 跨染色循环处理 (HE, Gram 等)
        results_by_stain: Dict[str, RegistrationResult] = {}
        all_4x_images = {plan.reference_stain: ref_4x_extracted}
        all_20x_images = {plan.reference_stain: ref_20x_extracted}
        overall_verdict = RunVerdict.PASS

        for stain in plan.target_stains_available:
            moving_asset = assets.get_slide(stain)
            if not moving_asset:
                continue

            print(f"\n2. Registering stain [{stain}]...")
            with KFBReader(moving_asset.path, default_mpp=self.cfg.default_mpp) as moving_reader:
                moving_lvl4_bgr, ds_mov_l4, _ = moving_reader.read_level_image(4)
                moving_spec = moving_reader.read_metadata()
                ds_mov_l2 = moving_spec.get_level_downsample(2)

                graph.moving_ds_lvl2 = ds_mov_l2
                graph.moving_ds_lvl4 = ds_mov_l4

                # 物理预期尺度比: Moving L4 -> Ref L4 映射的理论像素缩放为 MPP_moving / MPP_ref
                expected_scale = moving_spec.mpp_x / max(1e-5, ref_spec.mpp_x)

                global_res = self.global_registrar.register_stain(
                    moving_lvl4_bgr,
                    ref_lvl4_bgr,
                    target_ref_island,
                    expected_scale_from_mpp=expected_scale,
                )

                if not global_res.is_valid or global_res.mat_moving_to_ref_lvl4 is None:
                    print(f"   [ABSTAIN] Global alignment failed for {stain}")
                    overall_verdict = RunVerdict.ABSTAIN
                    results_by_stain[stain] = RegistrationResult(
                        sample_id=sample_id,
                        moving_stain=stain,
                        reference_stain=plan.reference_stain,
                        status=RegistrationStatus.ABSTAIN,
                        reason=global_res.details.get("reason", "Global alignment failed"),
                        metrics=global_res.metrics,
                    )
                    continue

                graph.set_global_cross_stain(global_res.mat_moving_to_ref_lvl4)

                # 4x 初始提取与局部微调
                mat_crop4_to_mov_l2 = graph.get_crop4_to_moving_lvl2()
                initial_moving_4x = WSISampler.sample_patch(
                    moving_reader,
                    mat_crop4_to_mov_l2,
                    (crop4_w, crop4_h),
                    level=2,
                )

                local_res = self.local_refiner.refine(initial_moving_4x, ref_4x_extracted)
                graph.set_local_refinement(local_res.mat_local_3x3)
                aligned_4x = local_res.aligned_image_bgr

                # 根据 requested_views 动态提取所有倍率视场 (如 4x, 10x, 20x, 40x)
                stain_extracted_views: Dict[str, np.ndarray] = {}
                for v in target_views:
                    if v.name.lower() in ("4x", "overview"):
                        stain_extracted_views[v.name] = aligned_4x
                    else:
                        mat_v_to_mov_l0 = graph.get_view_to_moving_lvl0(
                            target_mag=v.magnification_approx,
                            base_mag=4.0,
                            target_size=v.pixel_dimensions,
                        )
                        view_img = WSISampler.sample_patch(
                            moving_reader,
                            mat_v_to_mov_l0,
                            v.pixel_dimensions,
                            level=0,
                        )
                        stain_extracted_views[v.name] = view_img

                # QC 判定 (使用统一的 expected_scale 检验物理残差尺度)
                qc_status, qc_failure_code, qc_reason = self.qc_engine.evaluate_cross_stain(
                    global_res.metrics, expected_scale_from_mpp=expected_scale
                )
                if qc_status == RegistrationStatus.ABSTAIN:
                    overall_verdict = RunVerdict.ABSTAIN
                elif qc_status == RegistrationStatus.WARN and overall_verdict == RunVerdict.PASS:
                    overall_verdict = RunVerdict.REVIEW
                results_by_stain[stain] = RegistrationResult(
                    sample_id=sample_id,
                    moving_stain=stain,
                    reference_stain=plan.reference_stain,
                    status=qc_status,
                    reason=qc_reason,
                    failure_code=qc_failure_code,
                    transform_matrix_3x3=global_res.mat_moving_to_ref_lvl4.tolist(),
                    metrics=global_res.metrics,
                    qc_details={
                        "global": global_res.details,
                        "local": local_res.details,
                        "extracted_views": list(stain_extracted_views.keys()),
                    },
                )

        # 6. 确定产物落盘目录 (final / review / debug)
        target_out_dir = resolve_artifact_dir(sample_base_out, overall_verdict)
        print(f"\n[Routing] Overall Verdict: {overall_verdict.value} -> Saving to: {target_out_dir}")

        # 保存参考图像与移动对齐图像 (遵循配置的 DPI 与视图)
        dpi_tuple = (self.cfg.dpi, self.cfg.dpi)
        ref_name = plan.reference_stain.capitalize()
        ImageCropReader.save_publication_tiff(
            ref_4x_extracted,
            target_out_dir / f"{sample_id}-{ref_name}-4x-{self.cfg.dpi}dpi.tif",
            dpi=dpi_tuple,
        )
        ImageCropReader.save_publication_tiff(
            ref_20x_extracted,
            target_out_dir / f"{sample_id}-{ref_name}-20x-{self.cfg.dpi}dpi.tif",
            dpi=dpi_tuple,
        )

        for stain, res in results_by_stain.items():
            if res.status != RegistrationStatus.ABSTAIN:
                # 保存所有提取的视图
                extracted_views = res.qc_details.get("extracted_views", ["4x", "20x"])
                for v_name in extracted_views:
                    v_img = aligned_4x if v_name.lower() in ("4x", "overview") else aligned_20x
                    ImageCropReader.save_publication_tiff(
                        v_img,
                        target_out_dir / f"{sample_id}-{stain}-{v_name}-aligned-{self.cfg.dpi}dpi.tif",
                        dpi=dpi_tuple,
                    )
                if self.cfg.save_overlays:
                    ImageCropReader.save_overlay_png(
                        ref_4x_extracted,
                        aligned_4x,
                        target_out_dir / f"overlay-{stain}-4x-aligned.png",
                        dpi=dpi_tuple,
                    )
                    ImageCropReader.save_overlay_png(
                        ref_20x_extracted,
                        aligned_20x,
                        target_out_dir / f"overlay-{stain}-20x-aligned.png",
                        dpi=dpi_tuple,
                    )
        # 生成 Contact Sheets
        if self.cfg.save_contact_sheets and overall_verdict != RunVerdict.ABSTAIN:
            ContactSheetGenerator.create_contact_sheet(
                all_4x_images,
                f"Sample {sample_id} - 4x Aligned Cross-Stain Comparison ({overall_verdict.value})",
                target_out_dir / f"contact_sheet_4x.png",
            )
            ContactSheetGenerator.create_contact_sheet(
                all_20x_images,
                f"Sample {sample_id} - 20x Aligned Cross-Stain Comparison ({overall_verdict.value})",
                target_out_dir / f"contact_sheet_20x.png",
            )

        # 7. 保存结构化 JSON 报告
        elapsed = time.time() - start_time
        report = ReportGenerator.save_sample_report(
            sample_id=sample_id,
            results=results_by_stain,
            anchor_info=anchor_info,
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
            out_path=target_out_dir / "registration_report.json",
            elapsed_seconds=elapsed,
        )
        report["overall_status"] = overall_verdict.value
        report["artifact_tier"] = target_out_dir.name

        return report
