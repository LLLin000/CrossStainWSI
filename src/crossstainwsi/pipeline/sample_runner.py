"""
自适应工作流执行器 (SampleRunner)
实现输入证据与输出要求的完全解耦、Anchor坐标系局部微调、双尺度独立验证与全视图 Level 0 单次采样
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.domain import EvidenceView, FailureCode, QCMetrics, RegistrationResult, RegistrationStatus
from crossstainwsi.inventory.assets import SampleAssets
from crossstainwsi.inventory.discover import AssetDiscoverer
from crossstainwsi.io.image import ImageCropReader
from crossstainwsi.io.kfb import KFBReader
from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.planning.acquisition import AcquisitionProfile
from crossstainwsi.planning.execution_plan import ExecutionPlan, TaskType
from crossstainwsi.planning.goal import UserGoal, ViewSpec
from crossstainwsi.planning.planner import WorkflowPlanner
from crossstainwsi.qc.metrics import compute_same_image_metrics
from crossstainwsi.qc.rules import QCRuleEngine
from crossstainwsi.registration.global_reg import GlobalRegistrar
from crossstainwsi.registration.local_reg import LocalRefiner
from crossstainwsi.registration.reference_anchor import ReferenceAnchorLocator
from crossstainwsi.reporting.contact_sheet import ContactSheetGenerator
from crossstainwsi.reporting.report import ReportGenerator
from crossstainwsi.review.states import ArtifactTier, ConfidenceTier, RunVerdict, resolve_artifact_dir
from crossstainwsi.sampling.sampler import WSISampler
from crossstainwsi.tissue.islands import TissueSegmenter
from crossstainwsi.transforms.geom import affine, apply_mat, h, scale_matrix
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

        # 4. 根据输入证据构建真实 EvidenceView (输入与输出彻底解耦)
        ev = assets.roi_evidence
        is_mirrored_flag = plan.plan_details.get("is_mirrored", False)
        target_views = plan.requested_views

        with KFBReader(ref_slide_asset.path, default_mpp=self.cfg.default_mpp) as ref_reader:
            ref_spec = ref_reader.read_metadata()
            ref_lvl4_bgr, ds_ref_l4, _ = ref_reader.read_level_image(4)
            ds_ref_l2 = ref_reader.read_metadata().get_level_downsample(2)
            ref_islands = TissueSegmenter.find_tissue_islands(ref_lvl4_bgr)

            ref_extracted_views: Dict[str, np.ndarray] = {}

            # 分支 A: Native ROI 模式 (0 锚点误差)
            if plan.task_type == TaskType.NATIVE_ROI_MATCH:
                center_l0 = ev.native_center_l0
                # 默认 Level-0 像素尺寸 (例如 4x 标准视场对应 2257 * 5 = 11285 Level-0 px)
                size_l0 = ev.native_size_l0 or (11285, 6550)

                anchor_ev_view = EvidenceView(
                    id="native_anchor",
                    width_px=2257,
                    height_px=1310,
                    nominal_magnification=4.0,
                )

                graph = TransformGraph(
                    anchor_view=anchor_ev_view,
                    ref_ds_lvl2=ds_ref_l2,
                    ref_ds_lvl4=ds_ref_l4,
                    acquisition_profile=self.profile,
                )
                graph.set_native_reference_roi(center_l0, size_l0)

                # 直接从 Level 0 采样参考切片的各目标视图
                for v in target_views:
                    m_view_to_l0 = self._get_native_view_to_l0_matrix(center_l0, size_l0, v, ref_spec.mpp_x)
                    view_img = WSISampler.sample_patch(ref_reader, m_view_to_l0, v.pixel_dimensions, level=0)
                    ref_extracted_views[v.name] = view_img

                anchor_info = {"method": "NATIVE_WSI_COORDINATES", "center_l0": center_l0, "inliers": 9999}
                target_ref_island = TissueSegmenter.select_island_by_coordinate(
                    ref_islands, (center_l0[0] / ds_ref_l4, center_l0[1] / ds_ref_l4)
                )

            # 分支 B & C: 截图证据反查模式
            elif plan.task_type in (TaskType.DUAL_SCALE_REPRODUCE, TaskType.SINGLE_CROP_REPRODUCE, TaskType.HIGH_MAG_ASSISTED):
                v_primary = ev.get_primary_anchor_evidence()
                if not v_primary or not v_primary.source_path:
                    v_primary = EvidenceView(id="anchor_fallback", width_px=2257, height_px=1310, nominal_magnification=4.0, source_path=ev.crop_4x_path or ev.crop_20x_path)

                # 证据原图绝对不预翻转 (flip_horizontal=False)，完全由 ReferenceAnchorLocator 处理
                raw_crop_bgr, (crop_w, crop_h) = ImageCropReader.load_crop_bgr(v_primary.source_path, flip_horizontal=False)

                anchor_ev_view = EvidenceView(
                    id=v_primary.id,
                    width_px=crop_w,
                    height_px=crop_h,
                    nominal_magnification=v_primary.nominal_magnification,
                    source_path=v_primary.source_path,
                )

                graph = TransformGraph(
                    anchor_view=anchor_ev_view,
                    ref_ds_lvl2=ds_ref_l2,
                    ref_ds_lvl4=ds_ref_l4,
                    acquisition_profile=self.profile,
                )

                # 运行双奇偶性锚点定位
                anchor_res = self.anchor_locator.locate(
                    anchor_ev_view,
                    ref_reader,
                    force_mirror=True if is_mirrored_flag else None,
                    crop_bgr_override=raw_crop_bgr,
                )

                if not anchor_res.is_valid:
                    print(f"   [ABSTAIN] Reference anchor rejected: {anchor_res.failure_code.value}")
                    debug_out = resolve_artifact_dir(sample_base_out, RunVerdict.ABSTAIN)
                    report = ReportGenerator.save_sample_report(
                        sample_id=sample_id,
                        results={},
                        anchor_info={"error": anchor_res.failure_code.value, "details": anchor_res.details},
                        mpp_info={"mpp_x": ref_spec.mpp_x, "mpp_y": ref_spec.mpp_y, "provenance": ref_spec.mpp_source},
                        fov_info=self._compute_fov_info(target_views, ref_spec.mpp_x),
                        out_path=debug_out / "registration_report.json",
                        elapsed_seconds=time.time() - start_time,
                    )
                    report["overall_status"] = RunVerdict.ABSTAIN.value
                    return report

                graph.set_reference_anchor(anchor_res.mat_anchor_to_lvl4)

                # 双尺度独立核验：若存在高倍辅助证据 (如 20x)，在参考切片上执行独立一致性验证
                v_secondary = ev.get_secondary_verification_evidence()
                if v_secondary and v_secondary.source_path:
                    sec_raw_bgr, (sec_w, sec_h) = ImageCropReader.load_crop_bgr(v_secondary.source_path, flip_horizontal=False)
                    m_sec_to_ref_l0 = self._get_ref_view_to_l0_matrix(graph, v_secondary, ds_ref_l2)
                    sec_predicted_ref = WSISampler.sample_patch(ref_reader, m_sec_to_ref_l0, (sec_w, sec_h), level=0)
                    sec_qc = compute_same_image_metrics(sec_raw_bgr, sec_predicted_ref)

                    if sec_qc.inliers < 4 and (sec_qc.ncc_score or 0.0) < 0.20:
                        print(f"   [ABSTAIN] Secondary evidence cross-scale conflict (inliers={sec_qc.inliers}, ncc={sec_qc.ncc_score})")
                        debug_out = resolve_artifact_dir(sample_base_out, RunVerdict.ABSTAIN)
                        report = ReportGenerator.save_sample_report(
                            sample_id=sample_id,
                            results={},
                            anchor_info={"error": FailureCode.CROSS_SCALE_CONFLICT.value, "details": sec_qc.details},
                            mpp_info={"mpp_x": ref_spec.mpp_x, "mpp_y": ref_spec.mpp_y, "provenance": ref_spec.mpp_source},
                            fov_info=self._compute_fov_info(target_views, ref_spec.mpp_x),
                            out_path=debug_out / "registration_report.json",
                            elapsed_seconds=time.time() - start_time,
                        )
                        report["overall_status"] = RunVerdict.ABSTAIN.value
                        return report

                # 统一从 Reference WSI Level 0 单次重采样所有请求的输出视图 (保证完全相同的坐标系与未压缩画质)
                for v in target_views:
                    m_v_to_ref_l0 = self._get_ref_view_to_l0_matrix(graph, v, ds_ref_l2)
                    view_img = WSISampler.sample_patch(ref_reader, m_v_to_ref_l0, v.pixel_dimensions, level=0)
                    ref_extracted_views[v.name] = view_img

                anchor_info = {
                    "method": anchor_res.localization_method,
                    "center_l0": anchor_res.center_lvl0,
                    "inliers": anchor_res.metrics.inliers,
                    "is_mirrored": anchor_res.is_mirrored,
                }
                target_ref_island = TissueSegmenter.select_island_by_coordinate(ref_islands, anchor_res.center_lvl4)

            # 分支 E: 全片配准模式 (无任何截图)
            else:
                anchor_ev_view = EvidenceView(id="overview", width_px=ref_lvl4_bgr.shape[1], height_px=ref_lvl4_bgr.shape[0], nominal_magnification=4.0)
                graph = TransformGraph(anchor_view=anchor_ev_view, ref_ds_lvl2=ds_ref_l2, ref_ds_lvl4=ds_ref_l4, acquisition_profile=self.profile)
                graph.set_reference_anchor(np.eye(3))
                for v in target_views:
                    ref_extracted_views[v.name] = ref_lvl4_bgr
                anchor_info = {"method": "WHOLE_SLIDE_OVERVIEW"}
                target_ref_island = ref_islands[0]

        # 5. 跨染色循环处理 (HE, Gram 等)
        results_by_stain: Dict[str, RegistrationResult] = {}
        stain_extracted_views: Dict[str, Dict[str, np.ndarray]] = {}
        overall_verdict = RunVerdict.PASS

        # 提取 Anchor Frame 上的基准参考图像用于 Local Refinement (确保坐标系与物理 FOV 严格一致)
        with KFBReader(ref_slide_asset.path, default_mpp=self.cfg.default_mpp) as ref_reader:
            anchor_ref_img = WSISampler.sample_patch(
                ref_reader,
                graph.mat_anchor_to_ref_lvl2,
                (anchor_ev_view.width_px, anchor_ev_view.height_px),
                level=2,
            )

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

                # 物理预期尺度比: Matching Level 4 理论像素缩放为 (MPP_moving_L0 * DS_mov_L4) / (MPP_ref_L0 * DS_ref_L4)
                moving_effective_mpp = moving_spec.mpp_x * ds_mov_l4
                ref_effective_mpp = ref_spec.mpp_x * ds_ref_l4
                expected_scale = moving_effective_mpp / max(1e-5, ref_effective_mpp)

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
                        failure_code=FailureCode.FEATURE_MATCH_WEAK,
                        reason=global_res.details.get("reason", "Global alignment failed"),
                        metrics=global_res.metrics,
                    )
                    continue

                graph.set_global_cross_stain(global_res.mat_moving_to_ref_lvl4)

                # 在严格相同的 Anchor Frame (相同的像素尺寸与物理视场) 中提取 moving patch 进行局部微调
                mat_anchor_to_mov_l2 = graph.get_anchor_to_moving_lvl2()
                anchor_mov_img = WSISampler.sample_patch(
                    moving_reader,
                    mat_anchor_to_mov_l2,
                    (anchor_ev_view.width_px, anchor_ev_view.height_px),
                    level=2,
                )

                local_res = self.local_refiner.refine(anchor_mov_img, anchor_ref_img)
                graph.set_local_refinement(local_res.mat_local_3x3)

                # 所有请求的目标视图统一直接从 Moving Slide Level 0 单次重采样 (恪守只有一次插值原则)
                stain_extracted_views[stain] = {}
                for v in target_views:
                    mat_v_to_mov_l0 = graph.get_view_to_moving_lvl0(target_view=v)
                    view_img = WSISampler.sample_patch(
                        moving_reader,
                        mat_v_to_mov_l0,
                        v.pixel_dimensions,
                        level=0,
                    )
                    stain_extracted_views[stain][v.name] = view_img

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
                        "extracted_views": list(stain_extracted_views[stain].keys()),
                    },
                )

        # 6. 确定产物落盘目录 (final / review / debug)
        target_out_dir = resolve_artifact_dir(sample_base_out, overall_verdict)
        print(f"\n[Routing] Overall Verdict: {overall_verdict.value} -> Saving to: {target_out_dir}")

        dpi_tuple = (self.cfg.dpi, self.cfg.dpi)
        ref_name = plan.reference_stain.capitalize()

        # 保存参考切片各视图
        for v in target_views:
            if v.name in ref_extracted_views:
                ImageCropReader.save_publication_tiff(
                    ref_extracted_views[v.name],
                    target_out_dir / f"{sample_id}-{ref_name}-{v.name}-{self.cfg.dpi}dpi.tif",
                    dpi=dpi_tuple,
                )

        # 保存移动切片各对齐视图与 Overlays
        for stain, res in results_by_stain.items():
            if res.status != RegistrationStatus.ABSTAIN and stain in stain_extracted_views:
                for v in target_views:
                    if v.name in stain_extracted_views[stain]:
                        v_img = stain_extracted_views[stain][v.name]
                        ImageCropReader.save_publication_tiff(
                            v_img,
                            target_out_dir / f"{sample_id}-{stain}-{v.name}-aligned-{self.cfg.dpi}dpi.tif",
                            dpi=dpi_tuple,
                        )
                        if self.cfg.save_overlays and v.name in ref_extracted_views:
                            ImageCropReader.save_overlay_png(
                                ref_extracted_views[v.name],
                                v_img,
                                target_out_dir / f"overlay-{stain}-{v.name}-aligned.png",
                                dpi=dpi_tuple,
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
            fov_info=self._compute_fov_info(target_views, ref_spec.mpp_x),
            out_path=target_out_dir / "registration_report.json",
            elapsed_seconds=elapsed,
        )
        report["overall_status"] = overall_verdict.value
        report["artifact_tier"] = target_out_dir.name

        return report

    def _compute_fov_info(self, views: List[ViewSpec], ref_l0_mpp: float) -> Dict[str, float]:
        fov_info = {}
        for v in views:
            effective_mpp = ref_l0_mpp * (20.0 / max(0.1, v.magnification_approx))
            fov_info[f"{v.name}_width_um"] = v.pixel_dimensions[0] * effective_mpp
            fov_info[f"{v.name}_height_um"] = v.pixel_dimensions[1] * effective_mpp
        return fov_info

    def _get_ref_view_to_l0_matrix(self, graph: TransformGraph, view: Any, ds_ref_l2: float) -> np.ndarray:
        scale_l2_to_l0 = scale_matrix(ds_ref_l2, ds_ref_l2)
        m_view_to_anchor = self.profile.derive_view_to_anchor_matrix(graph.anchor_view, view)
        return scale_l2_to_l0 @ graph.mat_anchor_to_ref_lvl2 @ m_view_to_anchor

    def _get_native_view_to_l0_matrix(
        self,
        center_l0: Tuple[float, float],
        size_l0: Tuple[float, float],
        view: ViewSpec,
        ref_l0_mpp: float,
    ) -> np.ndarray:
        cx, cy = center_l0
        w_px, h_px = view.pixel_dimensions
        # 视场物理缩放: 20x 对应 1x Level 0 像素跨度
        view_to_l0_ratio = 20.0 / max(0.1, view.magnification_approx)
        w_l0 = float(w_px * view_to_l0_ratio)
        h_l0 = float(h_px * view_to_l0_ratio)
        sx = w_l0 / max(1.0, float(w_px))
        sy = h_l0 / max(1.0, float(h_px))
        tx = cx - (w_l0 / 2.0)
        ty = cy - (h_l0 / 2.0)
        return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
