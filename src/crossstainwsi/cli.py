"""
CrossStainWSI 极简命令行接口 (CLI)
参数设计与 GUI 完全解耦，支持直接无缝对接未来图形界面
"""

import argparse
from pathlib import Path
import sys
from typing import List

from crossstainwsi.inventory.discover import AssetDiscoverer
from crossstainwsi.pipeline.batch_runner import BatchRunner
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.pipeline.sample_runner import SampleRunner
from crossstainwsi.planning.goal import StainRequirement, UserGoal
from crossstainwsi.planning.planner import WorkflowPlanner


def main(args: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crossstainwsi",
        description="CrossStainWSI: Auditable Cross-Stain Whole-Slide Image Registration & Multi-Scale Extraction Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. discover (资产扫描)
    disc_parser = subparsers.add_parser("discover", help="Discover available WSI slides and existing ROI evidence")
    disc_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    disc_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")

    # 2. plan (工作流规划预览)
    plan_parser = subparsers.add_parser("plan", help="Generate and inspect the ExecutionPlan for a sample")
    plan_parser.add_argument("sample_id", type=str, help="Sample identifier (e.g. 4w-5-3, 2-2w-1)")
    plan_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    plan_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    plan_parser.add_argument("--ref-stain", type=str, default="masson", help="Reference stain name (default: masson)")

    # 3. run (单样本执行)
    run_parser = subparsers.add_parser("run", help="Execute registration and extraction on a single sample")
    run_parser.add_argument("sample_id", type=str, help="Sample identifier (e.g. 4w-5-3, 2-2w-1)")
    run_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    run_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    run_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
    run_parser.add_argument("--ref-stain", type=str, default="masson", help="Reference stain name (default: masson)")
    run_parser.add_argument("--stains", nargs="+", type=str, default=None, help="Target stains to register (e.g. HE Gram)")
    run_parser.add_argument("--views", "-m", nargs="+", type=str, default=["4x", "20x"], help="Requested output magnification views (e.g. 4x 20x 10x)")
    run_parser.add_argument("--dpi", type=int, default=300, help="Output image DPI (default: 300)")
    run_parser.add_argument("--scale-ratio", type=float, default=5.0, help="Magnification sampling scale ratio between 20x and 4x (default: 5.0)")
    run_parser.add_argument("--mirror", action="store_true", help="Force horizontal mirror correction for input crop")
    run_parser.add_argument("--no-overlay", action="store_true", help="Disable generating overlay comparison images")
    run_parser.add_argument("--contact-sheet", action="store_true", help="Enable generating side-by-side contact sheets")
    run_parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    # 4. batch (多样本批量执行)
    batch_parser = subparsers.add_parser("batch", help="Execute registration on a batch of samples")
    batch_parser.add_argument("sample_ids", nargs="+", type=str, help="List of sample identifiers")
    batch_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    batch_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    batch_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
    batch_parser.add_argument("--ref-stain", type=str, default="masson", help="Reference stain name (default: masson)")
    batch_parser.add_argument("--stains", nargs="+", type=str, default=None, help="Target stains to register (e.g. HE Gram)")
    batch_parser.add_argument("--views", "-m", nargs="+", type=str, default=["4x", "20x"], help="Requested output magnification views (e.g. 4x 20x 10x)")
    batch_parser.add_argument("--dpi", type=int, default=300, help="Output image DPI (default: 300)")
    batch_parser.add_argument("--scale-ratio", type=float, default=5.0, help="Sampling scale ratio (default: 5.0)")
    batch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed sample reports")
    batch_parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parsed = parser.parse_args(args)

    if parsed.command == "discover":
        cfg = PipelineConfig(base_dir=parsed.base_dir, tiff_dir=parsed.tiff_dir)
        discoverer = AssetDiscoverer(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            mirrored_sample_ids=cfg.mirrored_samples,
        )
        inventory = discoverer.discover()
        summary = inventory.summary()
        print("\n================ [Asset Discovery Summary] ================")
        print(f"Discovered {summary['total_samples']} samples in {parsed.base_dir}")
        print(f"Samples with crop evidence: {len(summary['samples_with_crops'])}")
        print(f"Samples WSI only (no crops): {len(summary['samples_wsi_only'])}")
        print("\nDetail sample list:")
        for sid, sample in sorted(inventory.samples.items()):
            print("  *", sample.describe())
        return 0

    elif parsed.command == "plan":
        cfg = PipelineConfig(base_dir=parsed.base_dir, tiff_dir=parsed.tiff_dir)
        discoverer = AssetDiscoverer(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            mirrored_sample_ids=cfg.mirrored_samples,
        )
        inventory = discoverer.discover()
        sample_assets = inventory.get_sample(parsed.sample_id)
        if not sample_assets:
            print(f"[ERROR] Sample '{parsed.sample_id}' not found in {parsed.base_dir}")
            return 1

        goal = UserGoal(
            reference_stain=parsed.ref_stain,
            stain_requirements=[
                StainRequirement("HE", is_required=True),
                StainRequirement("Gram", is_required=False),
            ],
        )
        planner = WorkflowPlanner(goal=goal)
        plan = planner.plan(sample_assets)
        print("\n" + plan.describe() + "\n")
        return 0

    elif parsed.command == "run":
        mirrored_set = set(PipelineConfig().mirrored_samples)
        if parsed.mirror:
            mirrored_set.add(parsed.sample_id)

        moving = parsed.stains or ["HE", "Gram"]
        goal = UserGoal.from_magnifications(
            mags=parsed.views,
            reference_stain=parsed.ref_stain,
            target_stains=moving,
            dpi=parsed.dpi,
        )
        cfg = PipelineConfig(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            output_dir=parsed.out_dir,
            reference_stain=parsed.ref_stain,
            moving_stains=moving,
            mirrored_samples=mirrored_set,
            dpi=parsed.dpi,
            sampling_scale_ratio=parsed.scale_ratio,
            save_overlays=not parsed.no_overlay,
            save_contact_sheets=parsed.contact_sheet,
            device=parsed.device,
        )
        runner = SampleRunner(config=cfg, goal=goal)
        runner.process(parsed.sample_id)
        return 0

    elif parsed.command == "batch":
        moving = parsed.stains or ["HE", "Gram"]
        cfg = PipelineConfig(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            output_dir=parsed.out_dir,
            reference_stain=parsed.ref_stain,
            moving_stains=moving,
            dpi=parsed.dpi,
            sampling_scale_ratio=parsed.scale_ratio,
            device=parsed.device,
        )
        runner = BatchRunner(config=cfg)
        runner.run_batch(parsed.sample_ids, overwrite=parsed.overwrite)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
