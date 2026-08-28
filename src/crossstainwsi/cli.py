"""
CrossStainWSI 增强型命令行接口 (CLI)
支持 discover (资产发现), plan (工作流规划), run (单样本执行), batch (批处理)
"""

import argparse
import json
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
        description="CrossStainWSI: Auditable Cross-Stain Histology WSI Registration & Workflow Toolkit",
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
    plan_parser.add_argument("--ref-stain", type=str, default="masson", help="Reference stain name")

    # 3. run (单样本执行)
    run_parser = subparsers.add_parser("run", help="Execute registration and extraction on a single sample")
    run_parser.add_argument("sample_id", type=str, help="Sample identifier (e.g. 4w-5-3, 2-2w-1)")
    run_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    run_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    run_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
    run_parser.add_argument("--ref-stain", type=str, default="masson", help="Reference stain name")
    run_parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")

    # 4. batch (多样本批量执行)
    batch_parser = subparsers.add_parser("batch", help="Execute registration on a batch of samples")
    batch_parser.add_argument("sample_ids", nargs="+", type=str, help="List of sample identifiers")
    batch_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    batch_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    batch_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
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
        cfg = PipelineConfig(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            output_dir=parsed.out_dir,
            reference_stain=parsed.ref_stain,
            device=parsed.device,
        )
        runner = SampleRunner(config=cfg)
        runner.process(parsed.sample_id)
        return 0

    elif parsed.command == "batch":
        cfg = PipelineConfig(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            output_dir=parsed.out_dir,
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
