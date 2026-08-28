"""
CrossStainWSI 命令行入口 (CLI)
"""

import argparse
from pathlib import Path
import sys
from typing import List

from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.pipeline.batch_runner import BatchRunner
from crossstainwsi.pipeline.sample_runner import SampleRunner


def main(args: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crossstainwsi",
        description="CrossStainWSI: Auditable Cross-Stain Histology WSI Registration Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. run (单样本)
    run_parser = subparsers.add_parser("run", help="Run registration on a single sample")
    run_parser.add_argument("sample_id", type=str, help="Sample identifier (e.g. 4w-5-3, 2-2w-1)")
    run_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    run_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    run_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
    run_parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")

    # 2. batch (多样本)
    batch_parser = subparsers.add_parser("batch", help="Run registration on a batch of samples")
    batch_parser.add_argument("sample_ids", nargs="+", type=str, help="List of sample identifiers")
    batch_parser.add_argument("--base-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"), help="Base WSI directory")
    batch_parser.add_argument("--tiff-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\tiff"), help="TIFF crop directory")
    batch_parser.add_argument("--out-dir", type=Path, default=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"), help="Output directory")
    batch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed sample reports")
    batch_parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        cfg = PipelineConfig(
            base_dir=parsed.base_dir,
            tiff_dir=parsed.tiff_dir,
            output_dir=parsed.out_dir,
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
