"""
批量样本配准执行器 (BatchRunner)
安全遍历样本集合，支持断点续跑与统计汇总报告，无任何自动关机行为
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from crossstainwsi.matching.loftr import LoFTRMatcher
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.pipeline.sample_runner import SampleRunner


class BatchRunner:
    """
    负责执行多样本批处理
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        # 共享单个 LoFTR 实例，避免重复加载模型导致显存碎片化
        device = self.cfg.get_torch_device()
        print(f"[BatchRunner] Initializing shared LoFTR model on {device}...")
        self.loftr = LoFTRMatcher(device=device)
        self.sample_runner = SampleRunner(config=self.cfg, loftr_matcher=self.loftr)

    def run_batch(self, sample_ids: List[str], overwrite: bool = False) -> Dict[str, Any]:
        print(f"================ Starting Batch Processing ({len(sample_ids)} samples) ================")
        start_time = time.time()
        summary = {
            "total_samples": len(sample_ids),
            "processed": 0,
            "success": 0,
            "warn": 0,
            "abstain_or_fail": 0,
            "results": {},
            "elapsed_seconds": 0.0,
        }

        for idx, sample_id in enumerate(sample_ids, 1):
            print(f"\n>>> [{idx}/{len(sample_ids)}] Processing sample: {sample_id}")
            report_file = self.cfg.output_dir / sample_id / "registration_report.json"

            if report_file.exists() and not overwrite:
                print(f"   [Skip] Report already exists for {sample_id}, loading cached summary.")
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        rep = json.load(f)
                    summary["results"][sample_id] = rep
                    summary["processed"] += 1
                    status = rep.get("overall_status", "PASS")
                    if status == "PASS":
                        summary["success"] += 1
                    elif status == "WARN":
                        summary["warn"] += 1
                    else:
                        summary["abstain_or_fail"] += 1
                    continue
                except Exception as e:
                    print(f"   [WARN] Failed to read cached report for {sample_id}: {e}. Re-running...")

            try:
                rep = self.sample_runner.process(sample_id)
                summary["results"][sample_id] = rep
                summary["processed"] += 1
                status = rep.get("overall_status", "PASS")
                if status == "PASS":
                    summary["success"] += 1
                elif status == "WARN":
                    summary["warn"] += 1
                else:
                    summary["abstain_or_fail"] += 1
            except Exception as e:
                print(f"   [ERROR] Sample {sample_id} failed with exception: {e}")
                summary["results"][sample_id] = {
                    "sample_id": sample_id,
                    "overall_status": "EXCEPTION",
                    "error": str(e),
                }
                summary["processed"] += 1
                summary["abstain_or_fail"] += 1

            # 每次处理完一个样本即时保存一次 batch_summary.json (断点安全)
            summary["elapsed_seconds"] = round(time.time() - start_time, 2)
            summary_path = self.cfg.output_dir / "batch_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n================ Batch Processing Complete ================")
        print(f"Total: {summary['total_samples']} | Success: {summary['success']} | Warn: {summary['warn']} | Fail/Abstain: {summary['abstain_or_fail']}")
        print(f"Total Elapsed Time: {summary['elapsed_seconds']:.2f}s")
        return summary
