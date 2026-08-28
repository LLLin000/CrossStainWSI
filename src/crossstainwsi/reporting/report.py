"""
配准结果结构化 JSON 报告生成器
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from crossstainwsi.domain import QCMetrics, RegistrationResult, RegistrationStatus


class ReportGenerator:
    """
    负责将单样本或全批次配准结果序列化为可审计的 JSON 报告
    """
    @staticmethod
    def save_sample_report(
        sample_id: str,
        results: Dict[str, RegistrationResult],
        anchor_info: Dict[str, Any],
        mpp_info: Dict[str, Any],
        fov_info: Dict[str, Any],
        out_path: Path,
        elapsed_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        overall_status = "PASS"
        stain_reports = {}

        for stain, res in results.items():
            stain_reports[stain] = {
                "status": res.status.value,
                "reason": res.reason,
                "metrics": {
                    "inliers": res.metrics.inliers,
                    "matches": res.metrics.matches,
                    "inlier_ratio": round(res.metrics.inlier_ratio, 4),
                    "spatial_coverage": round(res.metrics.spatial_coverage, 4),
                    "median_reproj_error": round(res.metrics.median_reproj_error, 4),
                    "scale": round(res.metrics.scale, 4),
                    "rotation_deg": round(res.metrics.rotation_deg, 2),
                    "method": res.metrics.method,
                },
                "qc_details": res.qc_details,
            }
            if res.status in (RegistrationStatus.FAIL, RegistrationStatus.ABSTAIN):
                overall_status = res.status.value
            elif res.status == RegistrationStatus.WARN and overall_status == "PASS":
                overall_status = "WARN"

        report_data = {
            "sample_id": sample_id,
            "overall_status": overall_status,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "mpp": mpp_info,
            "physical_fov": fov_info,
            "reference_anchor": anchor_info,
            "stains": stain_reports,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_data
