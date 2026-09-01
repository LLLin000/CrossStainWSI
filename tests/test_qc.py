from crossstainwsi.domain import FailureCode, QCMetrics, RegistrationStatus
from crossstainwsi.qc.rules import QCRuleEngine


def test_qc_rule_engine():
    engine = QCRuleEngine()

    # 1. 强对齐应获得 PASS
    strong_metrics = QCMetrics(
        inliers=80,
        matches=150,
        inlier_ratio=0.53,
        spatial_coverage=0.6,
        median_reproj_error=1.5,
        scale=1.002,
        rotation_deg=0.5,
    )
    status, failure_code, _ = engine.evaluate_cross_stain(strong_metrics)
    assert status == RegistrationStatus.PASS
    assert failure_code == FailureCode.NONE

    # 2. 内点极低应 ABSTAIN
    weak_metrics = QCMetrics(
        inliers=8,
        matches=150,
        inlier_ratio=0.05,
        spatial_coverage=0.1,
        scale=1.0,
    )
    status, failure_code, _ = engine.evaluate_cross_stain(weak_metrics)
    assert status == RegistrationStatus.ABSTAIN
    assert failure_code == FailureCode.FEATURE_MATCH_WEAK

    # 3. 异常缩放形变应 ABSTAIN
    deformed_metrics = QCMetrics(
        inliers=100,
        matches=150,
        inlier_ratio=0.66,
        spatial_coverage=0.6,
        scale=1.45,  # 严重异常尺度
    )
    status, failure_code, _ = engine.evaluate_cross_stain(deformed_metrics)
    assert status == RegistrationStatus.ABSTAIN
    assert failure_code == FailureCode.STRUCTURE_CONFLICT

    # 4. 临界质量应 WARN (归因为切片对应弱)
    marginal_metrics = QCMetrics(
        inliers=25,
        matches=100,
        inlier_ratio=0.25,
        spatial_coverage=0.25,
        scale=1.01,
    )
    status, failure_code, _ = engine.evaluate_cross_stain(marginal_metrics)
    assert status == RegistrationStatus.WARN
    assert failure_code == FailureCode.SECTION_CORRESPONDENCE_WEAK


def test_compute_same_image_metrics_nan_sanitation():
    import numpy as np
    from crossstainwsi.qc.metrics import compute_same_image_metrics

    # 纯白常数图像 (容易在 np.corrcoef 产生 NaN)
    white_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    metrics = compute_same_image_metrics(white_img, white_img)

    # 断言安全过滤为 -1.0 而不是 NaN
    assert not np.isnan(metrics.ncc_score)
    assert metrics.ncc_score == -1.0
    assert not np.isnan(metrics.edge_corr)
    assert metrics.edge_corr == -1.0
    assert metrics.details["tissue_fraction"] == 0.0
