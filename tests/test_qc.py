from crossstainwsi.domain import QCMetrics, RegistrationStatus
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
    status, _ = engine.evaluate_cross_stain(strong_metrics)
    assert status == RegistrationStatus.PASS

    # 2. 内点极低应 ABSTAIN
    weak_metrics = QCMetrics(
        inliers=8,
        matches=150,
        inlier_ratio=0.05,
        spatial_coverage=0.1,
        scale=1.0,
    )
    status, _ = engine.evaluate_cross_stain(weak_metrics)
    assert status == RegistrationStatus.ABSTAIN

    # 3. 异常缩放形变应 ABSTAIN
    deformed_metrics = QCMetrics(
        inliers=100,
        matches=150,
        inlier_ratio=0.66,
        spatial_coverage=0.6,
        scale=1.45,  # 严重异常尺度
    )
    status, _ = engine.evaluate_cross_stain(deformed_metrics)
    assert status == RegistrationStatus.ABSTAIN

    # 4. 临界质量应 WARN
    marginal_metrics = QCMetrics(
        inliers=25,
        matches=100,
        inlier_ratio=0.25,
        spatial_coverage=0.25,
        scale=1.01,
    )
    status, _ = engine.evaluate_cross_stain(marginal_metrics)
    assert status == RegistrationStatus.WARN
