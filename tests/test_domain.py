from pathlib import Path
from crossstainwsi.domain import (
    CoordinateSpace,
    PyramidLevel,
    QCMetrics,
    RegistrationResult,
    RegistrationStatus,
    ROI,
    SlideSpec,
    TransformType,
)


def test_slidespec_levels():
    levels = [
        PyramidLevel(level=0, dimensions=(40000, 30000), downsample=1.0),
        PyramidLevel(level=1, dimensions=(20000, 15000), downsample=2.0),
        PyramidLevel(level=2, dimensions=(10000, 7500), downsample=4.0),
        PyramidLevel(level=4, dimensions=(2500, 1875), downsample=16.0),
    ]
    spec = SlideSpec(
        id="sample1-HE",
        sample_id="sample1",
        stain="HE",
        path=Path("sample1-HE.kfb"),
        format="kfb",
        dimensions=(40000, 30000),
        levels=levels,
        mpp_x=0.44243,
        mpp_y=0.44243,
    )
    assert spec.get_level_downsample(2) == 4.0
    assert spec.get_level_downsample(4) == 16.0
    assert spec.get_level_dimensions(2) == (10000, 7500)
    assert spec.get_level_dimensions(4) == (2500, 1875)


def test_registration_result():
    metrics = QCMetrics(
        inliers=120,
        matches=200,
        inlier_ratio=0.6,
        spatial_coverage=0.75,
        median_reproj_error=1.2,
        scale=1.002,
        rotation_deg=0.5,
        method="LoFTR",
    )
    res = RegistrationResult(
        sample_id="test-1",
        moving_stain="HE",
        reference_stain="masson",
        status=RegistrationStatus.PASS,
        reason="High confidence alignment",
        metrics=metrics,
    )
    assert res.status == RegistrationStatus.PASS
    assert res.metrics.inliers == 120
