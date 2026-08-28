from pathlib import Path
from crossstainwsi.inventory.assets import AssetInventory, ROIEvidence, SampleAssets, SlideAsset
from crossstainwsi.inventory.discover import AssetDiscoverer
from crossstainwsi.review.states import ArtifactTier, RunVerdict, resolve_artifact_dir


def test_asset_inventory_summary():
    s1 = SampleAssets(
        sample_id="4w-5-3",
        slides={"masson": SlideAsset(stain="masson", path=Path("dummy"), format="kfb")},
        roi_evidence=ROIEvidence(crop_4x_path=Path("dummy_4x.tif")),
    )
    s2 = SampleAssets(
        sample_id="2-2w-4",
        slides={"masson": SlideAsset(stain="masson", path=Path("dummy"), format="kfb")},
        roi_evidence=ROIEvidence(),  # 无截图证据
    )
    inventory = AssetInventory(samples={"4w-5-3": s1, "2-2w-4": s2})
    summ = inventory.summary()
    assert summ["total_samples"] == 2
    assert "4w-5-3" in summ["samples_with_crops"]
    assert "2-2w-4" in summ["samples_wsi_only"]


def test_resolve_artifact_dir(tmp_path):
    out_pass = resolve_artifact_dir(tmp_path, RunVerdict.PASS)
    assert out_pass.name == ArtifactTier.FINAL.value
    assert out_pass.exists()

    out_review = resolve_artifact_dir(tmp_path, RunVerdict.REVIEW)
    assert out_review.name == ArtifactTier.REVIEW.value
    assert out_review.exists()

    out_abstain = resolve_artifact_dir(tmp_path, RunVerdict.ABSTAIN)
    assert out_abstain.name == ArtifactTier.DEBUG.value
    assert out_abstain.exists()

    out_incomplete = resolve_artifact_dir(tmp_path, RunVerdict.INCOMPLETE)
    assert out_incomplete.name == ArtifactTier.DEBUG.value
