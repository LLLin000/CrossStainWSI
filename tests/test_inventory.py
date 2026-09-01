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


def test_asset_discoverer_real_directory_regression(tmp_path):
    """
    真实临时目录发现回归测试：
    - 验证带连字符文件名 (4-4w-1-Masson.kfb) 正常扫描入库
    - 验证数字染色名 (CD68, Ki67) 正常解析
    - 验证 tiff 截图不会被错误识别为 WSI 切片
    """
    wsi_dir = tmp_path / "wsi"
    wsi_dir.mkdir()
    tiff_dir = tmp_path / "screenshots"
    tiff_dir.mkdir()

    # 创建虚拟切片文件
    (wsi_dir / "4-4w-1-Masson.kfb").write_bytes(b"dummy kfb masson")
    (wsi_dir / "4-4w-1-HE.kfb").write_bytes(b"dummy kfb he")
    (wsi_dir / "4-4w-1-CD68.kfb").write_bytes(b"dummy kfb cd68")

    # 创建虚拟截图文件
    (tiff_dir / "4-4w-1-Masson-4x.tif").write_bytes(b"dummy tiff 4x")
    (tiff_dir / "4-4w-1-CD68-20x.tif").write_bytes(b"dummy tiff 20x")

    discoverer = AssetDiscoverer(base_dir=wsi_dir, tiff_dir=tiff_dir)
    inventory = discoverer.discover()

    # 1. 验证样本成功发现
    sample = inventory.get_sample("4-4w-1")
    assert sample is not None

    # 2. 验证 3 个切片均被正确录入，截图未污染 slides
    assert len(sample.slides) == 3
    assert "Masson" in sample.slides
    assert "HE" in sample.slides
    assert "CD68" in sample.slides

    # 3. 验证两张证据截图被正确解析 (含染色名推导与倍率)
    ev = sample.roi_evidence
    assert len(ev.evidence_views) == 2
    assert ev.has_4x
    assert ev.has_20x
    assert ev.inferred_reference_stain == "Masson" or ev.inferred_reference_stain == "CD68"
