"""
结构表征与模态适配器单元测试 (tests/test_representations.py)
"""

import numpy as np
import pytest

from crossstainwsi.representation.brightfield import GenericBrightfieldAdapter
from crossstainwsi.representation.ihc import IHCDeconvolutionAdapter
from crossstainwsi.representation.fluorescence import (
    ChannelEvidenceSelector,
    FluorescenceAdapter,
)


def test_generic_brightfield_adapter():
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    # 绘制紫色组织区域
    img[20:100, 20:100] = (180, 50, 150)

    adapter = GenericBrightfieldAdapter()
    rep = adapter.adapt(img, mpp_xy=(0.44, 0.44))

    assert rep.modality == "brightfield"
    assert rep.tissue_mask is not None
    assert rep.tissue_mask.shape == (120, 120)
    assert rep.distance_field is not None
    assert len(rep.gradient_pyramid) == 3
    assert rep.feature_image is not None
    assert rep.informativeness["tissue_fraction"] > 0.3


def test_ihc_deconvolution_adapter():
    # 合成一张包含棕色 (DAB) 与蓝色 (苏木精) 的切片图像
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # 模拟 DAB 棕色抗原区域 (BGR: 蓝色低, 红色高)
    img[20:50, 20:50] = (30, 80, 160)
    # 模拟苏木精蓝色细胞核区域 (BGR: 红色低, 蓝色高)
    img[55:85, 55:85] = (160, 80, 30)

    adapter = IHCDeconvolutionAdapter()
    rep = adapter.adapt(img)

    assert "ihc_dab" in rep.modality
    assert rep.nuclear_density is not None
    assert rep.feature_image is not None
    assert rep.feature_image.shape == (100, 100)


def test_fluorescence_adapter_and_channel_selector():
    # 合成 3 通道荧光切片 (H, W, 3): Channel 0: 弱噪声, Channel 1: DAPI 强信号, Channel 2: 极弱
    ch0 = np.random.randint(0, 10, (120, 120), dtype=np.uint8)
    ch1 = np.zeros((120, 120), dtype=np.uint8)
    # 在 DAPI 通道绘制圆形细胞核光斑
    for y in [30, 60, 90]:
        for x in [30, 60, 90]:
            ch1[y-5:y+5, x-5:x+5] = 220

    ch2 = np.zeros((120, 120), dtype=np.uint8)
    multi_ch = np.stack([ch0, ch1, ch2], axis=-1)

    # 1. 验证通道优选自动选中具有高对比度的 DAPI 通道 (Channel 1)
    best_idx, info = ChannelEvidenceSelector.select_best_channels(multi_ch, ["CD20", "DAPI", "CD68"])
    assert best_idx == 1
    assert "DAPI" in info["best_name"]

    # 2. 验证荧光适配器处理
    adapter = FluorescenceAdapter()
    rep = adapter.adapt(multi_ch, ["CD20", "DAPI", "CD68"])

    assert rep.modality == "fluorescence"
    assert rep.nuclear_density is not None
    assert rep.feature_image is not None
    assert rep.representation_provenance["inverted_brightfield"] is True


def test_fluorescence_adapter_without_dapi_safe_fallback():
    """
    测试当荧光切片不包含 DAPI/Hoechst 且无核斑点时，
    NuclearChannelResolver 不盲目硬猜，nuclear_density 安全设为 None
    """
    from crossstainwsi.representation.fluorescence import NuclearChannelResolver
    # 仅含弥散抗原通道 (无 DAPI 斑点)
    ch0 = np.full((100, 100), 50, dtype=np.uint8)
    ch1 = np.full((100, 100), 80, dtype=np.uint8)
    multi_ch = np.stack([ch0, ch1], axis=-1)

    nuc_idx, data, name = NuclearChannelResolver.resolve_nuclear_channel(multi_ch, ["CD20", "CD68"])
    assert nuc_idx is None
    assert name == "none"

    adapter = FluorescenceAdapter()
    rep = adapter.adapt(multi_ch, ["CD20", "CD68"])
    assert rep.nuclear_density is None
    assert rep.feature_image is not None
