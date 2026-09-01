"""
免疫组化 (IHC) 模态适配器与色彩解卷积 (IHCDeconvolutionAdapter)
基于 Ruifrok-Johnston 色彩解卷积分离 DAB 阳性沉淀与苏木精细胞核结构骨架
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

try:
    from skimage.color import separate_stains, hed_from_rgb, hdx_from_rgb
except ImportError:
    separate_stains = None

from crossstainwsi.representation.builder import RepresentationBuilder
from crossstainwsi.representation.brightfield import GenericBrightfieldAdapter
from crossstainwsi.representation.contracts import CanonicalRepresentationSet


class IHCDeconvolutionAdapter:
    """
    负责解析 IHC 染色图像，分离 DAB 干扰并提取同源苏木精细胞核结构场
    """
    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.fallback_adapter = GenericBrightfieldAdapter()

    def adapt(
        self,
        img_bgr: np.ndarray,
        mpp_xy: Optional[Tuple[float, float]] = None,
        source_level: int = 4,
        stain_matrix: Optional[np.ndarray] = None,
    ) -> CanonicalRepresentationSet:
        # 如果 skimage 不可用，优雅回退到通用明场适配器
        if separate_stains is None:
            rep = self.fallback_adapter.adapt(img_bgr, mpp_xy=mpp_xy, source_level=source_level)
            rep.modality = "ihc_dab_fallback"
            return rep

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        conv_matrix = stain_matrix if stain_matrix is not None else hdx_from_rgb

        try:
            # 1. 执行 Ruifrok-Johnston 色彩解卷积 (Beer-Lambert 光学密度分离)
            stain_channels = separate_stains(img_rgb, conv_matrix)
            # Channel 0: Hematoxylin (苏木精), Channel 1: DAB (二氨基联苯胺), Channel 2: Residual (残差)
            h_raw = stain_channels[:, :, 0]
            dab_raw = stain_channels[:, :, 1]

            # 2. 归一化苏木精光学密度通道 (去除负值与极端光密度)
            h_norm = np.clip(h_raw, 0.0, np.percentile(h_raw[h_raw > 0], 99.0) if (h_raw > 0).any() else 1.0)
            h_max = h_norm.max()
            h_density = (h_norm / (h_max if h_max > 1e-4 else 1.0)).astype(np.float32)

            # 3. 提取组织前景与空间掩模
            tissue_mask = RepresentationBuilder.compute_tissue_mask(img_bgr)
            valid_mask = np.ones(img_bgr.shape[:2], dtype=bool)

            # 4. 生成供匹配器直接使用的特征图像 (对苏木精通道执行 CLAHE 增强)
            h_uint8 = np.clip(h_density * 255.0, 0, 255).astype(np.uint8)
            feature_img = self.clahe.apply(h_uint8)

            # 5. 几何轮廓与多尺度梯度
            coarse_contour = RepresentationBuilder.compute_coarse_contour(tissue_mask)
            distance_field = RepresentationBuilder.compute_distance_field(tissue_mask)
            gradient_pyramid = RepresentationBuilder.compute_gradient_pyramid(feature_img)

            # 6. 信息量评估
            info = RepresentationBuilder.compute_informativeness(feature_img, tissue_mask)
            # 记录 DAB 表达阳性区域占比
            dab_positive_fraction = float((dab_raw > 0.1).mean())
            info["dab_positive_fraction"] = dab_positive_fraction

            return CanonicalRepresentationSet(
                valid_mask=valid_mask,
                tissue_mask=tissue_mask,
                artifact_mask=None,
                coarse_contour=coarse_contour,
                distance_field=distance_field,
                gradient_pyramid=gradient_pyramid,
                nuclear_density=h_density * (tissue_mask > 0).astype(np.float32),
                feature_image=feature_img,
                mpp_xy=mpp_xy,
                source_level=source_level,
                modality="ihc_dab",
                representation_provenance={
                    "method": "Ruifrok_Color_Deconvolution",
                    "stain_matrix": "hdx_from_rgb",
                    "target_channel": "Hematoxylin",
                },
                informativeness=info,
            )

        except Exception as e:
            # 解卷积异常时安全回退
            rep = self.fallback_adapter.adapt(img_bgr, mpp_xy=mpp_xy, source_level=source_level)
            rep.modality = "ihc_dab_fallback"
            rep.representation_provenance["fallback_reason"] = str(e)
            return rep
