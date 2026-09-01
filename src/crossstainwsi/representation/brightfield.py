"""
通用明场组织学模态适配器 (GenericBrightfieldAdapter)
为 Masson、HE、Gram、番红固绿、天狼星红等亮场染色切片生成结构表征集
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.representation.builder import RepresentationBuilder
from crossstainwsi.representation.contracts import CanonicalRepresentationSet


class GenericBrightfieldAdapter:
    """
    通用明场组织学切片适配器
    """
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def adapt(
        self,
        img_bgr: np.ndarray,
        mpp_xy: Optional[Tuple[float, float]] = None,
        source_level: int = 4,
    ) -> CanonicalRepresentationSet:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe_enhanced = self.clahe.apply(gray)

        # 1. 空间有效掩模与组织前景掩模
        valid_mask = np.ones(gray.shape, dtype=bool)
        tissue_mask = RepresentationBuilder.compute_tissue_mask(img_bgr)

        # 2. 几何轮廓与距离场
        coarse_contour = RepresentationBuilder.compute_coarse_contour(tissue_mask)
        distance_field = RepresentationBuilder.compute_distance_field(tissue_mask)

        # 3. 多尺度梯度金字塔
        gradient_pyramid = RepresentationBuilder.compute_gradient_pyramid(clahe_enhanced)

        # 4. 信息量评估
        info = RepresentationBuilder.compute_informativeness(clahe_enhanced, tissue_mask)

        # 5. 细胞核概率场粗估 (基于形态学顶帽变换 Top-Hat 提取深色颗粒核结构)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        nuclear_density = (blackhat.astype(np.float32) / 255.0) * (tissue_mask > 0).astype(np.float32)

        return CanonicalRepresentationSet(
            valid_mask=valid_mask,
            tissue_mask=tissue_mask,
            artifact_mask=None,
            coarse_contour=coarse_contour,
            distance_field=distance_field,
            gradient_pyramid=gradient_pyramid,
            nuclear_density=nuclear_density,
            feature_image=clahe_enhanced,
            mpp_xy=mpp_xy,
            source_level=source_level,
            modality="brightfield",
            representation_provenance={"method": "CLAHE_Morphology_Brightfield"},
            informativeness=info,
        )
