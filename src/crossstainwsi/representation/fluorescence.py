"""
免疫荧光 (mIF/CyCIF) 与多通道模态适配器 (FluorescenceAdapter, NuclearChannelResolver, ChannelEvidenceSelector)
严格解耦细胞核定位通道 (Nuclear Channel) 与宏观组织辅助特征通道 (Informative Feature Channel)
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.representation.builder import RepresentationBuilder
from crossstainwsi.representation.contracts import CanonicalRepresentationSet


class NuclearChannelResolver:
    """
    负责从多通道切片中精确识别并提取 DAPI / Hoechst 细胞核染色通道
    """
    @staticmethod
    def resolve_nuclear_channel(
        channels_hwc: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], Optional[np.ndarray], str]:
        if channels_hwc.ndim == 2:
            return 0, channels_hwc, "single_channel"

        n_channels = channels_hwc.shape[2]
        dapi_idx = None

        # 1. 优先按命名关键词精确匹配
        if channel_names:
            for i, name in enumerate(channel_names):
                n_lower = name.lower()
                if "dapi" in n_lower or "hoechst" in n_lower or "nuclear" in n_lower or "dna" in n_lower:
                    dapi_idx = i
                    break

        # 2. 若无通道命名，根据斑点状核形态特征启发式评分
        if dapi_idx is None:
            spot_scores = []
            for i in range(n_channels):
                ch = channels_hwc[:, :, i]
                dyn = float(ch.max() - ch.min())
                if dyn < 1e-4:
                    spot_scores.append(-1.0)
                    continue
                # 顶帽变换提取细小圆点状细胞核
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                tophat = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
                score = float(tophat.mean() * dyn)
                spot_scores.append(score)
            dapi_idx = int(np.argmax(spot_scores)) if spot_scores else 0

        ch_data = channels_hwc[:, :, dapi_idx]
        ch_name = channel_names[dapi_idx] if channel_names and dapi_idx < len(channel_names) else f"Channel_{dapi_idx}"
        return dapi_idx, ch_data, ch_name


class ChannelEvidenceSelector:
    """
    负责在多通道荧光切片中智能评估并优选最具辅助结构信息量的通道 (如 Autofluorescence 或高对比度抗体)
    """
    @staticmethod
    def select_best_channels(
        channels: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        if channels.ndim == 2:
            return 0, {"scores": [1.0], "selected_name": "Channel_0"}

        if channels.shape[0] < channels.shape[2]:
            channels_hwc = np.transpose(channels, (1, 2, 0))
        else:
            channels_hwc = channels

        n_channels = channels_hwc.shape[2]
        scores = []

        for i in range(n_channels):
            ch = channels_hwc[:, :, i]
            ch_max = float(ch.max())
            ch_min = float(ch.min())
            dyn_range = ch_max - ch_min

            if dyn_range < 1e-4:
                scores.append(-1.0)
                continue

            blurred = cv2.GaussianBlur(ch.astype(np.float32), (5, 5), 1.0)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
            grad_energy = float(np.mean(gx * gx + gy * gy))

            name_bonus = 2.0 if channel_names and i < len(channel_names) and "dapi" in channel_names[i].lower() else 1.0
            score = grad_energy * dyn_range * name_bonus
            scores.append(score)

        best_idx = int(np.argmax(scores)) if scores else 0
        best_name = channel_names[best_idx] if channel_names and best_idx < len(channel_names) else f"Channel_{best_idx}"

        return best_idx, {
            "best_index": best_idx,
            "best_name": best_name,
            "all_scores": scores,
        }


class FluorescenceAdapter:
    """
    负责将多通道/单通道暗场免疫荧光图像转换为规范结构表征集
    """
    def __init__(
        self,
        log_kernel_size: int = 9,
        log_sigma: float = 1.8,
        clip_limit: float = 3.0,
    ):
        self.log_kernel_size = log_kernel_size
        self.log_sigma = log_sigma
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def _extract_log_nuclear_field(self, dapi_uint8: np.ndarray) -> np.ndarray:
        """
        使用 Laplacian of Gaussian (LoG) 滤波器提取斑状细胞核中心响应场
        """
        blurred = cv2.GaussianBlur(dapi_uint8.astype(np.float32), (self.log_kernel_size, self.log_kernel_size), self.log_sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        log_response = np.clip(-lap, 0.0, None)
        max_val = log_response.max()
        if max_val > 1e-4:
            log_response /= max_val
        return log_response.astype(np.float32)

    def adapt(
        self,
        img_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        mpp_xy: Optional[Tuple[float, float]] = None,
        source_level: int = 4,
    ) -> CanonicalRepresentationSet:
        # 统一转为 (H, W, C)
        if img_data.ndim == 3:
            if img_data.shape[0] < img_data.shape[2]:
                channels_hwc = np.transpose(img_data, (1, 2, 0))
            else:
                channels_hwc = img_data
        else:
            channels_hwc = img_data[:, :, None]

        # 1. 细胞核通道精确提取 (NuclearChannelResolver)
        nuc_idx, dapi_raw, nuc_name = NuclearChannelResolver.resolve_nuclear_channel(channels_hwc, channel_names)
        p_min, p_max = float(dapi_raw.min()), float(dapi_raw.max())
        if p_max - p_min > 1e-4:
            dapi_uint8 = ((dapi_raw - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
        else:
            dapi_uint8 = np.zeros(dapi_raw.shape[:2], dtype=np.uint8)

        nuclear_field = self._extract_log_nuclear_field(dapi_uint8)

        # 2. 辅助特征通道优选 (ChannelEvidenceSelector)
        feat_idx, ch_info = ChannelEvidenceSelector.select_best_channels(channels_hwc, channel_names)
        feat_raw = channels_hwc[:, :, feat_idx]
        f_min, f_max = float(feat_raw.min()), float(feat_raw.max())
        if f_max - f_min > 1e-4:
            feat_uint8 = ((feat_raw - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
        else:
            feat_uint8 = dapi_uint8

        # 3. 宏观组织支撑掩模 (Tissue Support Mask) 与 细胞核信号掩模 (Signal Mask)
        # 信号掩模 (局部光斑)
        _, signal_mask = cv2.threshold(dapi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 宏观组织支撑掩模: 通过大尺度高斯模糊 + 闭运算生成平滑的宏观组织包络 (Tissue Envelope)
        env_blur = cv2.GaussianBlur(dapi_uint8.astype(np.float32), (31, 31), 10.0)
        tissue_support_mask = (env_blur > 10.0).astype(np.uint8) * 255
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tissue_support_mask = cv2.morphologyEx(tissue_support_mask, cv2.MORPH_CLOSE, kernel_large)

        # 4. 生成伪明场增强图 (反相合成：255 - I_feat)
        pseudo_brightfield = 255 - feat_uint8
        feature_img = self.clahe.apply(pseudo_brightfield)

        # 5. 宏观组织外轮廓与距离场 (严格由组织宏观包络生成，绝非数万个孤立核斑点)
        coarse_contour = RepresentationBuilder.compute_coarse_contour(tissue_support_mask)
        distance_field = RepresentationBuilder.compute_distance_field(tissue_support_mask)
        gradient_pyramid = RepresentationBuilder.compute_gradient_pyramid(feature_img)

        # 6. 信息量评估
        info = RepresentationBuilder.compute_informativeness(feature_img, tissue_support_mask)
        info["nuclear_channel"] = nuc_name
        info["feature_channel"] = ch_info.get("best_name", "Unknown")

        return CanonicalRepresentationSet(
            valid_mask=(dapi_uint8 > 3) | (feat_uint8 > 3),
            tissue_mask=tissue_support_mask,
            artifact_mask=None,
            coarse_contour=coarse_contour,
            distance_field=distance_field,
            gradient_pyramid=gradient_pyramid,
            nuclear_density=nuclear_field,
            feature_image=feature_img,
            mpp_xy=mpp_xy,
            source_level=source_level,
            modality="fluorescence",
            representation_provenance={
                "method": "DAPI_LoG_Nuclear_Response",
                "nuclear_channel": nuc_name,
                "feature_channel": ch_info.get("best_name"),
                "inverted_brightfield": True,
            },
            informativeness=info,
        )
