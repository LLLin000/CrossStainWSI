"""
免疫荧光 (mIF/CyCIF) 与多通道模态适配器 (FluorescenceAdapter & ChannelEvidenceSelector)
实现动态信息通道优选、DAPI 细胞核 LoG 响应场提取与跨模态同源核结构场构建
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from crossstainwsi.representation.builder import RepresentationBuilder
from crossstainwsi.representation.contracts import CanonicalRepresentationSet


class ChannelEvidenceSelector:
    """
    负责在多通道荧光切片中智能评估并优选最具结构信息量的通道 (解决 DAPI 退色或过曝问题)
    """
    @staticmethod
    def select_best_channels(
        channels: np.ndarray, # 形状为 (H, W, C) 或 (C, H, W)
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        if channels.ndim == 2:
            return 0, {"scores": [1.0], "selected_name": "Channel_0"}

        # 统一转为 (H, W, C)
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

            # 高斯平滑抑制孤立单像素噪声，保留真实细胞核结构边界
            blurred = cv2.GaussianBlur(ch.astype(np.float32), (5, 5), 1.0)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
            grad_energy = float(np.mean(gx * gx + gy * gy))

            # DAPI 命名通道享有加权偏好
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

    def _extract_log_nuclear_field(self, dapi_gray: np.ndarray) -> np.ndarray:
        """
        使用 Laplacian of Gaussian (LoG) 滤波器提取斑状细胞核中心响应场
        """
        blurred = cv2.GaussianBlur(dapi_gray.astype(np.float32), (self.log_kernel_size, self.log_kernel_size), self.log_sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        # 细胞核中心在 LoG 中为负极值，取反并截断负值
        log_response = np.clip(-lap, 0.0, None)
        max_val = log_response.max()
        if max_val > 1e-4:
            log_response /= max_val
        return log_response

    def adapt(
        self,
        img_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        mpp_xy: Optional[Tuple[float, float]] = None,
        source_level: int = 4,
    ) -> CanonicalRepresentationSet:
        # 1. 通道智能优选
        if img_data.ndim == 3:
            best_idx, ch_info = ChannelEvidenceSelector.select_best_channels(img_data, channel_names)
            if img_data.shape[0] < img_data.shape[2]:
                primary_channel = img_data[best_idx, :, :]
            else:
                primary_channel = img_data[:, :, best_idx]
        else:
            primary_channel = img_data
            ch_info = {"best_name": "Single_Channel"}

        # 归一化到 0 ~ 255
        p_min, p_max = primary_channel.min(), primary_channel.max()
        if p_max - p_min > 1e-4:
            ch_uint8 = ((primary_channel - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
        else:
            ch_uint8 = np.zeros_like(primary_channel, dtype=np.uint8)

        # 2. 提取 LoG 细胞核响应场
        nuclear_field = self._extract_log_nuclear_field(ch_uint8)

        # 3. 生成有效信号掩模 (Signal Mask)
        thresh_val, signal_mask = cv2.threshold(ch_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_CLOSE, kernel)

        # 4. 生成伪明场增强图 (反相合成：255 - I，将黑底光斑转为白底细胞核)
        pseudo_brightfield = 255 - ch_uint8
        feature_img = self.clahe.apply(pseudo_brightfield)

        # 5. 几何轮廓与距离场
        coarse_contour = RepresentationBuilder.compute_coarse_contour(signal_mask)
        distance_field = RepresentationBuilder.compute_distance_field(signal_mask)
        gradient_pyramid = RepresentationBuilder.compute_gradient_pyramid(feature_img)

        # 6. 信息量评估
        info = RepresentationBuilder.compute_informativeness(feature_img, signal_mask)
        info["selected_channel"] = ch_info.get("best_name", "Unknown")

        return CanonicalRepresentationSet(
            valid_mask=(ch_uint8 > 5),
            tissue_mask=signal_mask,
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
                "selected_channel": ch_info.get("best_name"),
                "inverted_brightfield": True,
            },
            informativeness=info,
        )
