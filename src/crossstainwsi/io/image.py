"""
图像与截图文件读取与保存适配器
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class ImageCropReader:
    """
    负责读取和预处理手工截取的 4x/20x TIFF/PNG 图像
    """
    @staticmethod
    def load_crop_bgr(path: Path, flip_horizontal: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        读取截图为 BGR 格式 numpy 数组
        如果 flip_horizontal 为 True，在进入算法前先执行水平翻转（修正历史截图标注镜像）
        返回: (img_bgr, (width, height))
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Crop image file not found: {path}")

        pil_img = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        if flip_horizontal:
            bgr = cv2.flip(bgr, 1)

        h, w = bgr.shape[:2]
        return bgr, (w, h)

    @staticmethod
    def save_publication_tiff(
        img_bgr: np.ndarray,
        out_path: Path,
        dpi: Tuple[int, int] = (300, 300),
        compression: str = "tiff_lzw",
    ) -> None:
        """
        以 300 DPI、LZW 无损压缩格式保存出版级 TIFF 图像
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(out_path, dpi=dpi, compression=compression)

    @staticmethod
    def save_overlay_png(
        ref_bgr: np.ndarray,
        moving_bgr: np.ndarray,
        out_path: Path,
        alpha: float = 0.5,
        dpi: Tuple[int, int] = (300, 300),
    ) -> None:
        """
        生成半透明配准重叠对比图 (Overlay) 并保存
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if ref_bgr.shape[:2] != moving_bgr.shape[:2]:
            moving_resized = cv2.resize(
                moving_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            moving_resized = moving_bgr

        overlay = cv2.addWeighted(ref_bgr, alpha, moving_resized, 1.0 - alpha, 0)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(out_path, dpi=dpi)
