"""
出版级多模态多染色接触表 (Contact Sheet) 与拼图生成器
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ContactSheetGenerator:
    """
    负责生成 4x / 20x 跨染色并排对比拼图，带标注、300 DPI 分辨率与干净排版
    """
    @staticmethod
    def create_contact_sheet(
        images: Dict[str, np.ndarray],
        title: str,
        out_path: Path,
        dpi: Tuple[int, int] = (300, 300),
        border_px: int = 16,
        header_height: int = 80,
    ) -> None:
        """
        images: OrderedDict 或 Dict[标签名称, BGR 图像]
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        stain_names = list(images.keys())
        n_imgs = len(stain_names)
        if n_imgs == 0:
            return

        # 获取统一标准尺寸
        first_img = images[stain_names[0]]
        h_single, w_single = first_img.shape[:2]

        total_width = n_imgs * w_single + (n_imgs + 1) * border_px
        total_height = h_single + border_px * 2 + header_height

        canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # 绘制主标题
        draw.text((border_px, 20), title, fill=(20, 20, 20))

        # 逐个贴入图像并绘制标签
        for i, name in enumerate(stain_names):
            img_bgr = images[name]
            if img_bgr.shape[:2] != (h_single, w_single):
                img_bgr = cv2.resize(img_bgr, (w_single, h_single), interpolation=cv2.INTER_LINEAR)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_slice = Image.fromarray(img_rgb)

            x_pos = border_px + i * (w_single + border_px)
            y_pos = border_px + header_height

            canvas.paste(pil_slice, (x_pos, y_pos))

            # 绘制染色标签
            label_text = f"[{name}]"
            draw.text((x_pos + 8, y_pos - 24), label_text, fill=(40, 40, 40))

        canvas.save(out_path, dpi=dpi)
