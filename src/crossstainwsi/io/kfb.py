"""
KFB 格式切片适配器 (基于 kfbslide)
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

try:
    import kfbslide
except ImportError:
    kfbslide = None

from crossstainwsi.domain import PyramidLevel, SlideSpec
from crossstainwsi.io.base import SlideReader


class KFBReader(SlideReader):
    """
    KFB 格式读取器，使用 kfbslide 驱动并封装为 SlideReader
    """
    def __init__(self, path: Path, default_mpp: float = 0.44243):
        super().__init__(path, default_mpp)
        if kfbslide is None:
            raise RuntimeError("kfbslide library is required to read KFB files. Please install kfbslide.")
        if not self.path.exists():
            raise FileNotFoundError(f"KFB file not found: {self.path}")
        self._slide = kfbslide.OpenSlide(str(self.path))
        self._spec: Optional[SlideSpec] = None

    def read_metadata(self) -> SlideSpec:
        if self._spec is not None:
            return self._spec

        l0_dims = self._slide.dimensions
        levels = []
        for i, dims in enumerate(self._slide.level_dimensions):
            ds = float(self._slide.level_downsamples[i])
            levels.append(PyramidLevel(level=i, dimensions=dims, downsample=ds))

        mpp_raw_x = self._slide.properties.get("openslide.mpp-x")
        mpp_raw_y = self._slide.properties.get("openslide.mpp-y")
        if mpp_raw_x is not None and mpp_raw_y is not None:
            mpp_x, mpp_y = float(mpp_raw_x), float(mpp_raw_y)
            mpp_source = "metadata"
        else:
            mpp_x, mpp_y = self.default_mpp, self.default_mpp
            mpp_source = "configured_override"

        # 推断染色与样本ID
        name = self.path.stem
        parts = name.split("-")
        sample_id = "-".join(parts[:-1]) if len(parts) > 1 else name
        stain = parts[-1] if len(parts) > 1 else "Unknown"

        self._spec = SlideSpec(
            id=name,
            sample_id=sample_id,
            stain=stain,
            path=self.path,
            format="kfb",
            dimensions=l0_dims,
            levels=levels,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            mpp_source=mpp_source,
            properties=dict(self._slide.properties),
        )
        return self._spec

    def read_level_image(self, level: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        dims = self._slide.level_dimensions[level]
        ds = float(self._slide.level_downsamples[level])
        im_rgb = self._slide.read_region((0, 0), level, dims).convert("RGB")
        im_bgr = cv2.cvtColor(np.asarray(im_rgb), cv2.COLOR_RGB2BGR)
        return im_bgr, ds, dims

    def read_region(
        self,
        location_l0: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> np.ndarray:
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid region size: {size}")
        im_rgb = self._slide.read_region(location_l0, level, (w, h)).convert("RGB")
        im_bgr = cv2.cvtColor(np.asarray(im_rgb), cv2.COLOR_RGB2BGR)
        return im_bgr

    def close(self) -> None:
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass
            self._slide = None
