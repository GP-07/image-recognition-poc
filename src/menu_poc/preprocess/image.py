"""Image normalisation before OCR: EXIF orientation, RGB, bounded size.

Page rotation (photos taken sideways) is handled by the OCR engine, which can judge it from the
text boxes it detects; see ``menu_poc.ocr.rapid``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps

from menu_poc.constants import DEFAULT_MAX_IMAGE_SIDE, PREPROCESS_VERSION
from menu_poc.interfaces import Preprocessor


class ImagePreprocessor(Preprocessor):
    """EXIF transpose, convert to RGB and downscale so the longest side is <= ``max_side``."""

    def __init__(self, max_side: int = DEFAULT_MAX_IMAGE_SIDE) -> None:
        if max_side <= 0:
            raise ValueError("max_side must be positive")
        self.max_side = max_side

    def fingerprint(self) -> dict[str, Any]:
        return {
            "name": "image_preprocessor",
            "version": PREPROCESS_VERSION,
            "max_side": self.max_side,
        }

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image) or image
        if image.mode != "RGB":
            image = image.convert("RGB")
        longest = max(image.size)
        if longest > self.max_side:
            scale = self.max_side / longest
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image
