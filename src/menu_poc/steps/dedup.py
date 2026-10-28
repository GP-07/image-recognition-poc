"""Duplicate photos are answered with the original's result, before any OCR or LLM call.

Two checks, cheapest first:
- identical file bytes (sha256);
- perceptual difference hash (dHash) on a small grayscale thumbnail: catches the same photo
  re-encoded, stripped of EXIF or resized. On the sample data duplicates differ by 0-4 bits out
  of 256 and different menus by 100+, so a small threshold is safe.

The first photo of a group is the original; the pipeline feeds images largest first, so the
original is the one that gives the best OCR.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from menu_poc.constants import DEFAULT_DEDUP_MAX_DISTANCE, DHASH_SIZE
from menu_poc.data_classes import ImageContext, ImageRef
from menu_poc.models import ImageResult
from menu_poc.steps.base import Step


def dhash(image: Image.Image, size: int = DHASH_SIZE) -> np.ndarray:
    """Difference hash: sign of the horizontal gradient on a (size+1) x size thumbnail."""
    image.draft("L", (size * 8, size * 8))  # JPEG: decode at reduced scale, much faster
    gray = ImageOps.exif_transpose(image).convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


class DedupStep(Step):
    name = "dedup"

    def __init__(self, max_distance: int = DEFAULT_DEDUP_MAX_DISTANCE) -> None:
        super().__init__()
        self.max_distance = max_distance
        # Originals seen so far; their results are filled in place by the following steps.
        self._originals: list[tuple[ImageRef, np.ndarray, ImageResult]] = []

    def process(self, ctx: ImageContext) -> bool:
        image_hash = dhash(Image.open(io.BytesIO(ctx.ref.data)))  # lazy open: no full decode
        for ref, orig_hash, orig_result in self._originals:
            if ctx.ref.sha256 == ref.sha256:
                reason = "identical file"
            else:
                distance = int((image_hash != orig_hash).sum())
                if distance > self.max_distance:
                    continue
                reason = f"near-duplicate (distance {distance}/{image_hash.size})"
            result = ctx.result
            result.dishes = orig_result.dishes
            result.rules_confidence = orig_result.rules_confidence
            result.duplicate_of, result.duplicate_reason = ref.image_id, reason
            return False  # nothing else to do for this photo
        self._originals.append((ctx.ref, image_hash, ctx.result))
        return True
