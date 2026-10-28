"""Early duplicate detection, before any OCR or LLM call.

Two checks, cheapest first:
- identical file bytes (sha256);
- perceptual difference hash (dHash) on a small grayscale thumbnail: catches the same photo
  re-encoded, stripped of EXIF or resized. On the sample data duplicates differ by 0-1 bits out
  of 256 and different menus by 100+, so a small threshold is safe.

In each group the largest image (most pixels) is kept as the original: it gives the best OCR.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from menu_poc.io.dataset import ImageRef

HASH_SIZE = 16  # 16x16 = 256-bit hash


@dataclass(frozen=True)
class Duplicate:
    image_id: str
    original_id: str
    reason: str  # "identical file" | "near-duplicate (distance N/256)"


def dhash(image: Image.Image, size: int = HASH_SIZE) -> np.ndarray:
    """Difference hash: sign of the horizontal gradient on a (size+1) x size thumbnail."""
    image.draft("L", (size * 8, size * 8))  # JPEG: decode at reduced scale, much faster
    gray = ImageOps.exif_transpose(image).convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def find_duplicates(refs: list[ImageRef], max_distance: int = 10) -> dict[str, Duplicate]:
    """Map duplicate image id -> Duplicate (pointing at the kept original)."""
    infos = []
    for ref in refs:
        image = Image.open(io.BytesIO(ref.data))  # lazy: size is known before decoding
        infos.append((ref, image.size[0] * image.size[1], dhash(image)))
    # Largest first, so the first member of each group is the one we keep.
    infos.sort(key=lambda info: (-info[1], info[0].image_id))

    duplicates: dict[str, Duplicate] = {}
    originals: list[tuple[ImageRef, np.ndarray]] = []
    for ref, _, hash_ in infos:
        for orig, orig_hash in originals:
            if ref.sha256 == orig.sha256:
                reason = "identical file"
            else:
                distance = int((hash_ != orig_hash).sum())
                if distance > max_distance:
                    continue
                reason = f"near-duplicate (distance {distance}/{hash_.size})"
            duplicates[ref.image_id] = Duplicate(ref.image_id, orig.image_id, reason)
            break
        else:
            originals.append((ref, hash_))
    return duplicates
