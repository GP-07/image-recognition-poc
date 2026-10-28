"""RapidOCR adapter (PP-OCR models on ONNX Runtime, CPU).

Page orientation, decided from the text boxes of a first pass and only paid for when needed:

1. Sideways photos: when most text boxes are taller than wide, the page is re-read rotated by
   90° and 270° and the orientation that reads the most confident text (sum of
   ``len(text) * score``) wins. The retries run without the per-line 180° classifier, so an
   upside-down page reads as garbage instead of being silently fixed line by line. The upright
   pass is kept only if both rotations read clearly less text.
2. Tilted photos: when the median angle of the long text lines is >= ``DESKEW_MIN_DEG``, the page
   is rotated by that angle (rounded to whole degrees) and re-read. Axis-aligned boxes of tilted
   lines overlap their neighbours, which breaks row matching (a price drifts onto the next dish).

``OcrResult.rotation`` is the total counter-clockwise rotation applied, in whole degrees, and the
boxes are in the coordinates of that rotated image.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml
from PIL import Image

from menu_poc.models import OcrLine, OcrResult

ORIENTATION_VERSION = "3"
TALL_RATIO = 1.5  # a box is "tall" when height > TALL_RATIO * width
TALL_SHARE = 0.5  # the page looks sideways when more than this share of text boxes is tall
MIN_CHARS = 3  # ignore very short boxes when judging orientation (digits, icons)
KEEP_UPRIGHT = 0.8  # keep rotation 0 if the best sideways pass scores below this share of it
DESKEW_MIN_DEG = 1.5  # re-read tilted pages from this median line angle
DESKEW_MAX_DEG = 15.0  # larger angles are not a tilt but mixed orientations: leave them
LONG_LINE = 4.0  # lines at least this many times wider than tall vote for the tilt angle

# PIL transposes rotate counter-clockwise; the key is the rotation applied to the image.
_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 270: Image.Transpose.ROTATE_270}


def _rapidocr_info() -> dict[str, Any]:
    """Version and model identifiers of the installed rapidocr, read without loading models."""
    import rapidocr

    cfg_path = Path(rapidocr.__file__).parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    models = {
        task: {k: cfg[task][k] for k in ("engine_type", "lang_type", "model_type", "ocr_version")}
        for task in ("Det", "Cls", "Rec")
    }
    return {"rapidocr_version": version("rapidocr"), "models": models}


@dataclass
class _Item:
    text: str
    score: float
    bbox: tuple[float, float, float, float]
    angle: float  # degrees of the top edge; positive = descending to the right

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def _looks_sideways(items: list[_Item]) -> bool:
    """True when most text boxes are vertical (the photo was taken sideways)."""
    texty = [it for it in items if len(it.text.strip()) >= MIN_CHARS]
    if not texty:
        return False
    tall = sum(1 for it in texty if it.height > TALL_RATIO * it.width)
    return tall / len(texty) > TALL_SHARE


def _tilt(items: list[_Item]) -> float:
    """Median angle of the long text lines, 0 when there are too few to judge."""
    angles = [it.angle for it in items if it.width >= LONG_LINE * max(it.height, 1.0)]
    return median(angles) if len(angles) >= 5 else 0.0


def _score(items: list[_Item]) -> float:
    return sum(len(it.text.strip()) * it.score for it in items)


class RapidOcrEngine:
    """``OcrEngine`` backed by ``rapidocr.RapidOCR``; the model is created lazily, once."""

    name = "rapidocr"

    def __init__(self, detect_rotation: bool = True, deskew: bool = True) -> None:
        self.detect_rotation = detect_rotation
        self.deskew = deskew
        self._engine: Any = None

    def fingerprint(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **_rapidocr_info(),
            "orientation_version": ORIENTATION_VERSION,
            "detect_rotation": self.detect_rotation,
            "deskew": self.deskew,
        }

    def ensure_ready(self) -> None:
        self._get_engine()

    def _get_engine(self) -> Any:
        if self._engine is None:
            from rapidocr import RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        return self._engine

    def _read(self, image: Image.Image, use_cls: bool = True) -> list[_Item]:
        """Raw items in rapidocr's output order."""
        out = self._get_engine()(np.asarray(image), use_cls=use_cls)
        if out.boxes is None or out.txts is None or out.scores is None:
            return []
        items = []
        for poly, text, score in zip(out.boxes, out.txts, out.scores, strict=True):
            pts = np.asarray(poly, dtype=float)  # tl, tr, br, bl
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            dx, dy = pts[1] - pts[0]
            angle = math.degrees(math.atan2(dy, dx)) if dx > 0 else 0.0
            bbox = (float(x0), float(y0), float(x1), float(y1))
            items.append(_Item(str(text), float(score), bbox, angle))
        return items

    def run(self, image: Image.Image, image_id: str) -> OcrResult:
        start = time.perf_counter()
        rotation, page = 0, image
        items = self._read(image)

        if self.detect_rotation and _looks_sideways(items):
            trials = []
            for degrees, transpose in _TRANSPOSE.items():
                rotated = image.transpose(transpose)
                trials.append((rotated, degrees, self._read(rotated, use_cls=False)))
            rotated, degrees, best = max(trials, key=lambda t: _score(t[2]))
            if _score(best) >= KEEP_UPRIGHT * _score(items):
                rotation, page, items = degrees, rotated, best

        tilt = _tilt(items) if self.deskew else 0.0
        if DESKEW_MIN_DEG <= abs(tilt) <= DESKEW_MAX_DEG:
            degrees = round(tilt)
            # PIL rotates counter-clockwise; a line descending to the right needs a positive turn.
            leveled = page.rotate(
                degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"
            )
            leveled_items = self._read(leveled)
            if _score(leveled_items) >= KEEP_UPRIGHT * _score(items):
                rotation, page, items = (rotation + degrees) % 360, leveled, leveled_items

        texts = [it for it in items if it.text.strip()]
        lines = [
            OcrLine(id=i, text=it.text, bbox=it.bbox, confidence=it.score)
            for i, it in enumerate(texts)
        ]
        return OcrResult(
            image_id=image_id,
            width=page.width,
            height=page.height,
            lines=lines,
            engine=self.name,
            rotation=rotation,
            elapsed_s=round(time.perf_counter() - start, 3),
        )
