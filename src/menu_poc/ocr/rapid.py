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
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml
from PIL import Image

from menu_poc.constants import (
    DESKEW_MAX_DEG,
    DESKEW_MIN_DEG,
    KEEP_UPRIGHT,
    LONG_LINE,
    MIN_CHARS,
    MIN_TILT_VOTES,
    ORIENTATION_VERSION,
    ROTATION_TRANSPOSES,
    TALL_RATIO,
    TALL_SHARE,
)
from menu_poc.data_classes import OcrItem
from menu_poc.models import OcrLine, OcrResult


def _rapidocr_info() -> dict[str, Any]:
    """Version and model identifiers of the installed rapidocr, read without loading models."""
    import rapidocr

    config_path = Path(rapidocr.__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = ("engine_type", "lang_type", "model_type", "ocr_version")
    models = {
        task: {setting: config[task][setting] for setting in settings}
        for task in ("Det", "Cls", "Rec")
    }
    return {"rapidocr_version": version("rapidocr"), "models": models}


def _looks_sideways(items: list[OcrItem]) -> bool:
    """True when most text boxes are vertical (the photo was taken sideways)."""
    texty = [item for item in items if len(item.text.strip()) >= MIN_CHARS]
    if not texty:
        return False
    tall = sum(1 for item in texty if item.height > TALL_RATIO * item.width)
    return tall / len(texty) > TALL_SHARE


def _tilt(items: list[OcrItem]) -> float:
    """Median angle of the long text lines, 0 when there are too few to judge."""
    angles = [item.angle for item in items if item.width >= LONG_LINE * max(item.height, 1.0)]
    return median(angles) if len(angles) >= MIN_TILT_VOTES else 0.0


def _score(items: list[OcrItem]) -> float:
    return sum(len(item.text.strip()) * item.score for item in items)


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

    def _read(self, image: Image.Image, use_cls: bool = True) -> list[OcrItem]:
        """Raw items in rapidocr's output order."""
        output = self._get_engine()(np.asarray(image), use_cls=use_cls)
        if output.boxes is None or output.txts is None or output.scores is None:
            return []
        items = []
        for polygon, text, score in zip(output.boxes, output.txts, output.scores, strict=True):
            corners = np.asarray(polygon, dtype=float)  # tl, tr, br, bl
            x0, y0 = corners.min(axis=0)
            x1, y1 = corners.max(axis=0)
            dx, dy = corners[1] - corners[0]
            angle = math.degrees(math.atan2(dy, dx)) if dx > 0 else 0.0
            bbox = (float(x0), float(y0), float(x1), float(y1))
            items.append(OcrItem(str(text), float(score), bbox, angle))
        return items

    def run(self, image: Image.Image, image_id: str) -> OcrResult:
        start = time.perf_counter()
        rotation, page = 0, image
        items = self._read(image)

        if self.detect_rotation and _looks_sideways(items):
            trials = []
            for degrees, transpose in ROTATION_TRANSPOSES.items():
                rotated = image.transpose(transpose)
                trials.append((rotated, degrees, self._read(rotated, use_cls=False)))
            rotated, degrees, best = max(trials, key=lambda trial: _score(trial[2]))
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

        texts = [item for item in items if item.text.strip()]
        lines = [
            OcrLine(id=line_id, text=item.text, bbox=item.bbox, confidence=item.score)
            for line_id, item in enumerate(texts)
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
