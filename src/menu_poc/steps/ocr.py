"""Preprocess + OCR, cached by image content and engine settings."""

from __future__ import annotations

import time

from menu_poc.cache.store import ArtifactStore, make_key
from menu_poc.data_classes import ImageContext
from menu_poc.interfaces import OcrEngine, Preprocessor
from menu_poc.models import OcrResult
from menu_poc.steps.base import InputValidationError, Step


class OcrStep(Step):
    name = "ocr"

    def __init__(self, preprocessor: Preprocessor, engine: OcrEngine, store: ArtifactStore) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.engine = engine
        self.store = store

    def validate(self) -> list[str]:
        """Load the OCR models now: without OCR there is nothing to run, so this one is fatal."""
        try:
            self.engine.ensure_ready()
        except Exception as exc:
            raise InputValidationError(f"the OCR engine cannot run: {exc}") from exc
        return []

    def process(self, ctx: ImageContext) -> bool:
        ref = ctx.ref
        key = make_key(
            "ocr", ref.sha256, self.preprocessor.fingerprint(), self.engine.fingerprint()
        )
        cached = self.store.get("ocr", key)
        if cached is not None:
            # Identical bytes under another id share the entry: re-label it for this image.
            ocr = OcrResult.model_validate(cached).model_copy(update={"image_id": ref.image_id})
        else:
            start = time.perf_counter()
            ocr = self.engine.run(self.preprocessor(ref.open()), ref.image_id)
            if not ocr.elapsed_s:
                ocr.elapsed_s = time.perf_counter() - start
            self.store.put("ocr", key, ocr.model_dump(mode="json"))
        ctx.ocr, ctx.ocr_key = ocr, key
        # Time of producing the OCR, also when served from the cache: runs stay comparable.
        ctx.result.timings_s["ocr"] = ocr.elapsed_s
        return True
