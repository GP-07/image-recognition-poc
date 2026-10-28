"""Chain of Responsibility: every image travels through a chain of steps.

Each step does its own work on the shared ``ImageContext`` and decides whether the image goes on
to the next step. An exception stops the chain for that image only (the run continues), and the
output gathered so far is kept: e.g. if the LLM fails, the rules output stays.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from menu_poc.io.dataset import ImageRef
from menu_poc.models import ImageResult, OcrResult, ParseResult, StructureResult

log = logging.getLogger(__name__)


@dataclass
class ImageContext:
    """What the steps know about one image. Steps fill it in place, in chain order."""

    ref: ImageRef
    result: ImageResult
    ocr: OcrResult | None = None
    ocr_key: str | None = None  # cache key of the OCR output, reused by the LLM cache key
    parsed: ParseResult | None = None
    structured: StructureResult | None = None

    @classmethod
    def for_image(cls, ref: ImageRef) -> ImageContext:
        return cls(ref, ImageResult(image_id=ref.image_id, restaurant_id=ref.restaurant_id))


class Step(ABC):
    name: ClassVar[str]

    def __init__(self) -> None:
        self.next: Step | None = None

    def ensure_ready(self) -> None:  # noqa: B027  (optional hook, most steps need nothing)
        """Fail fast, before any image, when the step cannot run at all. Default: nothing."""

    def handle(self, ctx: ImageContext) -> ImageContext:
        try:
            proceed = self.process(ctx)
        except Exception as exc:  # one bad image must not stop the run
            log.exception("%s failed for %s", self.name, ctx.ref.image_id)
            ctx.result.error = f"{self.name}: {exc}"
            return ctx
        if proceed and self.next is not None:
            return self.next.handle(ctx)
        return ctx

    @abstractmethod
    def process(self, ctx: ImageContext) -> bool:
        """Do this step's work; return True to pass the image to the next step."""
