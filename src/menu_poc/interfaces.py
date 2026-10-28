"""Stage interfaces. Each stage is swappable: a new OCR engine, parser or LLM is one new class.

Every stage exposes a ``fingerprint()``: the settings that change its output. It is part of the
artifact cache key, so changing an engine version, a parameter or a prompt invalidates the cache.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from menu_poc.llm.engine import LLMEngine  # noqa: F401  (re-exported: stage interfaces live here)
from menu_poc.models import (
    Dish,
    OcrResult,
    ParseResult,
    StructureResult,
    VerificationResult,
)


class Preprocessor(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def __call__(self, image: Image.Image) -> Image.Image: ...


class OcrEngine(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def ensure_ready(self) -> None:
        """Load the models now; raise if the engine cannot run (checked before any image)."""
        ...

    def run(self, image: Image.Image, image_id: str) -> OcrResult: ...


class Parser(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def parse(self, ocr: OcrResult) -> ParseResult: ...


class Structurer(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def structure(self, ocr: OcrResult) -> StructureResult: ...


class Verifier(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def verify(self, dishes: list[Dish], ocr: OcrResult) -> VerificationResult: ...
