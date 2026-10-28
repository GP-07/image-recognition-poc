"""Stage interfaces (abstract base classes): a new OCR engine, parser, LLM or post-processor is one
new subclass. A subclass that misses a method fails when it is created, not in the middle of a run.

Every stage exposes a ``fingerprint()``: the settings that change its output. It is part of the
artifact cache key, so changing an engine version, a parameter or a prompt invalidates the cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from menu_poc.data_classes import RunOutcome
from menu_poc.models import (
    Dish,
    OcrResult,
    ParseResult,
    StructureResult,
    VerificationResult,
)


class Preprocessor(ABC):
    @abstractmethod
    def fingerprint(self) -> dict[str, Any]: ...

    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image: ...


class OcrEngine(ABC):
    @abstractmethod
    def fingerprint(self) -> dict[str, Any]: ...

    @abstractmethod
    def ensure_ready(self) -> None:
        """Load the models now; raise if the engine cannot run (checked before any image)."""

    @abstractmethod
    def run(self, image: Image.Image, image_id: str) -> OcrResult: ...


class Parser(ABC):
    @abstractmethod
    def fingerprint(self) -> dict[str, Any]: ...

    @abstractmethod
    def parse(self, ocr: OcrResult) -> ParseResult: ...


class Structurer(ABC):
    @abstractmethod
    def fingerprint(self) -> dict[str, Any]: ...

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """None when the model can be called, else why not (checked before any image)."""

    @abstractmethod
    def structure(self, ocr: OcrResult) -> StructureResult: ...


class Verifier(ABC):
    @abstractmethod
    def fingerprint(self) -> dict[str, Any]: ...

    @abstractmethod
    def verify(self, dishes: list[Dish], ocr: OcrResult) -> VerificationResult: ...


class PostProcessor(ABC):
    """Runs once on a finished run, after its files are saved (e.g. evaluation, exports)."""

    name: ClassVar[str]

    @abstractmethod
    def run(self, outcome: RunOutcome) -> list[Path]:
        """Do the work; return the files written into the run directory."""
