"""Type aliases shared across modules. Stage interfaces (Protocols) live in ``interfaces``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 (axis-aligned)
DishOrigin = Literal["rules", "llm"]
LLMPolicy = Literal["fallback", "always"]
Split = Literal["dev", "test"]

# Recorded answers of the fake LLM engine: by prompt key, or a ``(system, user, schema)`` callable.
FakeResponses = Mapping[str, dict[str, Any]] | Callable[[str, str, dict[str, Any]], dict[str, Any]]
Candidate = tuple[str, list[int]]  # grounding: normalised OCR text and its line ids

ModelT = TypeVar("ModelT", bound=BaseModel)
ValueT = TypeVar("ValueT")
