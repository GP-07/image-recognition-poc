"""Data contracts shared by every pipeline stage.

Coordinates are pixels in the *preprocessed* image (after EXIF transpose, resize and
page rotation), with the origin at the top-left corner.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 (axis-aligned)


class OcrLine(BaseModel):
    id: int
    text: str
    bbox: BBox
    confidence: float

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


class OcrResult(BaseModel):
    image_id: str
    width: int
    height: int
    lines: list[OcrLine]
    engine: str
    rotation: int = 0  # total counter-clockwise degrees applied, incl. small deskew (e.g. 358, 88)
    elapsed_s: float = 0.0

    def line(self, line_id: int) -> OcrLine:
        return next(line for line in self.lines if line.id == line_id)


class Dish(BaseModel):
    name: str
    description: str | None = None
    prices: list[float] = Field(default_factory=list)
    section: str | None = None
    source_line_ids: list[int] = Field(default_factory=list)
    origin: Literal["rules", "llm"]


class ParseResult(BaseModel):
    dishes: list[Dish]
    confidence: float  # 0..1, how well the rules understood the page
    signals: dict[str, float] = Field(default_factory=dict)  # inputs of the confidence score


class LLMUsage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False  # served from the artifact store, no API call made


class StructureResult(BaseModel):
    dishes: list[Dish]
    usage: LLMUsage
    prompt_version: str


class VerificationResult(BaseModel):
    kept: list[Dish]
    dropped: list[Dish]  # not found in the OCR text (likely hallucinated)


class ImageResult(BaseModel):
    image_id: str
    restaurant_id: str
    dishes: list[Dish] = Field(default_factory=list)
    rules_confidence: float = 0.0
    rules_dish_count: int = 0  # kept even when the LLM answer replaces the rules dishes
    rules_signals: dict[str, float] = Field(default_factory=dict)  # to calibrate the threshold
    llm_used: bool = False
    llm_usage: LLMUsage | None = None  # also set for a failed but billed call
    llm_prompt_version: str | None = None
    ungrounded_dropped: list[str] = Field(default_factory=list)
    timings_s: dict[str, float] = Field(default_factory=dict)
    duplicate_of: str | None = None  # same photo as this image id: dishes copied, nothing run
    duplicate_reason: str | None = None
    error: str | None = None
