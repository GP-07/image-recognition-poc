"""Pydantic models: the data contracts of the pipeline stages and of the evaluation.

Coordinates are pixels in the *preprocessed* image (after EXIF transpose, resize and
page rotation), with the origin at the top-left corner.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from menu_poc.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEDUP_MAX_DISTANCE,
    DEFAULT_GOLDEN_DIR,
    DEFAULT_MAX_IMAGE_SIDE,
    DEFAULT_RUNS_DIR,
    MATCH_THRESHOLD,
)
from menu_poc.typings import BBox, DishOrigin, LLMPolicy, Split

# --- run configuration ----------------------------------------------------------------------------


class RunConfig(BaseModel):
    input_path: Path
    runs_dir: Path = DEFAULT_RUNS_DIR  # one sub-directory per run
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR  # stage cache, reused across runs
    golden_dir: Path | None = DEFAULT_GOLDEN_DIR  # evaluate each run against it, when it exists

    # OCR + rules always run. The LLM step runs only when a model is configured: an alias
    # (haiku, sonnet, opus, ollama) or an exact id, stored resolved to the exact id.
    llm_model: str | None = None
    llm_policy: LLMPolicy = "fallback"
    llm_offline: bool = False  # only use cached LLM answers; never call the API

    confidence_threshold: float = Field(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    dedup: bool = True  # skip OCR/LLM for duplicate photos (same file or near-identical)
    dedup_max_distance: int = Field(DEFAULT_DEDUP_MAX_DISTANCE, ge=0)
    max_image_side: int = Field(DEFAULT_MAX_IMAGE_SIDE, gt=0)

    @field_validator("llm_model")
    @classmethod
    def _resolve_model(cls, value: str | None) -> str | None:
        # Imported here: the LLM engines import these models, so a top-level import is a cycle.
        from menu_poc.llm.engine import LLMEngine

        return None if value is None else LLMEngine.get_for_alias(value).model

    @property
    def uses_llm(self) -> bool:
        return self.llm_model is not None


# --- pipeline -------------------------------------------------------------------------------------


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
    origin: DishOrigin


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


# --- evaluation -----------------------------------------------------------------------------------


class GoldenDish(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dish name must not be empty")
        return value

    @property
    def variants(self) -> list[str]:
        """Name plus aliases (any of them counts as recognising the dish)."""
        return [self.name, *self.aliases]


class GoldenImage(BaseModel):
    image_id: str  # "<restaurant_id>/<file_stem>"
    split: Split
    reviewed: bool = False
    notes: str | None = None
    dishes: list[GoldenDish] = Field(default_factory=list)


class MatchedPair(BaseModel):
    pred: str
    gold: str  # canonical gold name
    variant: str  # name or alias that produced the best score
    score: float


class ImageMatch(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    matched: list[MatchedPair] = Field(default_factory=list)
    duplicate_pairs: list[MatchedPair] = Field(default_factory=list)
    unmatched_gold: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


class ImageMetrics(BaseModel):
    image_id: str
    restaurant_id: str
    labeled: bool
    split: Split | None = None
    n_pred: int
    n_gold: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    recall: float | None = None
    precision: float | None = None
    cost_usd: float = 0.0
    llm_used: bool = False
    llm_cached: bool = False
    llm_latency_s: float = 0.0
    time_s: float = 0.0
    ungrounded_dropped: int = 0
    duplicate_of: str | None = None  # duplicate photo: excluded from accuracy (counted once)
    error: str | None = None
    matched: list[MatchedPair] = Field(default_factory=list)
    duplicate_pairs: list[MatchedPair] = Field(default_factory=list)
    unmatched_gold: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


class AccuracyStats(BaseModel):
    n_images: int = 0
    n_gold: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    micro_recall: float | None = None
    micro_precision: float | None = None
    macro_recall: float | None = None
    macro_precision: float | None = None


class RunMetrics(BaseModel):
    run: str
    mode: str | None = None
    llm_model: str | None = None
    llm_policy: str | None = None
    threshold: float = MATCH_THRESHOLD
    only_reviewed: bool = False
    n_images: int = 0
    n_labeled: int = 0
    n_duplicate_images: int = 0  # skipped photos (same menu as another image)
    unlabeled: list[str] = Field(default_factory=list)
    accuracy: dict[str, AccuracyStats] = Field(default_factory=dict)  # keys: all, dev, test
    cost_total_usd: float = 0.0
    cost_per_image_usd: float = 0.0
    llm_share: float = 0.0  # share of images where the LLM was used
    llm_cached_share: float = 0.0  # share of LLM images served from cache (no API call)
    time_total_s: float = 0.0
    time_mean_s: float = 0.0
    llm_latency_total_s: float = 0.0
    llm_latency_mean_s: float = 0.0  # over images where the LLM was used
    ungrounded_dropped: int = 0
    n_errors: int = 0
    errors: dict[str, str] = Field(default_factory=dict)
    images: list[ImageMetrics] = Field(default_factory=list)
