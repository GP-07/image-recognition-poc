"""Dataclasses: plain in-memory records of the pipeline (never validated or loaded from JSON;
that is what the pydantic ``models`` are for)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from PIL import Image

from menu_poc.cache.store import sha256_bytes
from menu_poc.constants import ITEMS_FILE
from menu_poc.models import (
    Dish,
    ImageResult,
    OcrLine,
    OcrResult,
    ParseResult,
    RunConfig,
    StructureResult,
)
from menu_poc.typings import BBox

# --- input and output -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    restaurant_id: str
    filename: str
    data: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data)

    def open(self) -> Image.Image:
        image = Image.open(io.BytesIO(self.data))
        image.load()
        return image


@dataclass(frozen=True)
class ItemRow:
    """One line of ``items.csv``: an extracted menu item and the image it comes from."""

    image_id: str
    restaurant_id: str
    item: str
    description: str
    prices: str  # every price of the item, "|"-separated (sizes): "8.5|12"
    section: str
    source: str  # "rules" or "llm"
    duplicate_of: str  # set when the photo duplicates another one (item copied from it)

    @classmethod
    def from_dish(cls, dish: Dish, result: ImageResult) -> ItemRow:
        """Flatten a dish of an image result into a CSV row (missing values -> empty strings)."""
        return cls(
            image_id=result.image_id,
            restaurant_id=result.restaurant_id,
            item=dish.name,
            description=dish.description or "",
            prices="|".join(f"{price:g}" for price in dish.prices),
            section=dish.section or "",
            source=dish.origin,
            duplicate_of=result.duplicate_of or "",
        )

    @classmethod
    def columns(cls) -> list[str]:
        return [column.name for column in fields(cls)]


# --- pipeline -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """What ``Pipeline.run`` produced and saved: where, with which settings, the results, and the
    figures the command reports."""

    run_dir: Path
    requested_llm_model: str | None
    config: RunConfig  # what actually ran: no LLM model when the LLM step was disabled
    results: list[ImageResult]
    warnings: list[str]
    started: str
    finished: str

    @property
    def items_path(self) -> Path:
        return self.run_dir / ITEMS_FILE

    @property
    def n_items(self) -> int:
        return sum(len(result.dishes) for result in self.results)

    @property
    def errors(self) -> dict[str, str]:
        return {result.image_id: result.error for result in self.results if result.error}

    @property
    def duplicates(self) -> dict[str, str]:
        return {
            result.image_id: result.duplicate_of for result in self.results if result.duplicate_of
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "run": self.run_dir.name,
            "started": self.started,
            "finished": self.finished,
            "requested_llm_model": self.requested_llm_model,
            "llm_model": self.config.llm_model,  # None: OCR + rules only (see warnings)
            "n_images": len(self.results),
            "n_dishes": self.n_items,
            "duplicates": self.duplicates,
            "warnings": self.warnings,
            "errors": self.errors,
        }


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


# --- OCR ------------------------------------------------------------------------------------------


@dataclass
class OcrItem:
    """One raw text box of the OCR engine, before it becomes an ``OcrLine``."""

    text: str
    score: float
    bbox: BBox
    angle: float  # degrees of the top edge; positive = descending to the right

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


# --- rule-based parser ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceMatch:
    value: float
    start: int
    end: int


@dataclass
class LineInfo:
    """What the parser knows about one OCR line."""

    line: OcrLine
    prices: list[PriceMatch]
    price_only: bool
    noise: bool
    numbered: bool
    column: int = 0
    consumed: bool = False  # price-only line whose prices went to a row partner
    attached: list[float] = field(default_factory=list)  # prices from same-row price-only lines
    attached_ids: list[int] = field(default_factory=list)


@dataclass
class DishDraft:
    """A dish being assembled while the parser walks the page."""

    head: LineInfo
    prices: list[float]
    line_ids: list[int]
    desc: list[str] = field(default_factory=list)
    section: str | None = None
    last: OcrLine | None = None


@dataclass(frozen=True)
class ParseStats:
    """Counts collected by the parser for one page."""

    content_lines: int  # OCR lines left after the noise filter
    used_lines: int  # content lines used as dish name, description or section header
    price_tokens: int  # prices found on content lines
    attached_prices: int  # of those, prices that ended in a dish
    dishes: int
    priced_dishes: int
    mean_ocr_conf: float
    suspicious_names: int = 0  # dish names that read like descriptions or are fragments
    headers: int = 0  # lines taken as section titles


# --- LLM ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
