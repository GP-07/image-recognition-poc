"""Output helpers for run directories."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from menu_poc.models import Dish, ImageResult

M = TypeVar("M", bound=BaseModel)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


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
        return [f.name for f in fields(cls)]


def write_items_csv(path: Path, results: Iterable[ImageResult]) -> int:
    """Write one ``ItemRow`` per extracted item; returns the row count. UTF-8 with BOM so
    spreadsheet apps show accents correctly."""
    rows = [ItemRow.from_dish(dish, result) for result in results for dish in result.dishes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ItemRow.columns())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return len(rows)


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    with Path(path).open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]
