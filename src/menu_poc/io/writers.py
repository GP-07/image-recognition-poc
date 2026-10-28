"""Output helpers for run directories."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

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


ITEM_COLUMNS = [
    "image_id",
    "restaurant_id",
    "item",
    "description",
    "prices",
    "section",
    "source",
    "duplicate_of",
]


def write_items_csv(path: Path, results: Iterable[Any]) -> int:
    """One row per extracted menu item (the human-facing output). Returns the row count.

    ``prices`` holds every price of the item separated by ``|`` (sizes). UTF-8 with BOM so
    spreadsheet apps show accents correctly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ITEM_COLUMNS)
        writer.writeheader()
        for result in results:
            for dish in result.dishes:
                writer.writerow(
                    {
                        "image_id": result.image_id,
                        "restaurant_id": result.restaurant_id,
                        "item": dish.name,
                        "description": dish.description or "",
                        "prices": "|".join(f"{p:g}" for p in dish.prices),
                        "section": dish.section or "",
                        "source": dish.origin,
                        "duplicate_of": result.duplicate_of or "",
                    }
                )
                rows += 1
    return rows


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    with Path(path).open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]
