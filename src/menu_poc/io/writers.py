"""Output helpers for run directories."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from menu_poc.data_classes import ItemRow
from menu_poc.models import ImageResult
from menu_poc.typings import ModelT


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
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(item.model_dump_json() + "\n")


def write_items_csv(path: Path, results: Iterable[ImageResult]) -> int:
    """Write one ``ItemRow`` per extracted item; returns the row count. UTF-8 with BOM so
    spreadsheet apps show accents correctly."""
    rows = [ItemRow.from_dish(dish, result) for result in results for dish in result.dishes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ItemRow.columns())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return len(rows)


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    with Path(path).open(encoding="utf-8") as stream:
        return [model.model_validate_json(line) for line in stream if line.strip()]
