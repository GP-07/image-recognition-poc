"""LLM structurer: OCR text -> dishes, satisfying ``menu_poc.interfaces.Structurer``.

The model sees only the OCR text (never the image), as ``[id] (x,y) text`` lines in reading
order, and answers with JSON constrained by ``DISHES_SCHEMA``. No caching here: the pipeline
caches answers generically using ``fingerprint()``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from menu_poc.interfaces import LLMClient
from menu_poc.llm.text import render_lines
from menu_poc.models import Dish, LLMUsage, OcrResult, StructureResult

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "structure_v1.md"
RENDER_VERSION = "lines_v1"  # bump when render_lines / the user message format changes

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

DISHES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dishes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": _NULLABLE_STRING,
                    "prices": {"type": "array", "items": {"type": "number"}},
                    "section": _NULLABLE_STRING,
                    "source_line_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "description", "prices", "section", "source_line_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dishes"],
    "additionalProperties": False,
}


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_user_message(ocr: OcrResult) -> str:
    return (
        f"Menu photo OCR, {len(ocr.lines)} lines.\nFormat: [id] (x,y) text\n\n{render_lines(ocr)}"
    )


class LLMStructurer:
    def __init__(self, client: LLMClient, prompt_path: Path = DEFAULT_PROMPT_PATH) -> None:
        self.client = client
        self.prompt_path = Path(prompt_path)
        self.system = self.prompt_path.read_text(encoding="utf-8")
        # Name + content hash: editing the prompt invalidates cached answers automatically.
        self.prompt_version = f"{self.prompt_path.stem}@{_short_hash(self.system)}"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "stage": "llm_structurer",
            "client": self.client.fingerprint(),  # model + settings such as effort
            "prompt_version": self.prompt_version,
            "schema": _short_hash(json.dumps(DISHES_SCHEMA, sort_keys=True)),
            "render": RENDER_VERSION,
        }

    def structure(self, ocr: OcrResult) -> StructureResult:
        if not ocr.lines:  # nothing to read: skip the paid call
            usage = LLMUsage(model=self.client.model)
            return StructureResult(dishes=[], usage=usage, prompt_version=self.prompt_version)
        data, usage = self.client.complete_json(self.system, build_user_message(ocr), DISHES_SCHEMA)
        dishes = [dish for item in data.get("dishes", []) if (dish := _to_dish(item))]
        return StructureResult(dishes=dishes, usage=usage, prompt_version=self.prompt_version)


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _to_dish(item: Any) -> Dish | None:
    """Convert one JSON item to a Dish; tolerant of odd values, drops items without a name."""
    if not isinstance(item, dict):
        return None
    name = _clean_str(item.get("name"))
    if name is None:
        return None
    prices = [
        float(p)
        for p in item.get("prices") or []
        if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and p >= 0
    ]
    ids = [i for i in item.get("source_line_ids") or [] if isinstance(i, int)]
    return Dish(
        name=name,
        description=_clean_str(item.get("description")),
        prices=prices,
        section=_clean_str(item.get("section")),
        source_line_ids=list(dict.fromkeys(ids)),
        origin="llm",
    )
