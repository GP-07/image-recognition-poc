"""Text helpers shared by the LLM structurer and the grounding check.

- ``reading_order``: OCR lines sorted top-to-bottom, left-to-right within a visual row.
- ``render_lines``: the compact ``[id] (x,y) text`` listing sent to the LLM.
- ``normalize``: the comparison form of a string (lowercase, no accents/punctuation/numbering).
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata

from menu_poc.models import OcrLine, OcrResult

_NUMBERING = re.compile(r"^\s*(?:\d{1,3}|[a-z])\s*[.)\-:]\s*|^\s*\d{1,3}\s+(?=[^\W\d])")
_NON_ALNUM = re.compile(r"[\W_]+")  # Unicode-aware: keeps Greek, Cyrillic, CJK, Devanagari...


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Lowercase, strip accents, leading numbering ("12.", "a)", "46 ") and punctuation."""
    text = strip_accents(text).lower()
    text = _NUMBERING.sub("", text, count=1)
    return _NON_ALNUM.sub(" ", text).strip()


def estimate_tokens(text: str) -> int:
    """Rough token count (characters / 4); good enough for cost estimates."""
    return math.ceil(len(text) / 4)


def reading_order(lines: list[OcrLine], row_tolerance: float = 0.5) -> list[OcrLine]:
    """Sort lines by row (y_center), then by x0 within a row.

    Lines whose centres are within ``row_tolerance`` x the median line height of the row's
    first line belong to the same row, so a price far to the right sits next to its name.
    """
    if not lines:
        return []
    median_h = statistics.median(max(line.height, 1.0) for line in lines)
    tolerance = row_tolerance * median_h
    rows: list[list[OcrLine]] = []
    anchor = -math.inf
    for line in sorted(lines, key=lambda ln: (ln.y_center, ln.x0)):
        if rows and line.y_center - anchor <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
            anchor = line.y_center
    return [line for row in rows for line in _order_row(row)]


def _order_row(row: list[OcrLine]) -> list[OcrLine]:
    """Left to right; lines that overlap horizontally (stacked in one column) stay top-down."""
    groups: list[list[OcrLine]] = []
    right = -math.inf
    for line in sorted(row, key=lambda ln: ln.x0):
        if groups and line.x0 < right:
            groups[-1].append(line)
            right = max(right, line.x1)
        else:
            groups.append([line])
            right = line.x1
    return [line for group in groups for line in sorted(group, key=lambda ln: ln.y_center)]


def render_lines(ocr: OcrResult) -> str:
    """One OCR line per row: ``[id] (x,y) text``, x = left edge and y = vertical centre,
    both as integer percentages of the page width/height."""
    width = max(ocr.width, 1)
    height = max(ocr.height, 1)
    rows = []
    for line in reading_order(ocr.lines):
        x = round(100 * line.x0 / width)
        y = round(100 * line.y_center / height)
        text = " ".join(line.text.split())
        rows.append(f"[{line.id}] ({x},{y}) {text}")
    return "\n".join(rows)
