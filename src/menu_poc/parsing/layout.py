"""Page geometry over OCR lines: rows, columns, reading order and same-row lookups.

Everything is a cheap 1-D heuristic on axis-aligned boxes; slightly tilted photos are tolerated
through the overlap thresholds, strongly warped pages are not (those go to the LLM).
"""

from __future__ import annotations

from statistics import median

from menu_poc.constants import BLOCK_OVERLAP, GUTTER_MAX_COVER, GUTTER_MIN_WIDTH, ROW_OVERLAP
from menu_poc.models import OcrLine


def median_height(lines: list[OcrLine]) -> float:
    return median(line.height for line in lines) if lines else 0.0


def v_overlap(first: OcrLine, second: OcrLine) -> float:
    """Vertical overlap of two boxes as a share of the shorter one (0..1)."""
    inter = min(first.y1, second.y1) - max(first.y0, second.y0)
    shorter = min(first.height, second.height)
    return max(0.0, inter) / shorter if shorter > 0 else 0.0


def group_rows(lines: list[OcrLine], min_overlap: float = ROW_OVERLAP) -> list[list[OcrLine]]:
    """Lines whose boxes overlap vertically, top to bottom; each row is sorted left to right."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda ln: ln.y_center):
        for row in reversed(rows[-3:]):  # rows are y-sorted: only the last few can match
            if any(v_overlap(line, other) >= min_overlap for other in row):
                row.append(line)
                break
        else:
            rows.append([line])
    return [sorted(row, key=lambda ln: ln.x0) for row in rows]


def find_gutters(lines: list[OcrLine], width: float) -> list[tuple[float, float]]:
    """Vertical white bands (x ranges) that separate columns of text."""
    if not lines or width <= 0:
        return []
    bins = int(width) + 1
    cover = [0] * bins
    for line in lines:
        for position in range(max(0, int(line.x0)), min(bins, int(line.x1) + 1)):
            cover[position] += 1
    limit = max(1, GUTTER_MAX_COVER * len(lines))
    min_width = GUTTER_MIN_WIDTH * median_height(lines)
    left = min(line.x0 for line in lines)
    right = max(line.x1 for line in lines)

    gutters, start = [], None
    for position in range(int(left), int(right) + 1):
        if cover[position] <= limit:
            start = position if start is None else start
        elif start is not None:
            if position - start >= min_width:
                gutters.append((float(start), float(position)))
            start = None
    return gutters


def group_columns(lines: list[OcrLine], width: float | None = None) -> list[list[OcrLine]]:
    """Split lines into columns at the gutters, left to right; each column sorted top to bottom.

    A line that spans a gutter (a page title) goes to the column holding its left edge.
    """
    if not lines:
        return []
    width = width if width is not None else max(line.x1 for line in lines) + 1
    cuts = [(start + end) / 2 for start, end in find_gutters(lines, width)]
    columns: list[list[OcrLine]] = [[] for _ in range(len(cuts) + 1)]
    for line in lines:
        index = sum(1 for cut in cuts if line.x0 + 1 > cut)
        columns[index].append(line)
    return [sorted(col, key=lambda ln: (ln.y_center, ln.x0)) for col in columns if col]


def reading_order(lines: list[OcrLine], width: float | None = None) -> list[OcrLine]:
    """Column by column (left to right), top to bottom inside each column."""
    return [line for col in group_columns(lines, width) for line in col]


def left_neighbor(
    target: OcrLine,
    lines: list[OcrLine],
    min_overlap: float = ROW_OVERLAP,
    max_gap: float | None = None,
    obstacles: list[OcrLine] | None = None,
) -> OcrLine | None:
    """The line on the same row, left of ``target``, that best shares its row (e.g. the name of a
    price). Highest vertical overlap wins (in steps of 0.1), then the closest one.

    ``max_gap`` limits the horizontal distance; by default it is unlimited because prices often
    sit in their own column far to the right of the name, joined by dotted leaders. A candidate is
    rejected when one of ``obstacles`` (default: ``lines``) sits between it and ``target`` on the
    row: the name of a price is never behind another text column.
    """
    obstacles = lines if obstacles is None else obstacles
    best, best_key = None, (-1.0, 0.0)
    for line in lines:
        if line.id == target.id or line.x_center >= target.x0:
            continue
        overlap = v_overlap(line, target)
        if overlap < min_overlap:
            continue
        gap = max(0.0, target.x0 - line.x1)
        if max_gap is not None and gap > max_gap:
            continue
        key = (round(overlap, 1), -gap)
        if key > best_key and not _blocked(line, target, obstacles):
            best, best_key = line, key
    return best


def _blocked(left: OcrLine, right: OcrLine, obstacles: list[OcrLine]) -> bool:
    """True if some obstacle lies horizontally between ``left`` and ``right`` on their row."""
    for other in obstacles:
        if other.id in (left.id, right.id):
            continue
        between = other.x0 >= left.x1 - 1 and other.x1 <= right.x0 + 1
        if between and v_overlap(other, right) > BLOCK_OVERLAP:
            return True
    return False
