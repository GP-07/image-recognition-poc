"""Grounding check: keep an LLM dish only if its name is actually in the OCR text.

Satisfies ``menu_poc.interfaces.Verifier``. Names and OCR text are compared in normalised form
(lowercase, no accents, punctuation or leading numbering) with rapidfuzz, so OCR noise and small
"corrections" by the model still match, while invented dishes do not.

Matching order for each dish:
1. the text of its ``source_line_ids`` (each line, and all of them joined in reading order);
   ids that do not exist in the OCR are removed;
2. otherwise any single OCR line, or a line joined with its reading-order neighbour or with the
   line right below it in the same column (names split over two lines). On a match there, the
   dish's ``source_line_ids`` are replaced by the matching line ids.

Score = min(phrase score, worst word score):
- phrase: ``partial_ratio`` when the name is not longer than the candidate text, else plain
  ``ratio`` (so "Pollo tikka con arroz" does not match a line "Pollo");
- words: every name word of 3+ characters must appear (``partial_ratio``, which tolerates OCR
  noise and merged words) in the candidate, so "Croquetas de jamón" does not match
  "Croquetas de sémola"; a whole-phrase fuzzy score alone is too lenient for that.
Names of 3 characters or less must equal a whole word of the candidate.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from menu_poc.constants import (
    BELOW_MAX_GAP,
    BELOW_MIN_DROP,
    GROUNDING_THRESHOLD,
    GROUNDING_VERSION,
    MIN_WORD_CHARS,
    SHORT_NAME_CHARS,
)
from menu_poc.llm.text import normalize, reading_order
from menu_poc.models import Dish, OcrLine, OcrResult, VerificationResult
from menu_poc.typings import Candidate


def match_score(name: str, text: str) -> float:
    """Similarity (0-100) of a normalised name against a normalised candidate text."""
    if not name or not text:
        return 0.0
    if len(name) <= SHORT_NAME_CHARS:
        return 100.0 if name in text.split() else 0.0
    phrase = fuzz.partial_ratio(name, text) if len(name) <= len(text) else fuzz.ratio(name, text)
    words = [fuzz.partial_ratio(word, text) for word in name.split() if len(word) >= MIN_WORD_CHARS]
    return min([phrase, *words])


def _line_below(line: OcrLine, lines: list[OcrLine]) -> OcrLine | None:
    """Nearest line under ``line`` whose horizontal span overlaps it (same column)."""
    below = [
        other
        for other in lines
        if other.y_center > line.y1 - BELOW_MIN_DROP * line.height
        and other.y0 - line.y1 < BELOW_MAX_GAP * max(line.height, 1.0)
        and min(line.x1, other.x1) - max(line.x0, other.x0) > 0
        and other.id != line.id
    ]
    return min(below, key=lambda other: other.y_center, default=None)


def page_candidates(ocr: OcrResult) -> list[Candidate]:
    """Every line, plus each line joined with its reading-order neighbour and the line below."""
    ordered = reading_order(ocr.lines)
    norm = {line.id: normalize(line.text) for line in ordered}
    candidates: list[Candidate] = [(norm[line.id], [line.id]) for line in ordered]
    pairs: set[tuple[int, int]] = set()
    for current, following in zip(ordered, ordered[1:], strict=False):
        pairs.add((current.id, following.id))
    for line in ordered:
        below = _line_below(line, ordered)
        if below is not None:
            pairs.add((line.id, below.id))
    for first_id, second_id in sorted(pairs):
        joined = f"{norm[first_id]} {norm[second_id]}".strip()
        candidates.append((joined, [first_id, second_id]))
    return candidates


class GroundingVerifier:
    def __init__(self, threshold: float = GROUNDING_THRESHOLD) -> None:
        self.threshold = threshold

    def fingerprint(self) -> dict[str, Any]:
        return {"stage": "grounding", "threshold": self.threshold, "version": GROUNDING_VERSION}

    def verify(self, dishes: list[Dish], ocr: OcrResult) -> VerificationResult:
        by_id = {line.id: line for line in ocr.lines}
        order = {line.id: index for index, line in enumerate(reading_order(ocr.lines))}
        page: list[Candidate] | None = None  # built lazily, only if some dish needs it
        kept: list[Dish] = []
        dropped: list[Dish] = []

        for dish in dishes:
            name = normalize(dish.name)
            known = {line_id for line_id in dish.source_line_ids if line_id in by_id}
            ids = sorted(known, key=order.__getitem__)
            own: list[Candidate] = [(normalize(by_id[line_id].text), [line_id]) for line_id in ids]
            if len(ids) > 1:
                own.append((" ".join(text for text, _ in own), ids))

            if self._best(name, own) is not None:
                kept.append(dish.model_copy(update={"source_line_ids": ids}))
                continue
            if page is None:
                page = page_candidates(ocr)
            match = self._best(name, page)
            if match is not None:
                kept.append(dish.model_copy(update={"source_line_ids": match}))
            else:
                dropped.append(dish)
        return VerificationResult(kept=kept, dropped=dropped)

    def _best(self, name: str, candidates: list[Candidate]) -> list[int] | None:
        """Line ids of the best candidate scoring at least the threshold, else None."""
        best_ids, best_score = None, -1.0
        for text, ids in candidates:
            score = match_score(name, text)
            if score > best_score:
                best_ids, best_score = ids, score
        return best_ids if best_score >= self.threshold else None
