"""Match predicted dish names against the golden dishes of one image.

Rules:
- Only the predicted ``name`` is compared (never ``name + description``).
- A gold dish is a name plus language aliases; the score against a gold dish is the best score over
  its variants.
- Repeated predictions (same normalized name) are counted once; the repeats are ``duplicates``.
- One-to-one assignment that maximises the total score (Hungarian algorithm), so a prediction
  that matches several gold dishes does not block the one it matches best.
- A leftover prediction that matches a gold dish already matched counts as a ``duplicate`` (the
  same dish in another language, or repeated), not as a false positive.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from menu_poc.evaluation.golden import GoldenDish

DEFAULT_THRESHOLD = 85.0

_NUMBERING = re.compile(r"^\s*(?:n[ºo°]\s*)?\d{1,3}\s*[.)\-:]\s*|^\s*\d{1,3}\s+(?=\D)")
_PRICE = re.compile(r"\d+(?:[.,]\d{1,2})?\s*(?:€|eur\b|euros?\b)|(?:€|\$)\s*\d+(?:[.,]\d{1,2})?")
# Only decimal numbers: "lamb curry 12,50" is a price, but "combo 2" is part of the name.
_TRAILING_NUMBER = re.compile(r"\s\d+[.,]\d{1,2}\s*$")
_LEADERS = re.compile(r"[.·…_\-]{2,}")
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents, numbering, dotted leaders, prices and punctuation."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _NUMBERING.sub(" ", text)
    text = _LEADERS.sub(" ", text)
    text = _PRICE.sub(" ", text)
    text = _TRAILING_NUMBER.sub(" ", text)  # "lamb curry 12,50" (price without currency)
    text = _NON_WORD.sub(" ", text).replace("_", " ")
    return _SPACES.sub(" ", text).strip()


def similarity(pred: str, gold: str) -> float:
    """Score in 0..100 between two *normalized* names.

    ``max(ratio, token_sort_ratio)`` absorbs OCR character noise ("cury" vs "curry") and word
    order. ``token_set_ratio`` (100 when one token set contains the other) is used only when the
    prediction is at least as long as the gold name, so extra words in the prediction ("lamb curry
    tradicional") still match, but a truncated prediction ("curry") does not get credit for a
    longer gold dish ("lamb curry"). ``WRatio`` was discarded because its partial matching scores
    such truncations at 90 and even cross-dish pairs above 85.
    """
    if not pred or not gold:
        return 0.0
    score = max(fuzz.ratio(pred, gold), fuzz.token_sort_ratio(pred, gold))
    if len(pred) >= len(gold):
        score = max(score, fuzz.token_set_ratio(pred, gold))
    return float(score)


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


def _best_variant(pred_norm: str, dish: GoldenDish) -> tuple[float, str]:
    best = (0.0, dish.name)
    for variant in dish.variants:
        score = similarity(pred_norm, normalize(variant))
        if score > best[0]:
            best = (score, variant)
    return best


def match_dishes(
    predicted: list[str], gold: list[GoldenDish], threshold: float = DEFAULT_THRESHOLD
) -> ImageMatch:
    """Match predicted names to gold dishes of one image (see module docstring)."""
    pred_norm = [normalize(p) for p in predicted]
    first_seen: dict[str, int] = {}
    repeats: set[int] = set()
    for i, p in enumerate(pred_norm):
        if p and p in first_seen:
            repeats.add(i)
        first_seen.setdefault(p, i)

    scores: dict[tuple[int, int], tuple[float, str]] = {}
    weight = np.zeros((len(predicted), len(gold)))
    for i, p in enumerate(pred_norm):
        for j, dish in enumerate(gold):
            score, variant = _best_variant(p, dish)
            scores[i, j] = (score, variant)
            if score >= threshold and i not in repeats:
                # Plain edit similarity as a small tie-breaker between equal scores.
                weight[i, j] = score + fuzz.ratio(p, normalize(variant)) / 1000

    out = ImageMatch()
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    if weight.size:
        rows, cols = linear_sum_assignment(weight, maximize=True)
        for i, j in sorted(zip(rows.tolist(), cols.tolist(), strict=True)):
            if weight[i, j] <= 0:
                continue
            used_pred.add(i)
            used_gold.add(j)
            score, variant = scores[i, j]
            out.matched.append(
                MatchedPair(pred=predicted[i], gold=gold[j].name, variant=variant, score=score)
            )

    # Leftovers, including repeats of a matched prediction, become duplicates when they match an
    # already matched gold dish, and false positives otherwise.
    for i, name in enumerate(predicted):
        if i in used_pred:
            continue
        hits = [(scores[i, j], j) for j in used_gold if scores[i, j][0] >= threshold]
        if hits:
            (score, variant), j = max(hits, key=lambda h: h[0][0])
            out.duplicate_pairs.append(
                MatchedPair(pred=name, gold=gold[j].name, variant=variant, score=score)
            )
        else:
            out.false_positives.append(name)

    out.unmatched_gold = [d.name for j, d in enumerate(gold) if j not in used_gold]
    out.tp = len(out.matched)
    out.fp = len(out.false_positives)
    out.fn = len(out.unmatched_gold)
    out.duplicates = len(out.duplicate_pairs)
    return out
