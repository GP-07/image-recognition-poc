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

import unicodedata

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from menu_poc.constants import (
    MATCH_LEADERS_RE,
    MATCH_NUMBERING_RE,
    MATCH_PRICE_RE,
    MATCH_THRESHOLD,
    NON_WORD_RE,
    SPACES_RE,
    TIE_BREAK_SCALE,
    TRAILING_NUMBER_RE,
)
from menu_poc.models import GoldenDish, ImageMatch, MatchedPair


def normalize(text: str) -> str:
    """Lowercase, strip accents, numbering, dotted leaders, prices and punctuation."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = MATCH_NUMBERING_RE.sub(" ", text)
    text = MATCH_LEADERS_RE.sub(" ", text)
    text = MATCH_PRICE_RE.sub(" ", text)
    text = TRAILING_NUMBER_RE.sub(" ", text)  # "lamb curry 12,50" (price without currency)
    text = NON_WORD_RE.sub(" ", text).replace("_", " ")
    return SPACES_RE.sub(" ", text).strip()


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


def _best_variant(pred_norm: str, dish: GoldenDish) -> tuple[float, str]:
    best = (0.0, dish.name)
    for variant in dish.variants:
        score = similarity(pred_norm, normalize(variant))
        if score > best[0]:
            best = (score, variant)
    return best


def match_dishes(
    predicted: list[str], gold: list[GoldenDish], threshold: float = MATCH_THRESHOLD
) -> ImageMatch:
    """Match predicted names to gold dishes of one image (see module docstring)."""
    pred_norm = [normalize(name) for name in predicted]
    first_seen: dict[str, int] = {}
    repeats: set[int] = set()
    for pred_index, name in enumerate(pred_norm):
        if name and name in first_seen:
            repeats.add(pred_index)
        first_seen.setdefault(name, pred_index)

    scores: dict[tuple[int, int], tuple[float, str]] = {}
    weight = np.zeros((len(predicted), len(gold)))
    for pred_index, name in enumerate(pred_norm):
        for gold_index, dish in enumerate(gold):
            score, variant = _best_variant(name, dish)
            scores[pred_index, gold_index] = (score, variant)
            if score >= threshold and pred_index not in repeats:
                # Plain edit similarity as a small tie-breaker between equal scores.
                tie_break = fuzz.ratio(name, normalize(variant)) / TIE_BREAK_SCALE
                weight[pred_index, gold_index] = score + tie_break

    out = ImageMatch()
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    if weight.size:
        rows, cols = linear_sum_assignment(weight, maximize=True)
        for pred_index, gold_index in sorted(zip(rows.tolist(), cols.tolist(), strict=True)):
            if weight[pred_index, gold_index] <= 0:
                continue
            used_pred.add(pred_index)
            used_gold.add(gold_index)
            score, variant = scores[pred_index, gold_index]
            out.matched.append(
                MatchedPair(
                    pred=predicted[pred_index],
                    gold=gold[gold_index].name,
                    variant=variant,
                    score=score,
                )
            )

    # Leftovers, including repeats of a matched prediction, become duplicates when they match an
    # already matched gold dish, and false positives otherwise.
    for pred_index, name in enumerate(predicted):
        if pred_index in used_pred:
            continue
        hits = [
            (scores[pred_index, gold_index], gold_index)
            for gold_index in used_gold
            if scores[pred_index, gold_index][0] >= threshold
        ]
        if hits:
            (score, variant), gold_index = max(hits, key=lambda hit: hit[0][0])
            out.duplicate_pairs.append(
                MatchedPair(pred=name, gold=gold[gold_index].name, variant=variant, score=score)
            )
        else:
            out.false_positives.append(name)

    out.unmatched_gold = [
        dish.name for gold_index, dish in enumerate(gold) if gold_index not in used_gold
    ]
    out.tp = len(out.matched)
    out.fp = len(out.false_positives)
    out.fn = len(out.unmatched_gold)
    out.duplicates = len(out.duplicate_pairs)
    return out
