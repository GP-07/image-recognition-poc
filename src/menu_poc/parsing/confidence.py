"""How well did the rule-based parser understand a page? A documented 0..1 score.

The score answers "can we trust the rules here, or should the LLM look at this page?". It is the
weighted mean of four signals in 0..1, scaled down when the page yields suspiciously few dishes:

- ``price_attach``: share of the price tokens on the page that ended up in a dish. Prices are the
  parser's anchor; prices left over mean a layout it did not follow (size tables, prices under
  photos, per-section prices). Weight 0.40.
- ``priced_dishes``: share of dishes that carry a price. Dishes found only by numbering or
  position are weaker evidence. Weight 0.20.
- ``coverage``: share of content lines (noise removed) used as a dish, description or section.
  Weight 0.20.
- ``ocr_conf``: mean OCR confidence of the content lines. Weight 0.20.

The weighted mean is then multiplied by five factors in 0..1. Each one is a failure sign that
the mean would dilute, so it caps the score instead:

- ``volume = min(1, dishes / MIN_DISHES)``;
- ``density = min(1, (dishes / content lines) / MIN_DENSITY)``: many lines but few dishes means
  the page was not understood, even if those few dishes look clean;
- ``recall_proxy = dishes / (dishes + leftover prices)`` (1 when the page has fewer than
  ``MIN_PRICE_TOKENS`` prices): every price left over is probably a dish the rules missed (size
  tables, price grids, prices on another row), so this estimates the share of dishes found;
- ``name_gate = min(1, (1 - suspicious) / (1 - MAX_SUSPICIOUS_NAMES))``, where ``suspicious`` is
  the share of names that read like a description or are a fragment of a few letters: the rules
  took the wrong line as the name, or the name was cut;
- ``header_gate = min(1, MAX_HEADERS_PER_DISH / (headers / dishes))``: a menu has far fewer
  section titles than dishes; more titles than that means dish names in capitals were taken
  for titles.

No dish means confidence 0.
"""

from __future__ import annotations

from menu_poc.constants import (
    CONFIDENCE_GATES,
    CONFIDENCE_WEIGHTS,
    MAX_HEADERS_PER_DISH,
    MAX_SUSPICIOUS_NAMES,
    MIN_DENSITY,
    MIN_DISHES,
    MIN_PRICE_TOKENS,
)
from menu_poc.data_classes import ParseStats


def score(stats: ParseStats) -> tuple[float, dict[str, float]]:
    """Return (confidence, signals). Signals are stored in ``ParseResult.signals``."""
    content = max(stats.content_lines, 1)
    signals = {
        "price_attach": stats.attached_prices / stats.price_tokens if stats.price_tokens else 0.0,
        "priced_dishes": stats.priced_dishes / stats.dishes if stats.dishes else 0.0,
        "coverage": min(1.0, stats.used_lines / content),
        "ocr_conf": stats.mean_ocr_conf,
        "volume": min(1.0, stats.dishes / MIN_DISHES),
        "density": min(1.0, stats.dishes / content / MIN_DENSITY),
        "recall_proxy": 1.0,
        "name_gate": 1.0,
        "header_gate": 1.0,
        "dishes": float(stats.dishes),
        "content_lines": float(stats.content_lines),
        "price_tokens": float(stats.price_tokens),
        "headers": float(stats.headers),
    }
    if stats.dishes == 0:
        return 0.0, _rounded(signals)
    if stats.price_tokens >= MIN_PRICE_TOKENS:
        leftover = max(0, stats.price_tokens - stats.attached_prices)
        signals["recall_proxy"] = stats.dishes / (stats.dishes + leftover)
    suspicious = stats.suspicious_names / stats.dishes
    signals["name_gate"] = min(1.0, (1 - suspicious) / (1 - MAX_SUSPICIOUS_NAMES))
    if stats.headers:
        signals["header_gate"] = min(1.0, MAX_HEADERS_PER_DISH * stats.dishes / stats.headers)
    value = sum(weight * signals[name] for name, weight in CONFIDENCE_WEIGHTS.items())
    for gate in CONFIDENCE_GATES:
        value *= signals[gate]
    return round(max(0.0, min(1.0, value)), 3), _rounded(signals)


def _rounded(signals: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 3) for name, value in signals.items()}
