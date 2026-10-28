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

from dataclasses import dataclass

WEIGHTS = {"price_attach": 0.40, "priced_dishes": 0.20, "coverage": 0.20, "ocr_conf": 0.20}
MIN_DISHES = 4  # fewer dishes than this scales the score down linearly
MIN_DENSITY = 0.12  # expected dishes per content line (a dish spans ~1-6 lines with translations)
MIN_PRICE_TOKENS = 3  # with fewer prices on the page the recall proxy does not apply
MAX_SUSPICIOUS_NAMES = 0.2  # share of description-like / fragment names tolerated
MAX_HEADERS_PER_DISH = 0.8  # section titles per dish tolerated


@dataclass(frozen=True)
class ParseStats:
    """Counts collected by the parser for one page."""

    content_lines: int  # OCR lines left after the noise filter
    used_lines: int  # content lines used as dish name, description or section header
    price_tokens: int  # prices found on content lines
    attached_prices: int  # of those, prices that ended in a dish
    dishes: int
    priced_dishes: int
    mean_ocr_conf: float
    suspicious_names: int = 0  # dish names that read like descriptions or are fragments
    headers: int = 0  # lines taken as section titles


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
        return 0.0, {k: round(v, 3) for k, v in signals.items()}
    if stats.price_tokens >= MIN_PRICE_TOKENS:
        leftover = max(0, stats.price_tokens - stats.attached_prices)
        signals["recall_proxy"] = stats.dishes / (stats.dishes + leftover)
    suspicious = stats.suspicious_names / stats.dishes
    signals["name_gate"] = min(1.0, (1 - suspicious) / (1 - MAX_SUSPICIOUS_NAMES))
    if stats.headers:
        signals["header_gate"] = min(1.0, MAX_HEADERS_PER_DISH * stats.dishes / stats.headers)
    base = sum(weight * signals[name] for name, weight in WEIGHTS.items())
    gates = ("volume", "density", "recall_proxy", "name_gate", "header_gate")
    value = base
    for gate in gates:
        value *= signals[gate]
    return round(max(0.0, min(1.0, value)), 3), {k: round(v, 3) for k, v in signals.items()}
