"""Price detection in OCR text (Spanish / Catalan menu conventions).

Accepted: ``15,90€``, ``15,90 €``, ``€ 8.50``, ``8,95``, ``12.45``, ``3€``, ``10 EUR`` and several
prices on one line (pizza sizes). A number without a currency sign must have exactly two decimals,
except when it is alone on its line (``..12.5``, ``8.9``): a price column cut by the photo edge or
printed with one decimal.
Rejected: phone numbers, times (``12:30``), years, percentages, quantities with units (``1,5 l``)
and dish numbering (``46.``).
"""

from __future__ import annotations

from menu_poc.constants import (
    BARE_PRICE_RE,
    MAX_PRICE,
    MIN_PRICE,
    PHONE_RE,
    PORTION_NOTE_RE,
    PRICE_GROUPS,
    PRICE_LEADERS_RE,
    PRICE_RE,
    SPACES_RE,
)
from menu_poc.data_classes import PriceMatch


def find_prices(text: str) -> list[PriceMatch]:
    """All prices in ``text`` with their character spans, left to right."""
    phones = [phone.span() for phone in PHONE_RE.finditer(text)]
    matches = []
    for found in PRICE_RE.finditer(text):
        if any(start < found.end() and found.start() < end for start, end in phones):
            continue
        raw = next(found.group(group) for group in PRICE_GROUPS if found.group(group))
        value = float(raw.replace(",", "."))
        if MIN_PRICE <= value <= MAX_PRICE:
            matches.append(PriceMatch(value=value, start=found.start(), end=found.end()))
    bare = None if matches else BARE_PRICE_RE.match(text)
    if bare is not None:
        value = float(bare.group(1).replace(",", "."))
        if MIN_PRICE <= value <= MAX_PRICE:
            matches.append(PriceMatch(value=value, start=bare.start(1), end=bare.end(1)))
    return matches


def looks_like_phone(text: str) -> bool:
    """True when ``text`` contains a Spanish phone number."""
    return bool(PHONE_RE.search(text))


def strip_prices(text: str, matches: list[PriceMatch] | None = None) -> str:
    """``text`` without its prices, dotted leaders and dangling separators."""
    matches = find_prices(text) if matches is None else matches
    for match in sorted(matches, key=lambda match: match.start, reverse=True):
        text = text[: match.start] + " " + text[match.end :]
    text = PRICE_LEADERS_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip(" .,:;-/|€xX")


def is_price_only(text: str) -> bool:
    """True when the line holds prices and nothing else meaningful (leaders, separators)."""
    matches = find_prices(text)
    if not matches:
        return False
    rest = PORTION_NOTE_RE.sub(" ", strip_prices(text, matches))
    return sum(ch.isalpha() for ch in rest) <= 1
