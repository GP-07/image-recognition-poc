"""Price detection in OCR text (Spanish / Catalan menu conventions).

Accepted: ``15,90€``, ``15,90 €``, ``€ 8.50``, ``8,95``, ``12.45``, ``3€``, ``10 EUR`` and several
prices on one line (pizza sizes). A number without a currency sign must have exactly two decimals.
Rejected: phone numbers, times (``12:30``), years, percentages, quantities with units (``1,5 l``)
and dish numbering (``46.``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_PRICE = 0.5
MAX_PRICE = 500.0

_CUR = r"(?:€|eur(?:os?)?\b)"
_NUM_DEC = r"\d{1,3}[.,]\d{2}"
_NUM_ANY = r"\d{1,3}(?:[.,]\d{1,2})?"
# Not glued to other digits, separators or times. Letters are allowed: OCR often glues the
# price to the name ("Vindaloo12,95").
_BEFORE = r"(?<![\d,:/])(?<!\d\.)"
_AFTER = r"(?![\d%:/]|[.,]\d)"
_NO_UNIT = r"(?!\s?(?:cl|ml|l|kg|gr?|cm|uds?|unid\w*|pzs?|%)\b)"
# Per-portion suffix that belongs to the price: "11,95 €/8uds".
_PORTION = r"(?:\s?/\s?\d*\s?(?:uds?|unid\w*|pzs?|pcs?|piezas?|peces|unitats?)\b)?"

_PRICE_RE = re.compile(
    rf"{_CUR}\s?(?P<a>{_NUM_ANY}){_AFTER}"  # € 8.50
    rf"|{_BEFORE}(?P<b>{_NUM_ANY})\s?{_CUR}{_PORTION}"  # 3€, 15,90 €
    rf"|{_BEFORE}(?P<c>{_NUM_DEC}){_AFTER}{_NO_UNIT}",  # 8,95
    re.IGNORECASE,
)
# Spanish phone numbers: 9 digits split by spaces/dashes (93 444 47 37) or only by dots
# (934.44.47.37); mixing dots and spaces would also match a row of prices like "9.50 12.45".
_PHONE_RE = re.compile(r"(?<!\d)(?:\+34\s?)?[6-9](?:(?:[ -]?\d){8}|(?:\.?\d){8})(?!\d)")
_PORTION_NOTE_RE = re.compile(r"\([^()]{0,8}\)")  # "(8u)", "(2p)", "(4 pzs)"
_LEADERS_RE = re.compile(r"[.…_\-·•\s]{3,}")


@dataclass(frozen=True)
class PriceMatch:
    value: float
    start: int
    end: int


def find_prices(text: str) -> list[PriceMatch]:
    """All prices in ``text`` with their character spans, left to right."""
    blocked = [m.span() for m in _PHONE_RE.finditer(text)]
    matches = []
    for m in _PRICE_RE.finditer(text):
        if any(s < m.end() and m.start() < e for s, e in blocked):
            continue
        raw = m.group("a") or m.group("b") or m.group("c")
        value = float(raw.replace(",", "."))
        if MIN_PRICE <= value <= MAX_PRICE:
            matches.append(PriceMatch(value=value, start=m.start(), end=m.end()))
    return matches


def looks_like_phone(text: str) -> bool:
    """True when ``text`` contains a Spanish phone number."""
    return bool(_PHONE_RE.search(text))


def strip_prices(text: str, matches: list[PriceMatch] | None = None) -> str:
    """``text`` without its prices, dotted leaders and dangling separators."""
    matches = find_prices(text) if matches is None else matches
    for m in sorted(matches, key=lambda m: m.start, reverse=True):
        text = text[: m.start] + " " + text[m.end :]
    text = _LEADERS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .,:;-/|€xX")


def is_price_only(text: str) -> bool:
    """True when the line holds prices and nothing else meaningful (leaders, separators)."""
    matches = find_prices(text)
    if not matches:
        return False
    rest = _PORTION_NOTE_RE.sub(" ", strip_prices(text, matches))
    return sum(ch.isalpha() for ch in rest) <= 1
