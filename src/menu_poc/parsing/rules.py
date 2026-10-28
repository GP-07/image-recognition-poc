"""Rule-based dish extraction anchored on prices. The cheap baseline of the cascade.

Steps for one page:

1. Classify each OCR line: noise (phones, URLs, hours, VAT notes, allergen legends, garbage),
   price-only, has inline prices, numbered (``46. LAMB CURRY``), header-like.
2. Attach every price-only line to the text line on the same row to its left (prices often sit in
   a separate column, joined by dotted leaders). Several price boxes on a row (sizes) all attach.
3. Walk the lines in reading order (column by column, top to bottom):
   - a line with prices starts a dish; a numbered line starts a dish whose price may come later;
   - a price-only line left without a row partner gives its price to the open dish if it has none
     (sushi style: name, description, then price);
   - a short line without price that is ALL CAPS or taller than usual is a section header;
   - other lines right below an open dish (same column, small gap) are its description or
     translations.
4. Clean names (numbering, leaders, prices, ``Name: ingredients``) and score the result.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from menu_poc.models import Dish, OcrLine, OcrResult, ParseResult
from menu_poc.parsing import confidence
from menu_poc.parsing.layout import group_columns, left_neighbor, median_height
from menu_poc.parsing.prices import (
    PriceMatch,
    find_prices,
    is_price_only,
    looks_like_phone,
    strip_prices,
)

VERSION = "1"
MIN_OCR_CONF = 0.6  # lines read with less confidence are treated as noise
HEADER_MAX_WORDS = 5
HEADER_MAX_CHARS = 40
HEADER_HEIGHT = 1.3  # a line this many median heights tall reads as a header
TALL_HEADER_MAX_WORDS = 3
DESC_MAX_GAP = 1.2  # max vertical gap (median heights) between a dish line and its description
DESC_MAX_LINES = 6
MAX_NAME_CHARS = 70
NUMBERED_OVERLAP = 0.3  # row overlap enough to give a price to a numbered dish name
DESCRIPTIVE_WORDS = 6  # a priced line this long reads as a description, not a name
DECORATION_REPEATS = 3  # a header text repeated this often is a logo or slogan, not a section
PRICE_COLUMN_SHARE = 0.7  # a column with this share of price-only lines is a price column
ORPHAN_LOOKBACK = 3  # recent dishes an orphan price may belong to

_NUMBERING_RE = re.compile(r"^\s*[.·•]?\s*(\d{1,3})\s*(?:[.)\-:]\s*|\s+(?=[A-ZÀ-Ý]))")
_LEADING_JUNK_RE = re.compile(r"^[\s.,:;·•\-_*|/]+")
_TRAILING_JUNK_RE = re.compile(r"[\s.,:;·•\-_*|/(]+$")
_NOISE_RE = re.compile(
    r"www\.|https?://|\S+@\S+|\.(?:com|es|cat|net)\b"  # web, email
    r"|\biva\b|i\.v\.a|impuestos|\btax(es)?\b"  # VAT notes
    r"|al[·.]?l?[eè]rg[eè]n|allergen|alérgen"  # allergen legends
    r"|whats?app|instagram|facebook|wi-?fi|domicilio|delivery|pedidos?\b"
    r"|horario|abierto|obert|opening|\bhours\b|tarjeta|efectivo|cheques?",
    re.IGNORECASE,
)
# A "name" that is only a size label belongs to a size table, not to a dish.
_SIZE_RE = re.compile(
    r"^(?:peque[ñn]as?|medianas?|grandes?|familiar|small|medium|large|petita|mitjana|gran"
    r"|individual|media|entera|1/2|tapa|raci[oó]n|ració)$",
    re.IGNORECASE,
)
_HOURS_RE = re.compile(r"\d{1,2}\s*[:h]\s*\d{2}\s*(?:-|–|a|to|h)\s*\d{1,2}\s*[:h]?\s*\d{0,2}", re.I)


@dataclass
class _Info:
    line: OcrLine
    prices: list[PriceMatch]
    price_only: bool
    noise: bool
    numbered: bool
    column: int = 0
    consumed: bool = False  # price-only line whose prices went to a row partner
    attached: list[float] = field(default_factory=list)  # prices from same-row price-only lines
    attached_ids: list[int] = field(default_factory=list)


@dataclass
class _Draft:
    head: _Info
    prices: list[float]
    line_ids: list[int]
    desc: list[str] = field(default_factory=list)
    section: str | None = None
    last: OcrLine | None = None


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def is_noise(line: OcrLine) -> bool:
    """Lines that never hold a dish: contact data, hours, legal notes, garbage."""
    text = line.text.strip()
    if line.confidence < MIN_OCR_CONF:
        return True
    if _NOISE_RE.search(text) or _HOURS_RE.search(text) or looks_like_phone(text):
        return True
    return _letters(text) < 3 and not find_prices(text)


def clean_name(text: str) -> tuple[str, str | None]:
    """Dish name from a line: drop numbering, prices, leaders; split ``Name: ingredients``."""
    text = strip_prices(text)
    text = _NUMBERING_RE.sub("", text, count=1)
    text = _LEADING_JUNK_RE.sub("", text)
    rest = None
    if ":" in text:
        head, tail = text.split(":", 1)
        if _letters(head) >= 3 and _letters(tail) >= 3:
            text, rest = head, tail.strip()
    return _TRAILING_JUNK_RE.sub("", text).strip(), rest


def _is_size_price(text: str) -> bool:
    """``"MEDIANA 13,95€"``: a size label with its price, part of a size table."""
    return bool(find_prices(text)) and bool(_SIZE_RE.match(clean_name(text)[0]))


def _descriptive(text: str) -> bool:
    """Reads like ingredients rather than a dish name."""
    name = clean_name(text)[0]
    return bool(name) and (
        name[0].islower() or "," in name or len(name.split()) >= DESCRIPTIVE_WORDS
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_header(info: _Info, med_h: float) -> bool:
    text = info.line.text.strip()
    letters = [ch for ch in text if ch.isalpha()]
    if info.prices or info.attached or info.numbered or len(letters) < 3:
        return False
    if len(text) > HEADER_MAX_CHARS or len(text.split()) > HEADER_MAX_WORDS:
        return False
    if text.endswith((",", ";", ".")):
        return False
    caps = sum(ch.isupper() for ch in letters) / len(letters)
    tall = (
        info.line.height >= HEADER_HEIGHT * med_h
        and letters[0].isupper()
        and len(text.split()) <= TALL_HEADER_MAX_WORDS  # long tilted lines get tall boxes too
    )
    return caps >= 0.8 or tall


class RuleBasedParser:
    """``Parser`` that finds dishes with price / numbering / typography heuristics."""

    def fingerprint(self) -> dict[str, Any]:
        return {"name": "rules", "version": VERSION}

    def parse(self, ocr: OcrResult) -> ParseResult:
        infos = {
            line.id: _Info(
                line=line,
                prices=find_prices(line.text),
                price_only=is_price_only(line.text) or _is_size_price(line.text),
                noise=is_noise(line),
                numbered=bool(_NUMBERING_RE.match(line.text)) and _letters(line.text) >= 3,
            )
            for line in ocr.lines
        }
        content = [i for i in infos.values() if not i.noise]
        med_h = median_height([i.line for i in content]) or 1.0
        price_tokens = sum(len(i.prices) for i in content)

        columns = group_columns([i.line for i in content], ocr.width)
        for index, col in enumerate(columns):
            for line in col:
                infos[line.id].column = index
        self._attach_row_prices(content, columns)
        order = [infos[line.id] for col in columns for line in col]

        drafts, used, orphan_prices = self._walk(order, med_h)
        dishes = [d for d in (self._finish(draft) for draft in drafts) if d is not None]

        attached = sum(len(d.prices) for d in dishes)
        stats = confidence.ParseStats(
            content_lines=len(content),
            used_lines=len(used),
            price_tokens=price_tokens,
            attached_prices=min(attached, price_tokens),
            dishes=len(dishes),
            priced_dishes=sum(1 for d in dishes if d.prices),
            mean_ocr_conf=mean(i.line.confidence for i in content) if content else 0.0,
        )
        value, signals = confidence.score(stats)
        signals["orphan_prices"] = float(orphan_prices)
        return ParseResult(dishes=dishes, confidence=value, signals=signals)

    @staticmethod
    def _attach_row_prices(content: list[_Info], columns: list[list[OcrLine]]) -> None:
        """Give each price-only line's prices to the text line on its row, to the left.

        The partner must be in the price's column, or in the column just left of it when the
        price's column is a column of prices (names and prices split by a wide gap).
        """
        by_id = {i.line.id: i for i in content}
        price_columns = {
            index
            for index, col in enumerate(columns)
            if sum(by_id[ln.id].price_only for ln in col) >= PRICE_COLUMN_SHARE * len(col)
        }
        texts = [i.line for i in content if not i.price_only and _letters(i.line.text) >= 3]
        for info in sorted(content, key=lambda i: i.line.x0):
            if not info.price_only:
                continue
            allowed = {info.column}
            if info.column in price_columns:
                allowed.add(info.column - 1)
            near = [ln for ln in texts if by_id[ln.id].column in allowed]
            numbered = [ln for ln in near if by_id[ln.id].numbered]
            # A numbered name partly on the row beats a description fully on it (tilted photos).
            target = left_neighbor(
                info.line, numbered, NUMBERED_OVERLAP, obstacles=texts
            ) or left_neighbor(info.line, near, obstacles=texts)
            if target is None:
                continue
            owner = by_id[target.id]
            owner.attached.extend(p.value for p in info.prices)
            owner.attached_ids.append(info.line.id)
            info.consumed = True

    def _walk(self, order: list[_Info], med_h: float) -> tuple[list[_Draft], set[int], int]:
        drafts: list[_Draft] = []
        used: set[int] = set()
        section: str | None = None
        current: _Draft | None = None
        orphan_prices = 0
        pending: _Draft | None = None  # last header, in case it turns out to be a dish name
        decorations = Counter(
            _norm(clean_name(i.line.text)[0]) for i in order if _is_header(i, med_h)
        )

        for info in order:
            line = info.line
            if info.consumed:
                continue
            all_prices = [p.value for p in info.prices] + info.attached
            if info.price_only:
                # Price under its dish (name, description, then price): a recent priceless dish.
                owner = next(
                    (
                        d
                        for d in reversed(drafts[-ORPHAN_LOOKBACK:])
                        if not d.prices and self._near(d, info, med_h)
                    ),
                    None,
                )
                if owner is not None:
                    owner.prices = all_prices
                    owner.line_ids.append(line.id)
                    owner.last = line
                    used.add(line.id)
                else:
                    orphan_prices += len(all_prices)
                continue
            if (
                all_prices
                and not info.numbered
                and pending is not None
                and self._near(pending, info, med_h)
                and _descriptive(line.text)
            ):
                # "SHEPPERD" (looked like a header) then its ingredients carrying the price.
                current, section, pending = pending, pending.section, None
                drafts.append(current)
                used.update(current.line_ids)
            if (
                all_prices
                and not info.numbered
                and current is not None
                and not current.prices
                and self._near(current, info, med_h)
            ):
                # "46. LAMB CURRY" then its translation carrying the price: same dish.
                current.prices = all_prices
                current.desc.append(clean_name(line.text)[0])
                current.line_ids.extend([line.id, *info.attached_ids])
                current.last = line
                used.add(line.id)
                continue
            pending = None if (all_prices or info.numbered) else pending
            if all_prices or info.numbered:
                current = _Draft(
                    head=info,
                    prices=all_prices,
                    line_ids=[line.id, *info.attached_ids],
                    section=section,
                    last=line,
                )
                drafts.append(current)
                used.add(line.id)
                continue
            if _is_header(info, med_h):
                title = clean_name(line.text)[0]
                if title and decorations[_norm(title)] < DECORATION_REPEATS:
                    pending = _Draft(head=info, prices=[], line_ids=[line.id], section=section)
                    pending.last = line
                    section = title
                current = None
                used.add(line.id)
                continue
            if pending is not None and self._near(pending, info, med_h):
                if len(pending.desc) < DESC_MAX_LINES:
                    pending.desc.append(line.text.strip())
                    pending.line_ids.append(line.id)
                    pending.last = line
                continue
            pending = None
            if current is not None and self._near(current, info, med_h):
                if len(current.desc) < DESC_MAX_LINES:
                    current.desc.append(line.text.strip())
                    current.line_ids.append(line.id)
                    current.last = line
                    used.add(line.id)
                continue
            current = None  # unrelated text closes the open dish
        return drafts, used, orphan_prices

    @staticmethod
    def _near(draft: _Draft, info: _Info, med_h: float) -> bool:
        """``info`` sits right below the draft's last line, horizontally within its block."""
        head, last = draft.head.line, draft.last or draft.head.line
        gap = info.line.y0 - last.y1
        x0, x1 = min(head.x0, last.x0), max(head.x1, last.x1)
        overlaps = min(x1, info.line.x1) > max(x0, info.line.x0)
        return (
            info.column == draft.head.column and -med_h <= gap <= DESC_MAX_GAP * med_h and overlaps
        )

    @staticmethod
    def _finish(draft: _Draft) -> Dish | None:
        name, rest = clean_name(draft.head.line.text)
        if _letters(name) < 3 or len(name) > MAX_NAME_CHARS:
            return None
        parts = ([rest] if rest else []) + draft.desc
        description = " ".join(p for p in parts if p) or None
        return Dish(
            name=name,
            description=description,
            prices=draft.prices,
            section=draft.section,
            source_line_ids=sorted(set(draft.line_ids)),
            origin="rules",
        )
