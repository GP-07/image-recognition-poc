"""Rule-based dish extraction anchored on prices. The cheap baseline of the cascade.

Steps for one page:

1. Classify each OCR line: noise (phones, URLs, hours, VAT notes, allergen legends and diet
   codes, garbage), price-only (also ``8,00 (2p) por encargo``), has inline prices, enumerated
   (``46. LAMB CURRY``, ``· MARGARITA``), header-like.
2. Attach every price-only line to the text line on the same row to its left (prices often sit in
   a separate column, joined by dotted leaders). Several price boxes on a row (sizes) all attach.
3. Walk the lines in reading order (column by column, top to bottom):
   - a line with prices starts a dish; a numbered line starts a dish whose price may come later;
   - a price-only line left without a row partner gives its price to the open dish if it has none
     (sushi style: name, description, then price);
   - a short line without price that is ALL CAPS or taller than usual is a section header;
   - other lines right below an open dish (same column, small gap) are its description or
     translations, except an ALL CAPS line at the dish's indent (a sibling whose price was lost);
   - a lowercase line with a price continues the open dish's description: its price (tilted row)
     goes to the next name line.
4. Clean names (numbering, leaders, prices, ``Name: ingredients``) and score the result.
   Every orderable item is kept: dishes, desserts, beverages, add-ons.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from menu_poc.constants import (
    BULLET_RE,
    BULLETS,
    CAPS_SHARE,
    CONTINUATION_MIN_WORDS,
    DECORATION_REPEATS,
    DESC_MAX_GAP,
    DESC_MAX_LINES,
    DESCRIPTIVE_WORDS,
    DIET_CODE_RE,
    DISH_NUMBERING_RE,
    FRAGMENT_LETTERS,
    HEADER_HEIGHT,
    HEADER_MAX_CHARS,
    HEADER_MAX_WORDS,
    HOURS_RE,
    LEADING_JUNK_RE,
    MAX_NAME_CHARS,
    MIN_LETTERS,
    MIN_OCR_CONF,
    NOISE_RE,
    NON_ALNUM_ASCII_RE,
    NUMBERED_OVERLAP,
    ORPHAN_LOOKBACK,
    PRICE_COLUMN_SHARE,
    RULES_VERSION,
    SIZE_RE,
    TALL_HEADER_MAX_WORDS,
    TRAILING_JUNK_RE,
)
from menu_poc.data_classes import DishDraft, LineInfo, ParseStats
from menu_poc.models import Dish, OcrLine, OcrResult, ParseResult
from menu_poc.parsing import confidence
from menu_poc.parsing.layout import group_columns, left_neighbor, median_height
from menu_poc.parsing.prices import find_prices, is_price_only, looks_like_phone, strip_prices


def _letters(text: str) -> int:
    return sum(char.isalpha() for char in text)


def is_noise(line: OcrLine) -> bool:
    """Lines that never hold a dish: contact data, hours, legal notes, garbage."""
    text = line.text.strip()
    if line.confidence < MIN_OCR_CONF:
        return True
    if NOISE_RE.search(text) or HOURS_RE.search(text) or looks_like_phone(text):
        return True
    if DIET_CODE_RE.match(text):
        return True
    return _letters(text) < MIN_LETTERS and not find_prices(text)


def clean_name(text: str) -> tuple[str, str | None]:
    """Dish name from a line: drop numbering, prices, leaders; split ``Name: ingredients``."""
    text = strip_prices(text)
    text = DISH_NUMBERING_RE.sub("", text, count=1)
    text = LEADING_JUNK_RE.sub("", text)
    rest = None
    if ":" in text:
        head, tail = text.split(":", 1)
        if _letters(head) >= MIN_LETTERS and _letters(tail) >= MIN_LETTERS:
            text, rest = head, tail.strip()
    return TRAILING_JUNK_RE.sub("", text).strip(), rest


def _is_size_price(text: str) -> bool:
    """``"MEDIANA 13,95€"``: a size label with its price, part of a size table."""
    return bool(find_prices(text)) and bool(SIZE_RE.match(clean_name(text)[0]))


def _is_note(text: str) -> bool:
    """``"(2 uds.)"``, ``"8,00 (2p) por encargo"``: a portion or ordering note, never a name."""
    return clean_name(text)[0].startswith("(")


def _is_enumerated(text: str) -> bool:
    """Numbered (``46. LAMB CURRY``) or bulleted (``· MARGARITA``) list item."""
    if _letters(text) < MIN_LETTERS:
        return False
    bulleted = bool(BULLET_RE.match(text)) and not text.rstrip().endswith(tuple(BULLETS))
    return bulleted or bool(DISH_NUMBERING_RE.match(text))


def _continuation(text: str) -> bool:
    """A sentence that started on the line above: lowercase start and several words."""
    words = text.split()
    return len(words) >= CONTINUATION_MIN_WORDS and words[0][0].islower()


def _name_like(text: str) -> bool:
    """Short line that starts with a capital letter and does not read as ingredients."""
    name = clean_name(text)[0]
    return bool(name) and name[0].isupper() and not _descriptive(text)


def _descriptive(text: str) -> bool:
    """Reads like ingredients rather than a dish name."""
    name = clean_name(text)[0]
    return bool(name) and (
        name[0].islower() or "," in name or len(name.split()) >= DESCRIPTIVE_WORDS
    )


def _suspicious_name(name: str) -> bool:
    """A dish name that reads like a description, or a fragment of a few letters."""
    return _letters(name) <= FRAGMENT_LETTERS or name[0].islower() or _descriptive(name)


def _norm(text: str) -> str:
    return NON_ALNUM_ASCII_RE.sub("", text.lower())


def _caps_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def _is_header(info: LineInfo, med_h: float) -> bool:
    text = info.line.text.strip()
    letters = [char for char in text if char.isalpha()]
    if info.prices or info.attached or info.numbered or len(letters) < MIN_LETTERS:
        return False
    if len(text) > HEADER_MAX_CHARS or len(text.split()) > HEADER_MAX_WORDS:
        return False
    if text.endswith((",", ";", ".")):
        return False
    caps = _caps_ratio(text)
    tall = (
        info.line.height >= HEADER_HEIGHT * med_h
        and letters[0].isupper()
        and len(text.split()) <= TALL_HEADER_MAX_WORDS  # long tilted lines get tall boxes too
    )
    return caps >= CAPS_SHARE or tall


class RuleBasedParser:
    """``Parser`` that finds dishes with price / numbering / typography heuristics."""

    def fingerprint(self) -> dict[str, Any]:
        return {"name": "rules", "version": RULES_VERSION}

    def parse(self, ocr: OcrResult) -> ParseResult:
        infos = {
            line.id: LineInfo(
                line=line,
                prices=find_prices(line.text),
                price_only=(
                    is_price_only(line.text)
                    or _is_size_price(line.text)
                    or (_is_note(line.text) and bool(find_prices(line.text)))
                ),
                noise=is_noise(line),
                numbered=_is_enumerated(line.text),
            )
            for line in ocr.lines
        }
        content = [info for info in infos.values() if not info.noise]
        med_h = median_height([info.line for info in content]) or 1.0
        price_tokens = sum(len(info.prices) for info in content)

        columns = group_columns([info.line for info in content], ocr.width)
        for index, column in enumerate(columns):
            for line in column:
                infos[line.id].column = index
        self._attach_row_prices(content, columns)
        order = [infos[line.id] for column in columns for line in column]

        drafts, used, orphan_prices, headers = self._walk(order, med_h)
        finished = (self._finish(draft) for draft in drafts)
        dishes = [dish for dish in finished if dish is not None]

        attached = sum(len(dish.prices) for dish in dishes)
        stats = ParseStats(
            content_lines=len(content),
            used_lines=len(used),
            price_tokens=price_tokens,
            attached_prices=min(attached, price_tokens),
            dishes=len(dishes),
            priced_dishes=sum(1 for dish in dishes if dish.prices),
            mean_ocr_conf=mean(info.line.confidence for info in content) if content else 0.0,
            suspicious_names=sum(_suspicious_name(dish.name) for dish in dishes),
            headers=headers,
        )
        value, signals = confidence.score(stats)
        signals["orphan_prices"] = float(orphan_prices)
        return ParseResult(dishes=dishes, confidence=value, signals=signals)

    @staticmethod
    def _attach_row_prices(content: list[LineInfo], columns: list[list[OcrLine]]) -> None:
        """Give each price-only line's prices to the text line on its row, to the left.

        The partner must be in the price's column, or in the column just left of it when the
        price's column is a column of prices (names and prices split by a wide gap).
        """
        by_id = {info.line.id: info for info in content}
        price_columns = {
            index
            for index, column in enumerate(columns)
            if sum(by_id[line.id].price_only for line in column) >= PRICE_COLUMN_SHARE * len(column)
        }
        texts = [
            info.line
            for info in content
            if not info.price_only
            and _letters(info.line.text) >= MIN_LETTERS
            and not _is_note(info.line.text)
        ]
        for info in sorted(content, key=lambda info: info.line.x0):
            if not info.price_only:
                continue
            allowed = {info.column}
            if info.column in price_columns:
                allowed.add(info.column - 1)
            near = [line for line in texts if by_id[line.id].column in allowed]
            numbered = [line for line in near if by_id[line.id].numbered]
            # A numbered name partly on the row beats a description fully on it (tilted photos).
            target = left_neighbor(
                info.line, numbered, NUMBERED_OVERLAP, obstacles=texts
            ) or left_neighbor(info.line, near, obstacles=texts)
            if target is None:
                continue
            owner = by_id[target.id]
            owner.attached.extend(price.value for price in info.prices)
            owner.attached_ids.append(info.line.id)
            info.consumed = True

    def _walk(
        self, order: list[LineInfo], med_h: float
    ) -> tuple[list[DishDraft], set[int], int, int]:
        drafts: list[DishDraft] = []
        headers = 0
        used: set[int] = set()
        section: str | None = None
        current: DishDraft | None = None
        orphan_prices = 0
        pending: DishDraft | None = None  # last header, in case it turns out to be a dish name
        carry: list[float] = []  # price read on a description's last line, for the next name
        decorations = Counter(
            _norm(clean_name(info.line.text)[0]) for info in order if _is_header(info, med_h)
        )

        for info in order:
            line = info.line
            if info.consumed:
                continue
            all_prices = [price.value for price in info.prices] + info.attached
            carried, carry = carry, []
            if info.price_only:
                # Price under its dish (name, description, then price): a recent priceless dish.
                owner = next(
                    (
                        draft
                        for draft in reversed(drafts[-ORPHAN_LOOKBACK:])
                        if not draft.prices and self._near(draft, info, med_h)
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
                and current.prices
                and self._near(current, info, med_h)
                and _continuation(line.text)
            ):
                # "... salsa de tomate y especias. 10,00€": the description of a priced dish
                # ends on the row of the next dish's price (tilted photo), so the price waits
                # for the next name.
                current.desc.append(line.text.strip())
                current.line_ids.append(line.id)
                current.last = line
                used.add(line.id)
                carry = all_prices
                continue
            if carried and not all_prices and not info.numbered and _name_like(line.text):
                all_prices = carried
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
                current = DishDraft(
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
                headers += 1
                title = clean_name(line.text)[0]
                if title and decorations[_norm(title)] < DECORATION_REPEATS:
                    pending = DishDraft(head=info, prices=[], line_ids=[line.id], section=section)
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
            if current is not None and current.desc and self._sibling(current, info, med_h):
                # "TORI KAISU.." after "EBI TEMPURA / Langostinos rebozados": styled like the
                # dish name above and not a title (ends in leaders), so a new dish whose price
                # was lost (cut by the photo edge, unreadable).
                current = DishDraft(head=info, prices=[], line_ids=[line.id], section=section)
                current.last = line
                drafts.append(current)
                used.add(line.id)
                continue
            if current is not None and self._near(current, info, med_h):
                if len(current.desc) < DESC_MAX_LINES:
                    current.desc.append(line.text.strip())
                    current.line_ids.append(line.id)
                    current.last = line
                    used.add(line.id)
                continue
            current = None  # unrelated text closes the open dish
        return drafts, used, orphan_prices, headers

    @staticmethod
    def _near(draft: DishDraft, info: LineInfo, med_h: float) -> bool:
        """``info`` sits right below the draft's last line, horizontally within its block."""
        head, last = draft.head.line, draft.last or draft.head.line
        gap = info.line.y0 - last.y1
        x0, x1 = min(head.x0, last.x0), max(head.x1, last.x1)
        overlaps = min(x1, info.line.x1) > max(x0, info.line.x0)
        return (
            info.column == draft.head.column and -med_h <= gap <= DESC_MAX_GAP * med_h and overlaps
        )

    @staticmethod
    def _sibling(draft: DishDraft, info: LineInfo, med_h: float) -> bool:
        """``info`` looks like the open dish's name: short ALL CAPS line at the same indent."""
        head, text = draft.head.line, info.line.text
        return (
            info.column == draft.head.column
            and abs(info.line.x0 - head.x0) <= med_h
            and len(text.split()) <= HEADER_MAX_WORDS
            and _caps_ratio(text) >= CAPS_SHARE
            and _caps_ratio(head.text) >= CAPS_SHARE
        )

    @staticmethod
    def _finish(draft: DishDraft) -> Dish | None:
        name, rest = clean_name(draft.head.line.text)
        if _letters(name) < MIN_LETTERS or len(name) > MAX_NAME_CHARS:
            return None
        parts = ([rest] if rest else []) + draft.desc
        description = " ".join(part for part in parts if part) or None
        return Dish(
            name=name,
            description=description,
            prices=draft.prices,
            section=draft.section,
            source_line_ids=sorted(set(draft.line_ids)),
            origin="rules",
        )
