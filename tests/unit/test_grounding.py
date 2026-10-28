from __future__ import annotations

import pytest

from menu_poc.llm.grounding import GroundingVerifier, match_score
from menu_poc.llm.text import normalize
from menu_poc.models import Dish, OcrLine, OcrResult


def _ocr(*rows: tuple[int, str, float, float]) -> OcrResult:
    """Rows of (id, text, x0, y0); every line is 300 px wide and 30 px tall."""
    lines = [
        OcrLine(id=line_id, text=text, bbox=(x0, y0, x0 + 300, y0 + 30), confidence=0.9)
        for line_id, text, x0, y0 in rows
    ]
    return OcrResult(image_id="r/i", width=1000, height=1000, engine="t", lines=lines)


def _dish(name: str, ids: list[int] | None = None) -> Dish:
    return Dish(name=name, source_line_ids=ids or [], origin="llm")


OCR = _ocr(
    (0, "PIZZAS", 50, 50),
    (1, "12. Pizza Quattro Formaggi", 50, 100),
    (2, "Mozzarella, gorgonzola, parmesano", 50, 135),
    (3, "Coppa Tiramisü", 50, 200),
    (4, "Pollo Tikka", 50, 300),
    (5, "Masala", 50, 335),
    (6, "Cheescake de la casa .... 4,50€", 500, 100),
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12. Pizza Quattro Formaggi", "pizza quattro formaggi"),
        ("46 LAMB CURRY.", "lamb curry"),
        ("a) Crème brûlée!", "creme brulee"),
        ("Pizza 4 Quesos", "pizza 4 quesos"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_keeps_verbatim_name_with_its_ids():
    result = GroundingVerifier().verify([_dish("Pizza Quattro Formaggi", [1, 2])], OCR)
    assert [dish.name for dish in result.kept] == ["Pizza Quattro Formaggi"]
    assert result.kept[0].source_line_ids == [1, 2]


def test_keeps_ocr_noisy_or_lightly_corrected_names():
    dishes = [_dish("Coppa Tiramisù", [3]), _dish("Cheesecake de la casa", [6])]
    result = GroundingVerifier().verify(dishes, OCR)
    assert len(result.kept) == 2 and not result.dropped


def test_drops_invented_dish():
    dishes = [_dish("Lasagna alla Bolognese", [1]), _dish("Spaghetti Carbonara")]
    result = GroundingVerifier().verify(dishes, OCR)
    assert not result.kept and len(result.dropped) == 2


def test_longer_invented_name_does_not_match_a_short_line():
    # "Pollo Tikka" is in the OCR, but the extra words are not.
    result = GroundingVerifier().verify([_dish("Pollo Tikka con arroz basmati y naan", [4])], OCR)
    assert result.dropped and not result.kept


def test_unknown_ids_removed_and_fallback_to_page():
    result = GroundingVerifier().verify([_dish("Coppa Tiramisu", [99, 3])], OCR)
    assert result.kept[0].source_line_ids == [3]

    result = GroundingVerifier().verify([_dish("Coppa Tiramisu", [99])], OCR)
    assert result.kept[0].source_line_ids == [3]


def test_wrong_ids_fall_back_to_page_search():
    result = GroundingVerifier().verify([_dish("Coppa Tiramisu", [0])], OCR)
    assert result.kept[0].source_line_ids == [3]


def test_name_split_over_two_lines_matches_adjacent_lines():
    result = GroundingVerifier().verify([_dish("Pollo Tikka Masala")], OCR)
    assert result.kept and set(result.kept[0].source_line_ids) == {4, 5}


def test_short_names_need_a_whole_word():
    assert match_score("te", "tomate fresco") == 0
    assert match_score("te", "cafe o te") == 100


def test_empty_name_is_dropped():
    result = GroundingVerifier().verify([_dish("12.", [1])], OCR)
    assert result.dropped


def test_threshold_is_configurable():
    dish = _dish("Coppa Tiramisu della Nonna", [3])
    assert GroundingVerifier(threshold=50).verify([dish], OCR).kept
    assert GroundingVerifier(threshold=85).verify([dish], OCR).dropped


def test_shared_prefix_is_not_enough():
    ocr = _ocr((0, "Croquetas de sémola rellenas de carne picada", 50, 50))
    result = GroundingVerifier().verify([_dish("Croquetas de jamón", [0])], ocr)
    assert result.dropped and not result.kept
