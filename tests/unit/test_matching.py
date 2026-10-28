import pytest

from menu_poc.evaluation.golden import GoldenDish
from menu_poc.evaluation.matching import match_dishes, normalize, similarity

GOLD = [
    GoldenDish(name="Lamb Curry", aliases=["Xai al curry tradicional", "Cordero al curry"]),
    GoldenDish(name="Patatas Bravas"),
    GoldenDish(name="Tiramisú"),
]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("46. LAMB CURRY", "lamb curry"),
        ("12) Cordero al curry", "cordero al curry"),
        ("Tiramisú ........ 5,50 €", "tiramisu"),
        ("Patatas bravas 4.50", "patatas bravas"),
        ("  Crème   brûlée!! ", "creme brulee"),
        ("Pizza 4 quesos", "pizza 4 quesos"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_similarity_rewards_extra_words_but_not_truncation():
    assert similarity("lamb curry tradicional", "lamb curry") == 100
    assert similarity("curry", "lamb curry") < 85


def test_alias_match():
    m = match_dishes(["Cordero al curry"], GOLD)
    assert (m.tp, m.fp, m.fn) == (1, 0, 2)
    assert m.matched[0].gold == "Lamb Curry"
    assert m.matched[0].variant == "Cordero al curry"


def test_translation_is_duplicate_not_false_positive():
    m = match_dishes(["46. LAMB CURRY", "Xai al curry tradicional", "Cordero al curry"], GOLD)
    assert (m.tp, m.fp, m.duplicates) == (1, 0, 2)
    assert {p.gold for p in m.duplicate_pairs} == {"Lamb Curry"}


def test_ocr_noisy_name_still_matches():
    m = match_dishes(["Patatas Brabas", "Tiramisu ..... 5,50€", "Lamb Cury"], GOLD)
    assert (m.tp, m.fp, m.fn) == (3, 0, 0)


def test_unrelated_name_is_false_positive():
    m = match_dishes(["Ensalada mixta", "Telefono 93 123 45 67"], GOLD)
    assert (m.tp, m.fp, m.fn) == (0, 2, 3)
    assert m.false_positives == ["Ensalada mixta", "Telefono 93 123 45 67"]
    assert m.unmatched_gold == ["Lamb Curry", "Patatas Bravas", "Tiramisú"]


def test_one_to_one_greedy_by_best_score():
    gold = [GoldenDish(name="Chicken Curry"), GoldenDish(name="Lamb Curry")]
    m = match_dishes(["Lamb Curry", "Chicken Curry"], gold)
    assert {(p.pred, p.gold) for p in m.matched} == {
        ("Lamb Curry", "Lamb Curry"),
        ("Chicken Curry", "Chicken Curry"),
    }


def test_empty_inputs():
    assert match_dishes([], GOLD).fn == 3
    assert match_dishes(["x"], []).fp == 1
