import pytest

from menu_poc.parsing.prices import find_prices, is_price_only, looks_like_phone, strip_prices


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("15,90€", [15.9]),
        ("15,90 €", [15.9]),
        ("€ 8.50", [8.5]),
        ("8,95", [8.95]),
        ("12.45", [12.45]),
        ("3€", [3.0]),
        ("10 EUR", [10.0]),
        ("8,15€ 9,50€ 12,45€", [8.15, 9.5, 12.45]),
        ("8.15 9.50 12.45", [8.15, 9.5, 12.45]),
        ("Pollo Tikka..........8,50€", [8.5]),
        ("Cordero Vindaloo12,95", [12.95]),
        ("4 unitats x 3,90€ Amb formatge x 4,50€", [3.9, 4.5]),
        ("11,95 €/8uds", [11.95]),
        ("..12.5", [12.5]),  # price column cut by the photo edge
        ("8.9", [8.9]),
    ],
)
def test_finds_prices(text, expected):
    assert [m.value for m in find_prices(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Tel 93 444 47 37",
        "934.44.47.37",
        "4145252",
        "12:30",
        "De 19:30 a 23:00",
        "2024",
        "10%",
        "46. LAMB CURRY",
        "1,5 l",
        "Botella 0,75 cl",
        "Ref 1.25.50",
        "6.",
        "12",
        "Mix 1.5 kg",
    ],
)
def test_rejects_non_prices(text):
    assert find_prices(text) == []


def test_span_allows_stripping_the_price_from_the_name():
    text = "Pollo Tikka ....... 8,50 €"
    (match,) = find_prices(text)
    assert text[match.start : match.end].startswith("8,50")
    assert strip_prices(text) == "Pollo Tikka"


def test_price_only_lines():
    assert is_price_only("8,15€ 9,50€")
    assert is_price_only("4.00 (8u)")
    assert is_price_only("11,95 €/8uds")
    assert is_price_only("...13.5")
    assert not is_price_only("Margarita 8,15€")
    assert not is_price_only("no price here")


def test_phone_detection():
    assert looks_like_phone("Pedidos 93 250 92 16")
    assert not looks_like_phone("9,50 12,45 15,00")
