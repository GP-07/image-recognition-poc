from menu_poc.data_classes import ParseStats
from menu_poc.parsing.confidence import score


def _stats(**overrides):
    base = dict(
        content_lines=60,
        used_lines=50,
        price_tokens=20,
        attached_prices=20,
        dishes=20,
        priced_dishes=20,
        mean_ocr_conf=0.95,
    )
    return ParseStats(**{**base, **overrides})


def test_no_dishes_means_zero():
    value, signals = score(_stats(dishes=0, priced_dishes=0, attached_prices=0))
    assert value == 0.0
    assert "price_attach" in signals


def test_clean_page_scores_high():
    value, _ = score(_stats())
    assert value >= 0.9


def test_unattached_prices_push_below_threshold():
    value, signals = score(_stats(attached_prices=8, priced_dishes=8))
    assert signals["price_attach"] == 0.4
    assert value < 0.6


def test_few_dishes_on_a_busy_page_scores_low():
    value, signals = score(_stats(content_lines=200, dishes=3, priced_dishes=3, attached_prices=3))
    assert signals["density"] < 0.2
    assert value < 0.3


def test_score_is_bounded_and_monotonic_in_attachment():
    values = [score(_stats(attached_prices=attached))[0] for attached in range(0, 21, 5)]
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_leftover_prices_estimate_missed_dishes():
    value, signals = score(_stats(price_tokens=25))  # 5 prices left: ~5 dishes missed
    assert signals["recall_proxy"] == 0.8
    assert value < score(_stats())[0] * 0.81


def test_fragment_or_description_names_cap_the_score():
    clean, _ = score(_stats(suspicious_names=2))  # 10%: tolerated
    broken, signals = score(_stats(suspicious_names=10))  # half the names
    assert clean >= 0.9
    assert signals["name_gate"] < 0.7
    assert broken < 0.6


def test_more_titles_than_dishes_caps_the_score():
    ok, _ = score(_stats(headers=8))
    value, signals = score(_stats(headers=40))  # caps dish names taken for titles
    assert ok >= 0.9
    assert signals["header_gate"] == 0.4
    assert value < 0.6
