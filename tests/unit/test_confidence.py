from menu_poc.parsing.confidence import ParseStats, score


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
    values = [score(_stats(attached_prices=n))[0] for n in range(0, 21, 5)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)
