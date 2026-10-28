import pytest

from menu_poc.constants import MODEL_PRICES_PER_MTOK
from menu_poc.llm.pricing import UnknownModelError, cost_usd


def test_haiku_cost_per_million():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)


def test_typical_page_cost():
    # ~2K input + ~2K output tokens
    assert cost_usd("claude-haiku-5-5", 2000, 2000) == pytest.approx(0.0012)
    assert cost_usd("claude-sonnet-5-5", 2000, 2000) == pytest.approx(0.024)
    assert cost_usd("claude-opus-5-5", 2000, 2000) == pytest.approx(0.048)


def test_known_models():
    assert {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"} <= set(
        MODEL_PRICES_PER_MTOK
    )


def test_unknown_model_is_a_clear_error():
    with pytest.raises(UnknownModelError, match="gpt-x.*Known models"):
        cost_usd("gpt-x", 10, 10)


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        cost_usd("claude-haiku-5-5", -1, 0)
