"""Per-model token prices (USD per million tokens) and request cost.

Prices are the standard rates for prompts up to 100K input tokens (a menu page is ~3K).
Batch API (-50%) and prompt-caching discounts are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
    "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-opus-5-5": ModelPrice(input_per_mtok=4.00, output_per_mtok=20.00),
}


class UnknownModelError(ValueError):
    """No price is known for the model; add it to ``PRICES``."""


def price_for(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES))
        raise UnknownModelError(f"No price for model {model!r}. Known models: {known}") from None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one request. Output tokens include thinking tokens (billed as output)."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    price = price_for(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
