"""Request cost from the per-model token prices in ``constants.MODEL_PRICES_PER_MTOK``."""

from __future__ import annotations

from menu_poc.constants import MODEL_PRICES_PER_MTOK
from menu_poc.data_classes import ModelPrice


class UnknownModelError(ValueError):
    """No price is known for the model; add it to ``MODEL_PRICES_PER_MTOK``."""


def price_for(model: str) -> ModelPrice:
    try:
        input_price, output_price = MODEL_PRICES_PER_MTOK[model]
    except KeyError:
        known = ", ".join(sorted(MODEL_PRICES_PER_MTOK))
        raise UnknownModelError(f"No price for model {model!r}. Known models: {known}") from None
    return ModelPrice(input_per_mtok=input_price, output_per_mtok=output_price)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one request. Output tokens include thinking tokens (billed as output)."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    price = price_for(model)
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
