"""Model aliases: users pick ``haiku``, ``sonnet``, ``opus`` or ``ollama``; exact ids also work.

Ids starting with ``ollama/`` run a local model through Ollama (``ollama/<model>`` picks another
local model than the default).
"""

from __future__ import annotations

from menu_poc.llm.pricing import PRICES

OLLAMA_PREFIX = "ollama/"

ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "ollama": f"{OLLAMA_PREFIX}qwen2.5:7b",
}


def resolve_model(name: str) -> str:
    """Alias (case-insensitive) or exact id -> exact model id. Raises ValueError if unknown."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    if key.startswith(OLLAMA_PREFIX) or key in PRICES:
        return key
    raise ValueError(
        f"Unknown LLM model {name!r}. Use one of: {', '.join(ALIASES)} (or an exact model id)"
    )
