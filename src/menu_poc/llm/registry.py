"""Model aliases: users pick ``haiku``, ``sonnet``, ``opus`` or ``ollama``; exact ids also work.

``ollama`` runs the local model named by ``OLLAMA_MODEL`` (default ``qwen2.5:7b``).
"""

from __future__ import annotations

import os

from menu_poc.llm.pricing import PRICES

OLLAMA_PREFIX = "ollama/"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

ALIASES = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}


def resolve_model(name: str) -> str:
    """Alias (case-insensitive) or exact id -> exact model id. Raises ValueError if unknown."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    if key == "ollama":
        return OLLAMA_PREFIX + os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if key.startswith(OLLAMA_PREFIX) or key in PRICES:
        return key
    choices = ", ".join([*ALIASES, "ollama"])
    raise ValueError(f"Unknown LLM model {name!r}. Use one of: {choices} (or an exact model id)")
