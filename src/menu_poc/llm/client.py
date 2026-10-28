"""LLM client interface, its errors and the factory.

One implementation per file:
- ``anthropic_client.AnthropicClient``: Claude (Messages API, structured outputs).
- ``ollama_client.OllamaClient``: free local models through Ollama.
- ``fake_client.FakeLLMClient``: canned answers for tests and offline demos.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from menu_poc.llm.registry import OLLAMA_PREFIX, resolve_model
from menu_poc.models import LLMUsage


class LLMClient(Protocol):
    model: str

    def fingerprint(self) -> dict[str, Any]:
        """Settings that change the answer (part of the LLM cache key)."""
        ...

    def unavailable_reason(self) -> str | None:
        """None when the client can run (valid key, server up), else a human-readable reason."""
        ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMUsage]:
        """Return a JSON object that validates against ``schema`` plus usage/cost."""
        ...


class LLMError(RuntimeError):
    """The LLM call failed and retrying the same request will not help."""


class LLMRetryableError(LLMError):
    """Transient failure (rate limit, overload, 5xx, network) left after the client's retries."""


class LLMOutputError(LLMError):
    """The model answered (and may have billed) but the output is unusable (refusal, truncation,
    invalid JSON); ``usage`` keeps the cost."""

    def __init__(self, message: str, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def prompt_key(system: str, user: str) -> str:
    """Stable key of a prompt pair, used to look up recorded answers."""
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


def make_llm_client(model: str, fake_responses: Any | None = None) -> LLMClient:
    """Client for ``model`` (alias or exact id): ``ollama/...`` -> local Ollama, otherwise Claude.

    With ``fake_responses`` a fake named ``fake/<model>``, so its answers never share cache keys
    with the real model.
    """
    from menu_poc.llm.anthropic_client import AnthropicClient
    from menu_poc.llm.fake_client import FakeLLMClient
    from menu_poc.llm.ollama_client import OllamaClient

    model = resolve_model(model)
    if fake_responses is not None:
        return FakeLLMClient(fake_responses, model=f"fake/{model}")
    if model.startswith(OLLAMA_PREFIX):
        return OllamaClient(model)
    return AnthropicClient(model)
