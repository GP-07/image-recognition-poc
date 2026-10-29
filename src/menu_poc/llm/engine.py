"""LLM engines: one base class, one subclass per provider, chosen with ``LLMEngine.get_for_alias``.

Each subclass says whether it can handle a model name (``handles``) and which aliases it owns, so
adding a provider is adding a subclass; nothing else in the code base has to change:

- ``anthropic_engine.AnthropicEngine``: Claude (``haiku``, ``sonnet``, ``opus`` or an exact id).
- ``ollama_engine.OllamaEngine``: free local models (``ollama`` or ``ollama/<model>``).
- ``fake_engine.FakeEngine``: canned answers for tests; never selected by name.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from menu_poc.errors import TransientError
from menu_poc.models import LLMUsage


class LLMError(RuntimeError):
    """The LLM call failed and retrying the same request will not help."""


class LLMRetryableError(LLMError, TransientError):
    """Transient failure (rate limit, overload, 5xx, network) left after the engine's retries."""


class LLMOutputError(LLMError):
    """The model answered (and may have billed) but the output is unusable (refusal, truncation,
    invalid JSON); ``usage`` keeps the cost."""

    def __init__(self, message: str, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class UnknownModelError(ValueError):
    """No engine handles the requested model name."""


class LLMEngine(ABC):
    aliases: ClassVar[dict[str, str]] = {}  # alias -> exact model id, per subclass

    def __init__(self, model: str) -> None:
        self.model = model

    @classmethod
    def get_for_alias(cls, name: str) -> LLMEngine:
        """Instance of the engine that handles ``name`` (alias in any case, or exact model id).

        Building an engine is cheap (no network: clients connect on first use), so this is also
        how the configuration validates and resolves a model name.
        """
        key = name.strip().lower()
        for engine in cls.__subclasses__():
            if engine.handles(key):
                return engine(engine.aliases.get(key, key))
        known = ", ".join(alias for engine in cls.__subclasses__() for alias in engine.aliases)
        raise UnknownModelError(f"Unknown LLM model {name!r}. Use one of: {known} (or an exact id)")

    @classmethod
    @abstractmethod
    def handles(cls, name: str) -> bool:
        """True when this engine can run ``name`` (already lowercased)."""

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Settings that change the answer (part of the LLM cache key)."""

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """None when the engine can run (valid key, server up), else a human-readable reason."""

    @abstractmethod
    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMUsage]:
        """Return a JSON object that validates against ``schema`` plus usage/cost."""


def prompt_key(system: str, user: str) -> str:
    """Stable key of a prompt pair, used to look up recorded answers."""
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()
