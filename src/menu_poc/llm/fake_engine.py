"""Deterministic stand-in LLM: zero cost, never touches the network. For tests and offline demos."""

from __future__ import annotations

import json
import time
from typing import Any

from menu_poc.llm.engine import LLMEngine, LLMError, prompt_key
from menu_poc.llm.text import estimate_tokens
from menu_poc.models import LLMUsage
from menu_poc.typings import FakeResponses


class FakeEngine(LLMEngine):
    """Answers come from ``responses`` keyed by ``prompt_key`` or from a callable
    ``(system, user, schema) -> dict``. Token counts are estimated, cost is zero.
    Built directly by tests; ``get_for_alias`` never selects it."""

    @classmethod
    def handles(cls, name: str) -> bool:
        return False

    def __init__(self, responses: FakeResponses, model: str = "fake") -> None:
        super().__init__(model)
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def fingerprint(self) -> dict[str, Any]:
        return {"model": self.model}

    def unavailable_reason(self) -> str | None:
        return None

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMUsage]:
        start = time.perf_counter()
        self.calls.append((system, user))
        if callable(self._responses):
            data = self._responses(system, user, schema)
        else:
            key = prompt_key(system, user)
            if key not in self._responses:
                raise LLMError(f"FakeEngine has no recorded answer for prompt {key[:12]}")
            data = self._responses[key]
        data = json.loads(json.dumps(data))  # defensive copy, and proves it is JSON
        usage = LLMUsage(
            model=self.model,
            input_tokens=estimate_tokens(system) + estimate_tokens(user),
            output_tokens=estimate_tokens(json.dumps(data, ensure_ascii=False)),
            cost_usd=0.0,
            latency_s=time.perf_counter() - start,
        )
        return data, usage
