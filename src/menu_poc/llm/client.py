"""LLM clients satisfying ``menu_poc.interfaces.LLMClient``.

- ``AnthropicClient``: Claude Messages API with structured outputs (``output_config.format``), so
  the answer always parses against the JSON schema. No sampling parameters are sent
  (``temperature``/``top_p``/``top_k`` are rejected by current models).
- ``OllamaClient`` (``llm/ollama.py``): free local models, selected with ``ollama/<model>``.
- ``FakeLLMClient``: canned answers (by prompt hash or a callable), zero cost, never touches the
  network. Used by tests and offline demos.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from menu_poc.llm.pricing import cost_usd, price_for
from menu_poc.llm.text import estimate_tokens
from menu_poc.models import LLMUsage

DEFAULT_MODEL = "claude-haiku-5-5"
FakeResponses = Mapping[str, dict[str, Any]] | Callable[[str, str, dict[str, Any]], dict[str, Any]]


class LLMError(RuntimeError):
    """The LLM call failed and retrying the same request will not help."""


class LLMRetryableError(LLMError):
    """Transient failure (rate limit, overload, 5xx, network) left after the SDK's own retries."""


class LLMOutputError(LLMError):
    """The call succeeded (and was billed) but the answer is unusable; ``usage`` keeps the cost."""

    def __init__(self, message: str, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage

    """The model answered but the output is unusable (refusal, truncation, invalid JSON)."""


def prompt_key(system: str, user: str) -> str:
    """Stable key of a prompt pair, used to look up recorded answers."""
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


class AnthropicClient:
    """Claude via the official SDK. The SDK client is created lazily on the first call, so
    building this object needs no API key (e.g. when every answer comes from the cache)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 16000,
        effort: str = "low",
        timeout_s: float = 120.0,
        max_retries: int = 3,
        sdk_client: Any | None = None,
    ) -> None:
        price_for(model)  # fail fast on a model without a known price
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sdk = sdk_client

    def unavailable_reason(self) -> str | None:
        """None when credentials are configured, else why the LLM step cannot run."""
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return None
        return "ANTHROPIC_API_KEY is not set (environment or .env)"

    def _client(self) -> Any:
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.Anthropic(timeout=self.timeout_s, max_retries=self.max_retries)
        return self._sdk

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMUsage]:
        import anthropic

        start = time.perf_counter()
        try:
            response = self._client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code in (408, 409, 429) or exc.status_code >= 500
            cls = LLMRetryableError if retryable else LLMError
            raise cls(
                f"Anthropic API error {exc.status_code} ({self.model}): {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMRetryableError(
                f"Anthropic API connection error ({self.model}): {exc}"
            ) from exc
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        usage = LLMUsage(
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_usd(self.model, in_tok, out_tok),
            latency_s=time.perf_counter() - start,
        )
        try:
            _check_stop_reason(response, self.model, self.max_tokens)
            text = "".join(block.text for block in response.content if block.type == "text")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(f"Model {self.model} returned invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                kind = type(data).__name__
                raise LLMOutputError(f"Model {self.model} returned {kind}, not object")
        except LLMOutputError as exc:
            exc.usage = usage  # billed anyway: keep the cost visible
            raise
        return data, usage

    def fingerprint(self) -> dict[str, Any]:
        """Settings that change the answer (part of the LLM cache key)."""
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}


def _check_stop_reason(response: Any, model: str, max_tokens: int) -> None:
    reason = response.stop_reason
    if reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        raise LLMOutputError(
            f"Model {model} refused (category={category}): {explanation or 'no explanation'}"
        )
    if reason in ("max_tokens", "model_context_window_exceeded"):
        raise LLMOutputError(
            f"Model {model} output truncated ({reason}, max_tokens={max_tokens}); "
            "the JSON is incomplete"
        )


class FakeLLMClient:
    """Deterministic stand-in: answers come from ``responses`` keyed by ``prompt_key`` or from a
    callable ``(system, user, schema) -> dict``. Token counts are estimated, cost is zero."""

    def __init__(self, responses: FakeResponses, model: str = "fake") -> None:
        self.model = model
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

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
                raise LLMError(f"FakeLLMClient has no recorded answer for prompt {key[:12]}")
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

    def fingerprint(self) -> dict[str, Any]:
        return {"model": self.model}


def make_llm_client(model: str = DEFAULT_MODEL, fake_responses: FakeResponses | None = None) -> Any:
    """Client for ``model``: ``ollama/<name>`` -> local Ollama, otherwise Claude. With
    ``fake_responses`` a fake (named ``fake/<model>``, so its answers never share cache keys with
    the real model)."""
    if fake_responses is not None:
        return FakeLLMClient(fake_responses, model=f"fake/{model}")
    from menu_poc.llm.ollama import PREFIX, OllamaClient

    if model.startswith(PREFIX):
        return OllamaClient(model)
    return AnthropicClient(model)
