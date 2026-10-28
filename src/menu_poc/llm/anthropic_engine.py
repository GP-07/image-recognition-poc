"""Claude via the official Anthropic SDK.

Structured outputs (``output_config.format``) make the answer always parse against the JSON
schema. No sampling parameters are sent (``temperature``/``top_p``/``top_k`` are rejected by
current models).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from menu_poc.constants import ANTHROPIC_RETRYABLE_STATUS, MODEL_PRICES_PER_MTOK
from menu_poc.llm.engine import LLMEngine, LLMError, LLMOutputError, LLMRetryableError
from menu_poc.llm.pricing import cost_usd
from menu_poc.models import LLMUsage


class AnthropicEngine(LLMEngine):
    """The SDK client is created lazily on the first call, so building this object needs no API
    key (e.g. when every answer comes from the cache)."""

    aliases = {
        "haiku": "claude-haiku-5-5",
        "sonnet": "claude-sonnet-5-5",
        "opus": "claude-opus-5-5",
    }

    @classmethod
    def handles(cls, name: str) -> bool:
        return name in cls.aliases or name in MODEL_PRICES_PER_MTOK  # known price = billable

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 16000,
        effort: str = "low",
        timeout_s: float = 120.0,
        max_retries: int = 3,
        sdk_client: Any | None = None,
    ) -> None:
        super().__init__(model)
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sdk = sdk_client

    def fingerprint(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}

    def unavailable_reason(self) -> str | None:
        """Checked once at start-up with ``models.retrieve``: free, fast, and it proves both that
        the key is valid and that it can use the model, before any image is processed."""
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return "ANTHROPIC_API_KEY is not set (environment or .env)"
        import anthropic

        try:
            self._client().models.retrieve(self.model, timeout=10)
        except anthropic.AuthenticationError:
            return "ANTHROPIC_API_KEY is invalid (authentication failed)"
        except (anthropic.PermissionDeniedError, anthropic.NotFoundError):
            return f"this API key cannot use model {self.model}"
        except anthropic.APIConnectionError as exc:
            return f"Anthropic API not reachable ({exc})"
        except anthropic.APIStatusError as exc:
            return f"Anthropic API error {exc.status_code} while checking the key"
        return None

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
            retryable = exc.status_code in ANTHROPIC_RETRYABLE_STATUS or exc.status_code >= 500
            error_type = LLMRetryableError if retryable else LLMError
            raise error_type(
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
