"""Free, local LLM via Ollama (https://ollama.com): no API key, no cost, data stays on the machine.

Select it with ``--llm-model ollama/<model>``, e.g. ``ollama/qwen2.5:7b``. Ollama's structured
outputs (``format`` = JSON schema) keep the answer parseable, like the Anthropic client. Only the
standard library is used (one HTTP POST), so it adds no dependency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from menu_poc.llm.client import LLMError, LLMOutputError, LLMRetryableError
from menu_poc.models import LLMUsage

PREFIX = "ollama/"
DEFAULT_HOST = "http://localhost:11434"


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


class OllamaClient:
    def __init__(self, model: str, *, host: str | None = None, timeout_s: float = 600.0) -> None:
        self.model = model if model.startswith(PREFIX) else PREFIX + model
        self.name = self.model.removeprefix(PREFIX)
        self.host = host or ollama_host()
        self.timeout_s = timeout_s

    def fingerprint(self) -> dict[str, Any]:
        # temperature 0: local models allow it, so answers are as repeatable as they get.
        return {"model": self.model, "temperature": 0}

    def unavailable_reason(self) -> str | None:
        """None when the server answers and the model is pulled, else a human-readable reason."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                models = {m["name"] for m in json.load(resp).get("models", [])}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return f"Ollama server not reachable at {self.host} ({exc})"
        if self.name not in models and f"{self.name}:latest" not in models:
            return f"Ollama model {self.name!r} not pulled (run: ollama pull {self.name})"
        return None

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], LLMUsage]:
        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Ollama error {exc.code} ({self.model}): {exc.read()[:200]!r}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMRetryableError(f"Ollama connection error ({self.model}): {exc}") from exc

        usage = LLMUsage(
            model=self.model,
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
            cost_usd=0.0,  # local compute; its cost shows up as latency
            latency_s=time.perf_counter() - start,
        )
        if payload.get("done_reason") == "length":
            raise LLMOutputError(f"Model {self.model} output truncated", usage=usage)
        try:
            data = json.loads(payload["message"]["content"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMOutputError(f"Model {self.model} returned invalid JSON: {exc}", usage) from exc
        if not isinstance(data, dict):
            raise LLMOutputError(f"Model {self.model} returned {type(data).__name__}", usage)
        return data, usage
