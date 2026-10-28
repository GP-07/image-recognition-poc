"""Local Ollama client (HTTP mocked) and graceful degradation when the LLM cannot run."""

import io
import json
import urllib.error

import pytest

from menu_poc.cli import resolve_llm
from menu_poc.config import RunConfig
from menu_poc.llm import ollama
from menu_poc.llm.client import AnthropicClient, LLMOutputError, make_llm_client
from menu_poc.llm.ollama import OllamaClient


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payloads):
    def urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        for suffix, payload in payloads.items():
            if url.endswith(suffix):
                if isinstance(payload, Exception):
                    raise payload
                return _Resp(json.dumps(payload).encode())
        raise AssertionError(url)

    return urlopen


def test_factory_routes_ollama_models():
    client = make_llm_client("ollama/qwen2.5:7b")
    assert isinstance(client, OllamaClient) and client.name == "qwen2.5:7b"
    assert isinstance(make_llm_client("claude-haiku-5-5"), AnthropicClient)


def test_ollama_returns_json_with_zero_cost(monkeypatch):
    answer = {"dishes": [{"name": "Lamb curry"}]}
    chat = {
        "message": {"content": json.dumps(answer)},
        "prompt_eval_count": 900,
        "eval_count": 120,
        "done_reason": "stop",
    }
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/chat": chat}))
    data, usage = OllamaClient("qwen2.5:7b").complete_json("sys", "user", {"type": "object"})
    assert data == answer
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (900, 120, 0.0)


def test_ollama_truncation_is_an_output_error(monkeypatch):
    chat = {"message": {"content": "{"}, "done_reason": "length", "eval_count": 5}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/chat": chat}))
    with pytest.raises(LLMOutputError) as exc:
        OllamaClient("m").complete_json("s", "u", {})
    assert exc.value.usage.output_tokens == 5


def test_ollama_availability(monkeypatch):
    tags = {"models": [{"name": "qwen2.5:7b"}]}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/tags": tags}))
    assert OllamaClient("qwen2.5:7b").unavailable_reason() is None
    assert "not pulled" in OllamaClient("llama3.2").unavailable_reason()
    down = {"/api/tags": urllib.error.URLError("refused")}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen(down))
    assert "not reachable" in OllamaClient("qwen2.5:7b").unavailable_reason()


def test_missing_api_key_degrades_to_ocr_with_a_warning(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", mode="ocr+llm"), warnings)
    assert config.mode == "ocr" and client is None
    assert "ANTHROPIC_API_KEY" in warnings[0]


class _FakeModels:
    def __init__(self, error=None):
        self.error = error

    def retrieve(self, model_id, timeout=None):
        if self.error:
            raise self.error
        return {"id": model_id}


class _FakeSdk:
    def __init__(self, error=None):
        self.models = _FakeModels(error)


def test_api_key_present_keeps_the_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(AnthropicClient, "_client", lambda self: _FakeSdk())
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", mode="ocr+llm"), warnings)
    assert config.mode == "ocr+llm" and isinstance(client, AnthropicClient) and not warnings


def test_offline_mode_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    cfg = RunConfig(input_path="x.zip", mode="ocr+llm", llm_offline=True)
    config, client = resolve_llm(cfg, [])
    assert config.mode == "ocr+llm" and client is not None


def test_invalid_api_key_degrades_before_processing(monkeypatch):
    import anthropic
    import httpx2 as httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong")
    response = httpx.Response(401, request=httpx.Request("GET", "https://api.anthropic.com"))
    error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
    monkeypatch.setattr(AnthropicClient, "_client", lambda self: _FakeSdk(error))
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", mode="ocr+llm"), warnings)
    assert config.mode == "ocr" and client is None and "invalid" in warnings[0]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("HAIKU", "claude-haiku-5-5"),
        ("Sonnet", "claude-sonnet-5-5"),
        ("ollama", "ollama/qwen2.5:7b"),
        ("ollama/llama3.2:3b", "ollama/llama3.2:3b"),
        ("claude-haiku-5-5", "claude-haiku-5-5"),
    ],
)
def test_model_aliases(alias, expected, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert RunConfig(input_path="x.zip", llm_model=alias).llm_model == expected


def test_unknown_model_is_rejected_early():
    with pytest.raises(ValueError, match="haiku, sonnet, opus, ollama"):
        RunConfig(input_path="x.zip", llm_model="gpt-5")
