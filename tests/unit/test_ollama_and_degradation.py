"""Local Ollama client (HTTP mocked) and graceful degradation when the LLM cannot run."""

import io
import json
import urllib.error

import pytest

from menu_poc.cli import resolve_llm
from menu_poc.llm import ollama_engine as ollama
from menu_poc.llm.anthropic_engine import AnthropicEngine
from menu_poc.llm.engine import LLMEngine, LLMOutputError
from menu_poc.llm.ollama_engine import OllamaEngine
from menu_poc.models import RunConfig


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
    client = LLMEngine.get_for_alias("ollama/qwen2.5:7b")
    assert isinstance(client, OllamaEngine) and client.name == "qwen2.5:7b"
    assert isinstance(LLMEngine.get_for_alias("claude-haiku-5-5"), AnthropicEngine)


def test_ollama_returns_json_with_zero_cost(monkeypatch):
    answer = {"dishes": [{"name": "Lamb curry"}]}
    chat = {
        "message": {"content": json.dumps(answer)},
        "prompt_eval_count": 900,
        "eval_count": 120,
        "done_reason": "stop",
    }
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/chat": chat}))
    data, usage = OllamaEngine("qwen2.5:7b").complete_json("sys", "user", {"type": "object"})
    assert data == answer
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (900, 120, 0.0)


def test_ollama_truncation_is_an_output_error(monkeypatch):
    chat = {"message": {"content": "{"}, "done_reason": "length", "eval_count": 5}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/chat": chat}))
    with pytest.raises(LLMOutputError) as exc:
        OllamaEngine("m").complete_json("s", "u", {})
    assert exc.value.usage.output_tokens == 5


def test_ollama_availability(monkeypatch):
    tags = {"models": [{"name": "qwen2.5:7b"}]}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen({"/api/tags": tags}))
    assert OllamaEngine("qwen2.5:7b").unavailable_reason() is None
    assert "not pulled" in OllamaEngine("llama3.2").unavailable_reason()
    down = {"/api/tags": urllib.error.URLError("refused")}
    monkeypatch.setattr(ollama.urllib.request, "urlopen", _fake_urlopen(down))
    assert "not reachable" in OllamaEngine("qwen2.5:7b").unavailable_reason()


def test_missing_api_key_degrades_to_ocr_with_a_warning(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", llm_model="haiku"), warnings)
    assert not config.uses_llm and client is None
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
    monkeypatch.setattr(AnthropicEngine, "_client", lambda self: _FakeSdk())
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", llm_model="haiku"), warnings)
    assert config.uses_llm and isinstance(client, AnthropicEngine) and not warnings


def test_offline_mode_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    cfg = RunConfig(input_path="x.zip", llm_model="haiku", llm_offline=True)
    config, client = resolve_llm(cfg, [])
    assert config.uses_llm and client is not None


def test_invalid_api_key_degrades_before_processing(monkeypatch):
    import anthropic
    import httpx2 as httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong")
    response = httpx.Response(401, request=httpx.Request("GET", "https://api.anthropic.com"))
    error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
    monkeypatch.setattr(AnthropicEngine, "_client", lambda self: _FakeSdk(error))
    warnings: list[str] = []
    config, client = resolve_llm(RunConfig(input_path="x.zip", llm_model="haiku"), warnings)
    assert not config.uses_llm and client is None and "invalid" in warnings[0]


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
def test_model_aliases(alias, expected):
    assert RunConfig(input_path="x.zip", llm_model=alias).llm_model == expected


def test_unknown_model_is_rejected_early():
    with pytest.raises(ValueError, match="haiku, sonnet, opus, ollama"):
        RunConfig(input_path="x.zip", llm_model="gpt-5")


def test_no_llm_configured_warns_how_to_enable_it():
    warnings: list[str] = []
    config, engine = resolve_llm(RunConfig(input_path="x.zip"), warnings)
    assert engine is None and not config.uses_llm
    assert "OCR + rules only" in warnings[0] and "--llm-model haiku" in warnings[0]
