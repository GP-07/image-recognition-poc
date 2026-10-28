"""LLM clients without network: the Anthropic SDK is replaced by a stub returning real
``anthropic.types.Message`` objects (or raising the SDK's typed errors)."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from menu_poc.llm.client import (
    AnthropicClient,
    FakeLLMClient,
    LLMError,
    LLMOutputError,
    LLMRetryableError,
    make_llm_client,
    prompt_key,
)
from menu_poc.llm.pricing import UnknownModelError, cost_usd

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _message(text: str, stop_reason: str = "end_turn", **extra: Any) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 2000, "output_tokens": 1500},
            **extra,
        }
    )


class _StubMessages:
    def __init__(self, result: Message | Exception) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Message:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _StubSDK:
    def __init__(self, result: Message | Exception) -> None:
        self.messages = _StubMessages(result)


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls("boom", response=response, body=None)


def test_anthropic_client_request_shape_and_usage():
    sdk = _StubSDK(_message(json.dumps({"dishes": []})))
    client = AnthropicClient("claude-haiku-5-5", sdk_client=sdk)
    data, usage = client.complete_json("sys", "user", SCHEMA)

    assert data == {"dishes": []}
    kwargs = sdk.messages.kwargs
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "user"}]
    assert kwargs["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert kwargs["max_tokens"] == 16000
    assert not {"temperature", "top_p", "top_k"} & set(kwargs)
    assert usage.input_tokens == 2000 and usage.output_tokens == 1500
    assert usage.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", 2000, 1500))
    assert usage.latency_s >= 0 and not usage.cached


def test_refusal_raises_output_error():
    msg = _message(
        "",
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": None, "explanation": "nope"},
    )
    client = AnthropicClient(sdk_client=_StubSDK(msg))
    with pytest.raises(LLMOutputError, match="refused.*nope"):
        client.complete_json("s", "u", SCHEMA)


def test_max_tokens_raises_output_error():
    client = AnthropicClient(sdk_client=_StubSDK(_message('{"dishes": [', "max_tokens")))
    with pytest.raises(LLMOutputError, match="truncated"):
        client.complete_json("s", "u", SCHEMA)


def test_invalid_json_raises_output_error():
    client = AnthropicClient(sdk_client=_StubSDK(_message("not json")))
    with pytest.raises(LLMOutputError, match="invalid JSON"):
        client.complete_json("s", "u", SCHEMA)


@pytest.mark.parametrize(
    ("cls", "status", "expected"),
    [
        (anthropic.RateLimitError, 429, LLMRetryableError),
        (anthropic.InternalServerError, 500, LLMRetryableError),
        (anthropic.BadRequestError, 400, LLMError),
        (anthropic.AuthenticationError, 401, LLMError),
    ],
)
def test_api_errors_are_classified(cls, status, expected):
    client = AnthropicClient(sdk_client=_StubSDK(_status_error(cls, status)))
    with pytest.raises(expected) as info:
        client.complete_json("s", "u", SCHEMA)
    assert (info.type is LLMRetryableError) == (expected is LLMRetryableError)


def test_connection_error_is_retryable():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = AnthropicClient(sdk_client=_StubSDK(anthropic.APIConnectionError(request=request)))
    with pytest.raises(LLMRetryableError):
        client.complete_json("s", "u", SCHEMA)


def test_unknown_model_fails_at_construction():
    with pytest.raises(UnknownModelError):
        AnthropicClient("claude-unknown")


def test_fake_client_by_prompt_hash_and_callable():
    fake = FakeLLMClient({prompt_key("s", "u"): {"dishes": [{"name": "X"}]}})
    data, usage = fake.complete_json("s", "u", SCHEMA)
    assert data == {"dishes": [{"name": "X"}]}
    assert usage.cost_usd == 0.0 and usage.input_tokens > 0 and usage.model == "fake"
    assert fake.calls == [("s", "u")]
    with pytest.raises(LLMError, match="no recorded answer"):
        fake.complete_json("s", "other", SCHEMA)

    echo = FakeLLMClient(lambda system, user, schema: {"user": user})
    assert echo.complete_json("s", "hi", SCHEMA)[0] == {"user": "hi"}


def test_factory():
    real = make_llm_client("claude-sonnet-5-5")
    assert isinstance(real, AnthropicClient) and real.model == "claude-sonnet-5-5"
    fake = make_llm_client("claude-haiku-5-5", fake_responses={})
    assert isinstance(fake, FakeLLMClient) and fake.model == "fake/claude-haiku-5-5"
