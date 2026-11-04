"""The real OpenAI-compatible adapter exercised through a mocked HTTP wire."""

from __future__ import annotations

import json

import pytest

from acgn_rag.config import LLMConfig
from acgn_rag.core.ground import ProviderResponseError
from acgn_rag.core.ground.provider import make_openai_answer_client

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


def test_openai_answer_adapter_uses_answer_settings_and_returns_wire_content() -> None:
    pytest.importorskip("openai")
    respx = pytest.importorskip("respx")
    import httpx

    config = LLMConfig(
        answer_base_url="http://acgn-answer-test/v1",
        answer_api_key="test-answer-key",
        answer_model="test-answer-model",
        temperature=0.0,
    )
    canned = json.dumps(
        {
            "claims": [
                {
                    "text": "A supported answer.",
                    "evidence_indices": [1],
                }
            ]
        }
    )
    with respx.mock:
        route = respx.post("http://acgn-answer-test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-answer-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": canned},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        with httpx.Client(transport=httpx.MockTransport(respx.mock.handler)) as http_client:
            client = make_openai_answer_client(config, http_client=http_client)
            output = client.complete("system", "user", max_tokens=1024)

        assert output == canned
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "test-answer-model"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.0
        assert [message["role"] for message in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "chatcmpl-empty", "choices": []},
        {
            "id": "chatcmpl-no-content",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl-blank-content",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl-truncated",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"claims":['},
                    "finish_reason": "length",
                }
            ],
        },
        {
            "id": "chatcmpl-filtered",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter",
                }
            ],
        },
    ],
)
def test_openai_answer_adapter_rejects_incomplete_provider_responses(
    payload: dict[str, JsonValue],
) -> None:
    pytest.importorskip("openai")
    respx = pytest.importorskip("respx")
    import httpx

    config = LLMConfig(
        answer_base_url="http://acgn-answer-test/v1",
        answer_api_key="test-answer-key",
        answer_model="test-answer-model",
    )
    with respx.mock:
        respx.post("http://acgn-answer-test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        with httpx.Client(transport=httpx.MockTransport(respx.mock.handler)) as http_client:
            client = make_openai_answer_client(config, http_client=http_client)
            with pytest.raises(ProviderResponseError):
                client.complete("system", "user", max_tokens=1024)
