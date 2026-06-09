import json

import httpx
import pytest
from openai import AsyncOpenAI

from nexus.config import ModelCfg
from nexus.llm.client import ChatClient, LLMError, _parse_json_payload, _parse_sse_line


async def _use_mock_transport(
    client: ChatClient,
    handler: httpx.MockTransport | httpx.SyncHandler | httpx.AsyncHandler,
) -> None:
    await client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._client = AsyncOpenAI(
        api_key="test",
        base_url=client.base_url,
        max_retries=0,
        http_client=client._http_client,
    )


def test_provider_routing_deepinfra() -> None:
    cfg = ModelCfg(provider="deepinfra", model="m", api_key="k")
    c = ChatClient.from_cfg(cfg, role="x")
    assert c.base_url.startswith("https://api.deepinfra.com")
    assert c.role == "x"


def test_provider_routing_ollama_no_key_needed() -> None:
    cfg = ModelCfg(provider="ollama", model="qwen2.5:3b")
    c = ChatClient.from_cfg(cfg, role="light")
    assert c.base_url.startswith("http://localhost:11434")


def test_explicit_base_url_overrides_provider_default() -> None:
    cfg = ModelCfg(provider="ollama", model="m", base_url="http://other:9999")
    c = ChatClient.from_cfg(cfg, role="x")
    assert c.base_url == "http://other:9999"


def test_unknown_provider_raises() -> None:
    cfg = ModelCfg(provider="weird-cloud", model="m")
    with pytest.raises(LLMError):
        ChatClient.from_cfg(cfg, role="x")


def test_parse_json_payload_strict() -> None:
    assert _parse_json_payload('{"a": 1}') == {"a": 1}


def test_parse_json_payload_extracts_from_noisy_text() -> None:
    noisy = 'Sure, here is the JSON:\n```json\n{"name": "x", "n": 2}\n```\n'
    out = _parse_json_payload(noisy)
    assert out == {"name": "x", "n": 2}


def test_parse_json_payload_empty_returns_empty_dict() -> None:
    assert _parse_json_payload("") == {}


def test_parse_json_payload_unparseable_raises() -> None:
    with pytest.raises(LLMError):
        _parse_json_payload("not json at all")


def test_parse_sse_line_reads_openai_delta() -> None:
    payload = _parse_sse_line(
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
    )

    assert payload == {
        "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]
    }
    assert _parse_sse_line("data: [DONE]") is None


@pytest.mark.asyncio
async def test_deepinfra_chat_stream_collects_and_emits_tokens() -> None:
    lines = [
        {
            "choices": [
                {"delta": {"content": "hello "}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 2}, "choices": []},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        content = "".join(f"data: {json.dumps(line)}\n\n" for line in lines)
        content += "data: [DONE]\n\n"
        return httpx.Response(200, content=content.encode("utf-8"))

    seen: list[dict[str, str]] = []

    async def token_sink(token: dict[str, str]) -> None:
        seen.append(token)

    client = ChatClient.from_cfg(
        ModelCfg(provider="deepinfra", model="m", api_key="k"),
        role="drafter",
        token_sink=token_sink,
    )
    await _use_mock_transport(client, handler)
    try:
        response = await client.chat([{"role": "user", "content": "go"}])
    finally:
        await client.aclose()

    assert response.content == "hello world"
    assert response.usage.prompt == 3
    assert response.usage.completion == 2
    assert [token["text"] for token in seen] == ["hello ", "world"]
    assert all(token["role"] == "drafter" for token in seen)


@pytest.mark.asyncio
async def test_deepinfra_json_mode_does_not_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "stream" not in body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = ChatClient.from_cfg(
        ModelCfg(provider="deepinfra", model="m", api_key="k"),
        role="critic",
    )
    await _use_mock_transport(client, handler)
    try:
        payload, usage = await client.chat_json([{"role": "user", "content": "go"}])
    finally:
        await client.aclose()

    assert payload == {"ok": True}
    assert usage.total == 2


@pytest.mark.asyncio
async def test_chat_json_can_stream_when_requested() -> None:
    lines = [
        {"choices": [{"delta": {"content": "{\"ok\":"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " true}"}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 2, "completion_tokens": 3}, "choices": []},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["response_format"] == {"type": "json_object"}
        content = "".join(f"data: {json.dumps(line)}\n\n" for line in lines)
        content += "data: [DONE]\n\n"
        return httpx.Response(200, content=content.encode("utf-8"))

    seen: list[dict[str, str]] = []

    async def token_sink(token: dict[str, str]) -> None:
        seen.append(token)

    client = ChatClient.from_cfg(
        ModelCfg(provider="deepinfra", model="m", api_key="k"),
        role="architect",
        token_sink=token_sink,
    )
    await _use_mock_transport(client, handler)
    try:
        payload, usage = await client.chat_json(
            [{"role": "user", "content": "go"}],
            stream=True,
        )
    finally:
        await client.aclose()

    assert payload == {"ok": True}
    assert usage.total == 5
    assert [token["text"] for token in seen] == ["{\"ok\":", " true}"]
    assert all(token["role"] == "architect" for token in seen)


@pytest.mark.asyncio
async def test_chat_uses_configured_sampling_defaults() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = ChatClient.from_cfg(
        ModelCfg(
            provider="openai",
            model="m",
            api_key="k",
            base_url="https://example.test/v1",
            temperature=0.0,
            top_p=0.9,
        ),
        role="drafter",
    )
    await _use_mock_transport(client, handler)
    try:
        await client.chat_markdown([{"role": "user", "content": "go"}])
    finally:
        await client.aclose()

    assert seen["temperature"] == 0.0
    assert seen["top_p"] == 0.9


@pytest.mark.asyncio
async def test_chat_json_repairs_invalid_json_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            content = '{"ok":'
        else:
            body = json.loads(request.content.decode("utf-8"))
            assert "not valid complete JSON" in body["messages"][-1]["content"]
            content = '{"ok": true}'
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = ChatClient.from_cfg(
        ModelCfg(provider="openai", model="m", api_key="k", base_url="https://example.test/v1"),
        role="critic",
    )
    await _use_mock_transport(client, handler)
    try:
        payload, usage = await client.chat_json([{"role": "user", "content": "go"}])
    finally:
        await client.aclose()

    assert payload == {"ok": True}
    assert usage.total == 4
    assert calls == 2


@pytest.mark.asyncio
async def test_stream_failure_retries_non_stream() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            return httpx.Response(500, text="stream broke")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "fallback"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    client = ChatClient.from_cfg(
        ModelCfg(provider="deepinfra", model="m", api_key="k"),
        role="drafter",
    )
    await _use_mock_transport(client, handler)
    try:
        response = await client.chat([{"role": "user", "content": "go"}])
    finally:
        await client.aclose()

    assert response.content == "fallback"
    assert response.usage.total == 3
    assert calls == 2
