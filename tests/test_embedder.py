from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from nexus.config import ModelCfg
from nexus.ingest.embedder import EmbedderClient, EmbedderError


async def _use_mock_transport(client: EmbedderClient, handler) -> None:
    await client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    base_url = client.base_url
    if client.provider == "jina-local" and not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url}/v1"
    client._client = AsyncOpenAI(
        api_key="test",
        base_url=base_url,
        max_retries=0,
        http_client=client._http_client,
    )
    client._health_client = httpx.AsyncClient(base_url=client.base_url)


@pytest.mark.asyncio
async def test_physical_batch_size_error_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": (
                        "input (667 tokens) is too large to process. increase the "
                        "physical batch size (current batch size: 512)"
                    )
                }
            },
        )

    client = EmbedderClient("http://embedder.test")
    await _use_mock_transport(client, handler)

    with pytest.raises(EmbedderError, match="physical batch size"):
        await client._call(["x"])

    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_deepinfra_embedder_uses_openai_embeddings_path() -> None:
    client = EmbedderClient.from_cfg(
        ModelCfg(
            provider="deepinfra",
            model="Qwen/Qwen3-Embedding-4B",
            api_key="key",
            base_url="https://api.deepinfra.com/v1/openai",
            instruction_profile="qwen3",
        )
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 2.0]}
                ],
                "model": "Qwen/Qwen3-Embedding-4B",
            },
        )

    await _use_mock_transport(client, handler)

    out = await client.embed_query("auth middleware", vector="dense_code")

    assert out == [1.0, 2.0]
    assert seen["path"].endswith("/embeddings")
    assert seen["json"]["model"] == "Qwen/Qwen3-Embedding-4B"
    assert seen["json"]["input"][0].startswith("Instruct: Given a developer search query")
    await client.aclose()
