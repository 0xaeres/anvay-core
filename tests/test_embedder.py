from __future__ import annotations

import pytest

from nexus.config import ModelCfg
from nexus.ingest.embedder import EmbedderClient, EmbedderError


@pytest.mark.asyncio
async def test_physical_batch_size_error_is_not_retried(monkeypatch) -> None:
    calls = 0

    async def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1

        class Response:
            status_code = 500
            text = (
                '{"error":{"message":"input (667 tokens) is too large to process. '
                'increase the physical batch size (current batch size: 512)"}}'
            )

        return Response()

    client = EmbedderClient("http://embedder.test")
    monkeypatch.setattr(client._client, "post", fake_post)

    with pytest.raises(EmbedderError, match="physical batch size"):
        await client._call(["x"])

    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_deepinfra_embedder_uses_openai_embeddings_path(monkeypatch) -> None:
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

    async def fake_post(path, *, json):
        seen["path"] = path
        seen["json"] = json

        class Response:
            status_code = 200

            def json(self):
                return {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

        return Response()

    monkeypatch.setattr(client._client, "post", fake_post)

    out = await client.embed_query("auth middleware", vector="dense_code")

    assert out == [1.0, 2.0]
    assert seen["path"] == "/embeddings"
    assert seen["json"]["model"] == "Qwen/Qwen3-Embedding-4B"
    assert seen["json"]["input"][0].startswith("Instruct: Given a developer search query")
    await client.aclose()
