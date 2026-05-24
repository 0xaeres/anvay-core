from __future__ import annotations

import pytest

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
