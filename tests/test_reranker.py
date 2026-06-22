from __future__ import annotations

import pytest

from anvay.config import ModelCfg
from anvay.retrieval.reranker import RerankerClient


@pytest.mark.asyncio
async def test_deepinfra_reranker_uses_inference_model_path(monkeypatch) -> None:
    client = RerankerClient.from_cfg(
        ModelCfg(
            provider="deepinfra",
            model="Qwen/Qwen3-Reranker-4B",
            api_key="key",
            base_url="https://api.deepinfra.com/v1/inference",
        )
    )
    seen = {}

    async def fake_post(path, *, json):
        seen["path"] = path
        seen["json"] = json

        class Response:
            status_code = 200

            def json(self):
                return {"scores": [0.2, 0.9]}

        return Response()

    monkeypatch.setattr(client._client, "post", fake_post)

    out = await client.rerank("query", ["doc-a", "doc-b"], top_k=1)

    assert seen["path"] == "/Qwen/Qwen3-Reranker-4B"
    assert seen["json"]["queries"] == ["query"]
    assert seen["json"]["documents"] == ["doc-a", "doc-b"]
    assert [(r.index, r.score) for r in out] == [(1, 0.9)]
    await client.aclose()
