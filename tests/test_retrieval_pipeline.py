from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.retrieval import pipeline


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_query(self, text: str, *, vector: str) -> list[float]:
        self.calls.append(vector)
        return [1.0] if vector == "dense_code" else [2.0]


@pytest.mark.asyncio
async def test_auto_retrieval_embeds_code_and_text_queries(monkeypatch) -> None:
    embedder = FakeEmbedder()
    ctx = SimpleNamespace(embedder=embedder)

    async def fake_hybrid_search(**kwargs):
        assert kwargs["query_vectors"] == {"code": [1.0], "text": [2.0]}
        return []

    monkeypatch.setattr(pipeline, "_hybrid_search", fake_hybrid_search)

    result = await pipeline.retrieve(
        ctx=ctx, product_id="p", query="how does auth work?", mode="auto"
    )

    assert result.hits == []
    assert embedder.calls == ["dense_code", "dense_text"]
