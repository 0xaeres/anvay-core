from __future__ import annotations

import httpx
import pytest
from qdrant_client.http import models as qm

from anvay.ingest.indexer import Indexer, IndexerError
from anvay.ingest.models import Chunk, ChunkKind, EmbeddedChunk, ResourceRef


def test_indexer_builds_qdrant_turboquant_config() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_enabled=True,
        quantization_type="turboquant",
        quantization_bits="bits2",
        quantization_always_ram=False,
    )

    config = indexer._dense_quantization_config()

    assert isinstance(config, qm.TurboQuantization)
    assert config.turbo.bits == qm.TurboQuantBitSize.BITS2
    assert config.turbo.always_ram is False


def test_indexer_can_disable_quantization() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_enabled=False,
    )

    assert indexer._dense_quantization_config() is None


def test_indexer_rejects_unknown_quantization_bits() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_bits="bits3",
    )

    with pytest.raises(IndexerError, match=r"unsupported vector_store\.quantization\.bits"):
        indexer._dense_quantization_config()


@pytest.mark.asyncio
async def test_indexer_retries_transient_qdrant_read_error(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def upsert(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadError("socket closed")
            return None

    fake = FakeClient()
    indexer = Indexer(url="http://127.0.0.1:6333")
    indexer.client = fake  # type: ignore[assignment]
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("anvay.ingest.indexer.asyncio.sleep", fake_sleep)

    ref = ResourceRef(source_id="local:test", uri="a.py", mime="text/x-python")
    chunk = Chunk(
        product_id="demo",
        resource=ref,
        content="print('x')",
        start_line=1,
        end_line=1,
        kind=ChunkKind.CODE,
    )
    embedded = EmbeddedChunk(chunk=chunk, vector=[1.0, 0.0], vector_name="dense_code")

    assert await indexer.upsert([embedded]) == 1
    assert fake.calls == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_indexer_splits_large_upserts() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def upsert(self, **kwargs):
            self.batch_sizes.append(len(kwargs["points"]))

    fake = FakeClient()
    indexer = Indexer(url="http://127.0.0.1:6333", upsert_batch_size=2)
    indexer.client = fake  # type: ignore[assignment]

    ref = ResourceRef(source_id="local:test", uri="a.py", mime="text/x-python")
    embedded = []
    for line in range(1, 6):
        chunk = Chunk(
            product_id="demo",
            resource=ref,
            content=f"print({line})",
            start_line=line,
            end_line=line,
            kind=ChunkKind.CODE,
        )
        embedded.append(
            EmbeddedChunk(chunk=chunk, vector=[1.0, 0.0], vector_name="dense_code")
        )

    assert await indexer.upsert(embedded) == 5
    assert fake.batch_sizes == [2, 2, 1]


@pytest.mark.asyncio
async def test_ids_by_resource_paginates_past_scroll_limit() -> None:
    class _Pt:
        def __init__(self, pid: str) -> None:
            self.id = pid

    class FakeClient:
        def __init__(self) -> None:
            self.scroll_calls: list[tuple[str, object]] = []

        async def scroll(self, *, collection_name, scroll_filter, limit, offset,
                         with_payload, with_vectors):
            self.scroll_calls.append((collection_name, offset))
            if collection_name.endswith("text"):
                return [], None
            # Two pages for the code collection.
            if offset is None:
                return [_Pt(f"code-{i}") for i in range(1024)], "page2"
            return [_Pt("code-last")], None

    fake = FakeClient()
    indexer = Indexer(url="http://127.0.0.1:6333")
    indexer.client = fake  # type: ignore[assignment]

    found = await indexer.ids_by_resource(product_id="demo", resource_uri="big.py")

    code_ids = found[indexer._code]
    assert len(code_ids) == 1025
    assert "code-last" in code_ids
    # Second page requested with the returned offset.
    assert (indexer._code, "page2") in fake.scroll_calls


@pytest.mark.asyncio
async def test_delete_by_resource_deletes_all_paginated_ids() -> None:
    class _Pt:
        def __init__(self, pid: str) -> None:
            self.id = pid

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def scroll(self, *, collection_name, scroll_filter, limit, offset,
                         with_payload, with_vectors):
            if collection_name.endswith("text"):
                return [], None
            if offset is None:
                return [_Pt("a"), _Pt("b")], "next"
            return [_Pt("c")], None

        async def delete(self, *, collection_name, points_selector):
            self.deleted.extend(str(p) for p in points_selector.points)

    fake = FakeClient()
    indexer = Indexer(url="http://127.0.0.1:6333")
    indexer.client = fake  # type: ignore[assignment]

    deleted = await indexer.delete_by_resource(product_id="demo", resource_uri="big.py")

    assert sorted(deleted) == ["a", "b", "c"]
    assert sorted(fake.deleted) == ["a", "b", "c"]
