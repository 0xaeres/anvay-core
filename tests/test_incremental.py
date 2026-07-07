from __future__ import annotations

from pathlib import Path

import pytest

from anvay.ingest import incremental
from anvay.ingest.chunker import chunk_resource
from anvay.ingest.models import ResourceRef
from anvay.registry import Registry
from anvay.retrieval.sparse import SparseVector

CONTENT_V1 = "\n".join(f"line {i} of the original document body text" for i in range(40))
CONTENT_V2 = "\n".join(f"line {i} of the rewritten document body text" for i in range(40))


def _ref(uri: str = "doc.txt") -> ResourceRef:
    return ResourceRef(source_id="local:test", uri=uri, mime="text/plain")


class FakeEnricher:
    async def enrich(self, chunks, *, doc_contents):
        return chunks


class FakeEmbedder:
    async def embed_chunks(self, chunks):
        from anvay.ingest.models import EmbeddedChunk

        return [
            EmbeddedChunk(chunk=c, vector=[1.0, 0.0], vector_name="dense_text")
            for c in chunks
        ]


class FakeIndexer:
    requires_sparse_vectors = True

    def __init__(self, existing: dict[str, list[str]] | None = None, *, fail_upsert: bool = False):
        self.existing = existing or {}
        self.fail_upsert = fail_upsert
        self.calls: list[str] = []
        self.upserted_ids: list[str] = []
        self.deleted_ids: list[str] = []

    async def ids_by_resource(self, *, product_id: str, resource_uri: str):
        self.calls.append("ids_by_resource")
        return dict(self.existing)

    async def upsert(self, embedded, *, sparse_by_id=None, **kwargs):
        self.calls.append("upsert")
        if self.fail_upsert:
            raise RuntimeError("qdrant down")
        self.upserted_ids.extend(e.chunk.id for e in embedded)
        return len(embedded)

    async def delete_points_by_ids(self, point_ids):
        self.calls.append("delete")
        if isinstance(point_ids, dict):
            ids = [pid for v in point_ids.values() for pid in v]
        else:
            ids = list(point_ids)
        self.deleted_ids.extend(ids)
        return len(ids)


@pytest.fixture(autouse=True)
def _fake_sparse(monkeypatch):
    async def fake(texts):
        return [SparseVector(indices=[0], values=[1.0]) for _ in texts]

    monkeypatch.setattr(incremental, "aencode_passages", fake)


@pytest.mark.asyncio
async def test_upsert_happens_before_delete() -> None:
    indexer = FakeIndexer(existing={"anvay_text": ["old-1", "old-2"]})
    await incremental.reindex_resource(
        product_id="demo",
        resource=_ref(),
        content=CONTENT_V2,
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
    )
    assert indexer.calls.index("upsert") < indexer.calls.index("delete")
    assert set(indexer.deleted_ids) == {"old-1", "old-2"}


@pytest.mark.asyncio
async def test_upsert_failure_leaves_old_points_intact() -> None:
    indexer = FakeIndexer(
        existing={"anvay_text": ["old-1", "old-2"]}, fail_upsert=True
    )
    with pytest.raises(RuntimeError):
        await incremental.reindex_resource(
            product_id="demo",
            resource=_ref(),
            content=CONTENT_V2,
            embedder=FakeEmbedder(),
            enricher=FakeEnricher(),
            indexer=indexer,
        )
    assert "delete" not in indexer.calls
    assert indexer.deleted_ids == []


@pytest.mark.asyncio
async def test_unchanged_chunk_ids_are_not_deleted() -> None:
    chunks = chunk_resource("demo", _ref(), CONTENT_V1)
    assert chunks
    same_ids = [c.id for c in chunks]
    indexer = FakeIndexer(existing={"anvay_text": list(same_ids)})
    result = await incremental.reindex_resource(
        product_id="demo",
        resource=_ref(),
        content=CONTENT_V1,
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
    )
    assert result.chunks_indexed == len(same_ids)
    assert indexer.deleted_ids == []
    assert "delete" not in indexer.calls


@pytest.mark.asyncio
async def test_empty_new_content_deletes_all_old_points() -> None:
    indexer = FakeIndexer(existing={"anvay_text": ["old-1"], "anvay_code": ["old-2"]})
    result = await incremental.reindex_resource(
        product_id="demo",
        resource=_ref(),
        content="",
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
    )
    assert result.chunks_indexed == 0
    assert set(indexer.deleted_ids) == {"old-1", "old-2"}
    assert "upsert" not in indexer.calls


@pytest.mark.asyncio
async def test_manifest_written_last_with_indexed_chunk_ids(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.db")
    indexer = FakeIndexer()
    await incremental.reindex_resource(
        product_id="demo",
        resource=_ref(),
        content=CONTENT_V1,
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
        registry=registry,
        source_key="local:test",
        embedding_version="v1hash",
    )
    rows = registry.list_resource_manifests("demo", "local:test")
    assert len(rows) == 1
    row = rows[0]
    assert row["resourceUri"] == "doc.txt"
    assert sorted(row["chunkIds"]) == sorted(indexer.upserted_ids)
    assert row["embeddingVersion"] == "v1hash"


@pytest.mark.asyncio
async def test_manifest_not_written_when_upsert_fails(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.db")
    indexer = FakeIndexer(fail_upsert=True)
    with pytest.raises(RuntimeError):
        await incremental.reindex_resource(
            product_id="demo",
            resource=_ref(),
            content=CONTENT_V1,
            embedder=FakeEmbedder(),
            enricher=FakeEnricher(),
            indexer=indexer,
            registry=registry,
            source_key="local:test",
        )
    # Only the pending stub exists — no chunk ids were committed.
    rows = registry.list_resource_manifests("demo", "local:test")
    assert len(rows) == 1
    assert rows[0]["indexStatus"] == "pending"
    assert rows[0]["chunkIds"] == []
