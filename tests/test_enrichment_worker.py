from __future__ import annotations

from pathlib import Path

import pytest

from nexus.config import NexusConfig
from nexus.ingest.enrichment_worker import EnrichmentWorker
from nexus.ingest.models import EmbeddedChunk
from nexus.registry import Registry
from nexus.retrieval.sparse import SparseVector


def _config(tmp_path: Path) -> NexusConfig:
    return NexusConfig(
        models={
            "council": {"provider": "test", "model": "test"},
            "light": {"provider": "test", "model": "test"},
            "embedding": {"provider": "test", "model": "embed-v1", "url": "http://embed"},
            "reranker": {"provider": "test", "model": "test", "url": "http://rerank"},
        },
        storage={
            "proposal_queue": tmp_path / "proposals.db",
            "council_checkpoint": tmp_path / "council.sqlite",
        },
    )


class FakeEmbedder:
    async def embed_chunks(self, chunks):
        return [
            EmbeddedChunk(chunk=c, vector=[0.1, 0.2], vector_name="dense_text")
            for c in chunks
        ]

    async def aclose(self) -> None:
        pass


class FakeEnricher:
    async def enrich(self, chunks, *, doc_contents):
        for chunk in chunks:
            chunk.context_summary = "Enriched context."
        return chunks

    async def aclose(self) -> None:
        pass


class FakeIndexer:
    requires_sparse_vectors = True

    def __init__(self):
        self.upserted = []

    async def upsert(self, embedded, *, sparse_by_id=None, **kwargs):
        self.upserted.append((list(embedded), sparse_by_id or {}, kwargs))
        return len(embedded)

    async def aclose(self) -> None:
        pass


async def _fake_sparse(texts):
    return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def _enqueue(registry: Registry, *, manifest_hash: str = "h1") -> None:
    registry.upsert_resource_manifest(
        {
            "product": "p",
            "sourceKey": "source",
            "resourceUri": "doc.txt",
            "contentHash": manifest_hash,
            "mime": "text/plain",
            "sizeBytes": 100,
            "lastSeenSync": "now",
            "chunkIds": ["chunk"],
            "indexedAt": "now",
            "embeddingVersion": "v",
        }
    )
    registry.enqueue_enrichment_job(
        {
            "product": "p",
            "sourceKey": "source",
            "resourceUri": "doc.txt",
            "sourceId": "local:test",
            "mime": "text/plain",
            "sizeBytes": 100,
            "contentHash": "h1",
            "content": "hello world " * 20,
        }
    )


@pytest.mark.asyncio
async def test_worker_processes_enrichment_job(tmp_path: Path, monkeypatch) -> None:
    from nexus.ingest import enrichment_worker

    registry = Registry(tmp_path / "registry.db")
    _enqueue(registry)
    indexer = FakeIndexer()
    worker = EnrichmentWorker(
        registry=registry,
        config=_config(tmp_path),
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
    )
    monkeypatch.setattr(enrichment_worker, "aencode_passages", _fake_sparse)

    processed = await worker.process_one()

    assert processed is True
    assert registry.enrichment_job_counts("p")["pending"] == 0
    assert indexer.upserted
    embedded, sparse_by_id, _kwargs = indexer.upserted[0]
    assert embedded[0].chunk.context_summary == "Enriched context."
    assert sparse_by_id


@pytest.mark.asyncio
async def test_worker_drops_stale_job(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.db")
    _enqueue(registry, manifest_hash="newer")
    indexer = FakeIndexer()
    worker = EnrichmentWorker(
        registry=registry,
        config=_config(tmp_path),
        embedder=FakeEmbedder(),
        enricher=FakeEnricher(),
        indexer=indexer,
    )

    processed = await worker.process_one()

    assert processed is True
    assert registry.enrichment_job_counts("p")["pending"] == 0
    assert not indexer.upserted
