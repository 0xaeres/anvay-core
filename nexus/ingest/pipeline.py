"""End-to-end ingestion pipeline orchestrator.

Pulls resources from a source, chunks them, optionally enriches with contextual
summaries, embeds, and upserts into Qdrant.

Design:
- Files are collected into batches of FILE_BATCH_SIZE before any embedding call.
- Within each batch, reads are concurrent (READ_CONCURRENCY semaphore).
- A bad file is skipped; it does not abort the batch or the run.
- The embedder is called once per batch (not once per file).
- Transient embedder errors are retried with exponential backoff in EmbedderClient.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from nexus.config import NexusConfig
from nexus.ingest.chunker import chunk_resource
from nexus.ingest.embedder import EmbedderClient, EmbedderError
from nexus.ingest.enricher import ContextualEnricher
from nexus.ingest.indexer import Indexer
from nexus.ingest.models import Chunk, ResourceRef
from nexus.retrieval.sparse import aencode_passages

log = logging.getLogger(__name__)



class _Source(Protocol):
    source_id: str

    async def list_resources(self): ...
    async def read_resource(self, resource: ResourceRef) -> str: ...


@dataclass
class IngestStats:
    resources_seen: int = 0
    resources_indexed: int = 0
    resources_skipped: int = 0
    chunks_produced: int = 0
    chunks_indexed: int = 0
    embed_errors: int = 0  # batches that failed to embed (token limit, server error)


async def run_ingest(
    *,
    product_id: str,
    source: _Source,
    config: NexusConfig,
    enrich: bool = True,
) -> IngestStats:
    """One pass: discover → batch-read → chunk → enrich → embed → index."""
    stats = IngestStats()

    file_batch_size = config.ingestion.file_batch_size
    read_concurrency = config.ingestion.read_concurrency

    embedder = EmbedderClient(
        base_url=config.models.embedding.url or "http://localhost:8080",
        batch_size=config.ingestion.embed_batch_size,
    )
    enricher = ContextualEnricher(
        base_url=config.models.light.base_url or "https://api.deepinfra.com/v1/openai",
        model=config.models.light.model,
        api_key=config.models.light.api_key,
        enrich_code=config.ingestion.enrich_chunks.code,
        enrich_docs=config.ingestion.enrich_chunks.docs,
        concurrency=config.ingestion.enricher_concurrency,
    )
    indexer = Indexer(url=config.vector_store.url)

    try:
        await indexer.ensure_collections()

        pending: list[ResourceRef] = []

        async def flush(resources: list[ResourceRef]) -> None:
            if not resources:
                return

            sem = asyncio.Semaphore(read_concurrency)

            async def _read(r: ResourceRef) -> tuple[ResourceRef, str] | None:
                async with sem:
                    try:
                        return r, await source.read_resource(r)
                    except OSError as e:
                        log.debug("skipping %s: %s", r.uri, e)
                        stats.resources_skipped += 1
                        return None

            pairs = await asyncio.gather(*[_read(r) for r in resources])

            all_chunks: list[Chunk] = []
            indexed_count = 0
            for pair in pairs:
                if pair is None:
                    continue
                r, content = pair
                chunks = chunk_resource(product_id, r, content)
                if not chunks:
                    stats.resources_skipped += 1
                    continue
                all_chunks.extend(chunks)
                indexed_count += 1

            if not all_chunks:
                return

            if enrich:
                all_chunks = await enricher.enrich(all_chunks)

            try:
                embedded = await embedder.embed_chunks(all_chunks)
            except EmbedderError as e:
                log.error("embed failed for batch of %d chunks: %s", len(all_chunks), e)
                stats.resources_skipped += indexed_count
                stats.embed_errors += 1
                return

            sparse_vecs = await aencode_passages([c.content for c in all_chunks])
            sparse_by_id = {c.id: sv for c, sv in zip(all_chunks, sparse_vecs, strict=True)}
            n = await indexer.upsert(embedded, sparse_by_id=sparse_by_id)

            stats.chunks_produced += len(all_chunks)
            stats.chunks_indexed += n
            stats.resources_indexed += indexed_count

        async for resource in source.list_resources():
            stats.resources_seen += 1
            pending.append(resource)
            if len(pending) >= file_batch_size:
                await flush(pending)
                pending = []

        await flush(pending)
        return stats
    finally:
        await embedder.aclose()
        await enricher.aclose()
        await indexer.aclose()


async def run_query(
    *,
    product_id: str,
    text: str,
    config: NexusConfig,
    top_k: int = 10,
    mode: str = "auto",
) -> list[dict]:
    """Dense retrieval. BM25 + GraphRAG + rerank land in Slice 2/6."""
    embedder = EmbedderClient(base_url=config.models.embedding.url or "http://localhost:8080")
    indexer = Indexer(url=config.vector_store.url)
    try:
        vectors_to_search: list[str]
        if mode == "code":
            vectors_to_search = ["dense_code"]
        elif mode == "text":
            vectors_to_search = ["dense_text"]
        else:
            vectors_to_search = ["dense_code", "dense_text"]

        all_hits: list[dict] = []
        for v in vectors_to_search:
            qv = await embedder.embed_query(text, vector=v)  # type: ignore[arg-type]
            hits = await indexer.search_dense(
                product_id=product_id,
                query_vector=qv,
                vector_name=v,
                top_k=top_k,
            )
            for h in hits:
                h["vector_name"] = v
            all_hits.extend(hits)

        all_hits.sort(key=lambda h: h["score"], reverse=True)
        return all_hits[:top_k]
    finally:
        await embedder.aclose()
        await indexer.aclose()
