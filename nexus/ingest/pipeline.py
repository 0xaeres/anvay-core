"""End-to-end ingestion pipeline orchestrator.

Pulls resources from a source, chunks them, optionally enriches with contextual
summaries, embeds, and upserts into the configured retrieval index.

Design:
- Files are collected into batches of FILE_BATCH_SIZE before any embedding call.
- Within each batch, reads are concurrent (READ_CONCURRENCY semaphore).
- A bad file is skipped; it does not abort the batch or the run.
- The embedder is called once per batch (not once per file).
- Transient embedder errors are retried with exponential backoff in EmbedderClient.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from nexus.config import NexusConfig
from nexus.ingest.chunker import chunk_resource
from nexus.ingest.embedder import EmbedderClient, EmbedderError
from nexus.ingest.enricher import ContextualEnricher
from nexus.ingest.indexer_factory import create_indexer
from nexus.ingest.models import Chunk, ResourceRef
from nexus.retrieval.sparse import aencode_passages

log = logging.getLogger(__name__)

IngestEventSink = Callable[[dict], Awaitable[None]]


class _Source(Protocol):
    source_id: str

    async def list_resources(self): ...
    async def read_resource(self, resource: ResourceRef) -> str: ...


class _Registry(Protocol):
    def list_resource_manifests(self, product_id: str, source_key: str) -> list[dict]: ...
    def upsert_resource_manifest(self, row: dict) -> None: ...
    def delete_resource_manifest(
        self, product_id: str, source_key: str, resource_uri: str
    ) -> bool: ...


@dataclass
class IngestStats:
    resources_seen: int = 0
    resources_indexed: int = 0
    resources_skipped: int = 0
    resources_failed: int = 0
    chunks_produced: int = 0
    chunks_indexed: int = 0
    embed_errors: int = 0  # batches that failed to embed (token limit, server error)
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0


@dataclass
class _ResourcePayload:
    ref: ResourceRef
    content: str
    content_hash: str
    prior: dict | None
    action: str


def embedding_version(config: NexusConfig) -> str:
    """Hash embedding-affecting config. Change => full source re-embed."""
    payload = {
        "embedding_provider": config.models.embedding.provider,
        "embedding_model": config.models.embedding.model,
        "embedding_url": config.models.embedding.url,
        "embedding_base_url": config.models.embedding.base_url,
        "embedding_dim": config.models.embedding.dim,
        "embedding_instruction_profile": config.models.embedding.instruction_profile,
        "vector_quantization_enabled": config.vector_store.quantization.enabled,
        "vector_quantization_type": config.vector_store.quantization.type,
        "vector_quantization_bits": config.vector_store.quantization.bits,
        "reranker_provider": config.models.reranker.provider,
        "reranker_model": config.models.reranker.model,
        "reranker_url": config.models.reranker.url,
        "reranker_base_url": config.models.reranker.base_url,
        "quality_gate_threshold": config.ingestion.quality_gate_threshold,
        "light_provider": config.models.light.provider,
        "light_model": config.models.light.model,
        "light_base_url": config.models.light.base_url,
        "enrich_code": config.ingestion.enrich_chunks.code,
        "enrich_docs": config.ingestion.enrich_chunks.docs,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


async def run_ingest(
    *,
    product_id: str,
    source: _Source,
    config: NexusConfig,
    enrich: bool = True,
    event_sink: IngestEventSink | None = None,
    registry: _Registry | None = None,
    source_key: str | None = None,
) -> IngestStats:
    """One pass: discover → batch-read → chunk → enrich → embed → index."""
    stats = IngestStats()

    file_batch_size = config.ingestion.file_batch_size
    read_concurrency = config.ingestion.read_concurrency

    embedder = EmbedderClient.from_cfg(
        config.models.embedding,
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
    indexer = create_indexer(config)
    batch_no = 0
    sync_id = _utc_now()
    version = embedding_version(config)
    manifest_by_uri: dict[str, dict] = {}
    current_uris: set[str] = set()
    delta_enabled = registry is not None and source_key is not None

    async def emit(level: str, stage: str, msg: str, **extra) -> None:
        event = {"level": level, "stage": stage, "msg": msg, **extra}
        log.info("ingest.%s product=%s source=%s %s", stage, product_id, source.source_id, msg)
        if event_sink is not None:
            await event_sink(event)

    try:
        await emit("stage", "prepare", "Ensuring vector collections exist")
        await indexer.ensure_collections()
        if delta_enabled:
            manifest_by_uri = {
                row["resourceUri"]: row
                for row in registry.list_resource_manifests(product_id, source_key)
            }
            await emit(
                "stage",
                "manifest",
                f"Loaded {len(manifest_by_uri)} manifest resource(s)",
                source_key=source_key,
                resources=len(manifest_by_uri),
                embedding_version=version,
            )
        await emit("stage", "discover", "Discovering resources")

        pending: list[_ResourcePayload] = []

        async def flush(items: list[_ResourcePayload]) -> None:
            nonlocal batch_no
            if not items:
                return
            batch_no += 1
            batch_id = batch_no
            await emit(
                "stage",
                "chunk",
                f"Chunking batch {batch_id}: {len(items)} changed resource(s)",
                batch=batch_id,
                resources=len(items),
            )

            all_chunks: list[Chunk] = []
            doc_contents: dict[str, str] = {}
            chunks_by_uri: dict[str, list[Chunk]] = {}
            payload_by_uri = {item.ref.uri: item for item in items}
            for item in items:
                chunks = chunk_resource(product_id, item.ref, item.content)
                if not chunks:
                    await emit(
                        "debug",
                        "chunk",
                        f"No chunks produced for {item.ref.uri}",
                        batch=batch_id,
                        uri=item.ref.uri,
                    )
                    stats.resources_skipped += 1
                    continue
                all_chunks.extend(chunks)
                chunks_by_uri[item.ref.uri] = chunks
                doc_contents[item.ref.uri] = item.content

            if not all_chunks:
                await emit(
                    "stage",
                    "chunk",
                    f"Batch {batch_id} produced no chunks",
                    batch=batch_id,
                    chunks=0,
                )
                return

            code_chunks = sum(1 for c in all_chunks if c.kind.value == "code")
            doc_chunks = len(all_chunks) - code_chunks
            await emit(
                "stage",
                "chunk",
                (
                    f"Batch {batch_id} produced {len(all_chunks)} chunk(s) "
                    f"({code_chunks} code, {doc_chunks} docs)"
                ),
                batch=batch_id,
                chunks=len(all_chunks),
                code_chunks=code_chunks,
                doc_chunks=doc_chunks,
            )

            if enrich:
                await emit(
                    "stage",
                    "enrich",
                    f"Enriching batch {batch_id}: {len(all_chunks)} chunk(s)",
                    batch=batch_id,
                    chunks=len(all_chunks),
                )
                all_chunks = await enricher.enrich(all_chunks, doc_contents=doc_contents)
                enriched = sum(1 for c in all_chunks if c.context_summary)
                await emit(
                    "stage",
                    "enrich",
                    f"Batch {batch_id} enriched {enriched}/{len(all_chunks)} chunk(s)",
                    batch=batch_id,
                    chunks=len(all_chunks),
                    enriched=enriched,
                )
            else:
                await emit(
                    "stage",
                    "enrich",
                    f"Skipping enrichment for batch {batch_id}",
                    batch=batch_id,
                    chunks=len(all_chunks),
                )

            try:
                await emit(
                    "stage",
                    "embed",
                    f"Embedding dense vectors for batch {batch_id}: {len(all_chunks)} chunk(s)",
                    batch=batch_id,
                    chunks=len(all_chunks),
                )
                embedded = await embedder.embed_chunks(all_chunks)
            except EmbedderError as e:
                log.error("embed failed for batch of %d chunks: %s", len(all_chunks), e)
                await emit(
                    "error",
                    "embed",
                    f"Embedding failed for batch {batch_id}: {e}",
                    batch=batch_id,
                    chunks=len(all_chunks),
                )
                stats.resources_failed += len(items)
                stats.embed_errors += 1
                return
            await emit(
                "stage",
                "embed",
                f"Batch {batch_id} dense embedding complete",
                batch=batch_id,
                chunks=len(embedded),
            )

            sparse_by_id = {}
            if getattr(indexer, "requires_sparse_vectors", True):
                await emit(
                    "stage",
                    "sparse",
                    f"Encoding BM25 sparse vectors for batch {batch_id}",
                    batch=batch_id,
                    chunks=len(all_chunks),
                )
                sparse_vecs = await aencode_passages(
                    [c.text_for_embedding() for c in all_chunks]
                )
                sparse_by_id = {
                    c.id: sv for c, sv in zip(all_chunks, sparse_vecs, strict=True)
                }
            content_hash_by_id = {
                c.id: payload_by_uri[c.resource.uri].content_hash for c in all_chunks
            }
            await emit(
                "stage",
                "upsert",
                f"Upserting batch {batch_id} into vector store",
                batch=batch_id,
                chunks=len(embedded),
            )
            indexed_at = _utc_now()
            n = await indexer.upsert(
                embedded,
                sparse_by_id=sparse_by_id,
                source_key=source_key,
                content_hash_by_id=content_hash_by_id,
                embedding_version=version,
                indexed_at=indexed_at,
            )

            indexed_resources = 0
            for uri, chunks in chunks_by_uri.items():
                item = payload_by_uri[uri]
                new_ids = [c.id for c in chunks]
                old_ids = item.prior.get("chunkIds", []) if item.prior else []
                stale_ids = sorted(set(old_ids) - set(new_ids))
                if stale_ids:
                    await emit(
                        "stage",
                        "cleanup_stale",
                        f"Deleting {len(stale_ids)} stale chunk(s) for {uri}",
                        batch=batch_id,
                        uri=uri,
                        chunks=len(stale_ids),
                    )
                    await indexer.delete_points_by_ids(stale_ids)
                if delta_enabled:
                    registry.upsert_resource_manifest(
                        {
                            "product": product_id,
                            "sourceKey": source_key,
                            "resourceUri": uri,
                            "contentHash": item.content_hash,
                            "mime": item.ref.mime,
                            "sizeBytes": item.ref.size_bytes,
                            "lastSeenSync": sync_id,
                            "chunkIds": new_ids,
                            "indexedAt": indexed_at,
                            "embeddingVersion": version,
                        }
                    )
                    await emit(
                        "stage",
                        "manifest_update",
                        f"Manifest updated for {uri}",
                        batch=batch_id,
                        uri=uri,
                        chunks=len(new_ids),
                    )
                indexed_resources += 1

            stats.chunks_produced += len(all_chunks)
            stats.chunks_indexed += n
            stats.resources_indexed += indexed_resources
            await emit(
                "stage",
                "upsert",
                f"Batch {batch_id} indexed {n} chunk(s)",
                batch=batch_id,
                chunks_indexed=n,
                resources_indexed=indexed_resources,
            )

        sem = asyncio.Semaphore(read_concurrency)

        async def read_and_classify(r: ResourceRef) -> _ResourcePayload | None:
            async with sem:
                try:
                    content = await source.read_resource(r)
                except OSError as e:
                    log.debug("skipping %s: %s", r.uri, e)
                    await emit("warn", "read", f"Skipping unreadable resource: {r.uri} ({e})", uri=r.uri)
                    stats.resources_skipped += 1
                    return None

            digest = _content_hash(content)
            prior = manifest_by_uri.get(r.uri) if delta_enabled else None
            if prior and prior["contentHash"] == digest and prior["embeddingVersion"] == version:
                stats.unchanged += 1
                await emit("stage", "skip", f"Unchanged: {r.uri}", uri=r.uri)
                return None

            action = "added" if prior is None else "updated"
            if action == "added":
                stats.added += 1
            else:
                stats.updated += 1
            await emit("stage", "diff", f"{action.title()}: {r.uri}", action=action, uri=r.uri)
            return _ResourcePayload(
                ref=r,
                content=content,
                content_hash=digest,
                prior=prior,
                action=action,
            )

        resource_batch: list[ResourceRef] = []

        async def flush_reads(resources: list[ResourceRef]) -> None:
            if not resources:
                return
            await emit(
                "stage",
                "read",
                f"Reading {len(resources)} resource(s) for diff",
                resources=len(resources),
            )
            items = await asyncio.gather(*[read_and_classify(r) for r in resources])
            for item in items:
                if item is not None:
                    pending.append(item)
                    if len(pending) >= file_batch_size:
                        await flush(pending)
                        pending.clear()

        async for resource in source.list_resources():
            stats.resources_seen += 1
            current_uris.add(resource.uri)
            resource_batch.append(resource)
            if len(resource_batch) >= file_batch_size:
                await flush_reads(resource_batch)
                resource_batch = []

        await flush_reads(resource_batch)
        await flush(pending)

        if delta_enabled:
            removed_rows = [
                row for uri, row in manifest_by_uri.items() if uri not in current_uris
            ]
            for row in removed_rows:
                uri = row["resourceUri"]
                chunk_ids = row.get("chunkIds", [])
                try:
                    await emit(
                        "stage",
                        "delete_removed",
                        f"Deleting removed resource: {uri}",
                        uri=uri,
                        chunks=len(chunk_ids),
                    )
                    await indexer.delete_points_by_ids(chunk_ids)
                    registry.delete_resource_manifest(product_id, source_key, uri)
                    stats.removed += 1
                except Exception as e:
                    stats.resources_failed += 1
                    log.exception("delete removed resource failed for %s", uri)
                    await emit(
                        "error",
                        "delete_removed",
                        f"Failed deleting removed resource {uri}: {e}",
                        uri=uri,
                    )

            await emit(
                "stage",
                "diff",
                (
                    f"Delta summary: added={stats.added}, updated={stats.updated}, "
                    f"removed={stats.removed}, unchanged={stats.unchanged}"
                ),
                added=stats.added,
                updated=stats.updated,
                removed=stats.removed,
                unchanged=stats.unchanged,
                failed=stats.resources_failed,
            )
        await emit(
            "stage",
            "complete",
            (
                f"Ingest complete: {stats.resources_indexed} resource(s), "
                f"{stats.chunks_indexed} chunk(s)"
            ),
            resources_indexed=stats.resources_indexed,
            resources_skipped=stats.resources_skipped,
            resources_failed=stats.resources_failed,
            chunks_indexed=stats.chunks_indexed,
            embed_errors=stats.embed_errors,
            added=stats.added,
            updated=stats.updated,
            removed=stats.removed,
            unchanged=stats.unchanged,
        )
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
    """Legacy dense-only query helper; the production path is retrieval.pipeline.retrieve()."""
    embedder = EmbedderClient.from_cfg(config.models.embedding)
    indexer = create_indexer(config)
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
