"""Incremental ingest path used by the continuous index daemon.

Given a (product_id, ResourceRef, content) tuple, this:

1. Asks the indexer for the existing point IDs at that resource URI.
2. Re-chunks the new content, enriches + embeds + sparse-encodes.
3. Upserts the fresh chunks FIRST (new points live before anything is removed).
4. Prunes only the stale old points (old IDs minus the fresh chunk IDs).
5. Optionally writes the resource manifest row last, so the registry only
   claims what the index actually holds.

Ordering is deliberate: a crash mid-way leaves old+new points coexisting for
one cycle (deduped downstream) instead of losing the resource from the index.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from anvay.ingest.chunker import chunk_resource
from anvay.ingest.embedder import EmbedderClient
from anvay.ingest.enricher import ContextualEnricher
from anvay.ingest.indexer import Indexer
from anvay.ingest.models import ResourceRef
from anvay.retrieval.sparse import aencode_passages

log = logging.getLogger(__name__)


@dataclass
class IncrementalResult:
    chunks_deleted: int
    chunks_indexed: int


async def reindex_resource(
    *,
    product_id: str,
    resource: ResourceRef,
    content: str,
    embedder: EmbedderClient,
    enricher: ContextualEnricher,
    indexer: Indexer,
    enrich: bool = True,
    registry: object | None = None,
    source_key: str = "",
    embedding_version: str = "",
) -> IncrementalResult:
    track_status = registry is not None and bool(source_key)
    if track_status:
        registry.mark_resource_index_pending(
            product_id,
            source_key,
            resource.uri,
            at=datetime.now(UTC).isoformat(),
        )

    old_buckets = await indexer.ids_by_resource(
        product_id=product_id, resource_uri=resource.uri
    )

    chunks = chunk_resource(product_id, resource, content)
    if enrich and chunks:
        chunks = await enricher.enrich(chunks, doc_contents={resource.uri: content})

    inserted = 0
    new_ids: set[str] = {c.id for c in chunks}
    if chunks:
        embedded = await embedder.embed_chunks(chunks)
        sparse_by_id = {}
        if getattr(indexer, "requires_sparse_vectors", True):
            sparse_vecs = await aencode_passages(
                [c.sparse_text_for_embedding() for c in chunks]
            )
            sparse_by_id = {c.id: sv for c, sv in zip(chunks, sparse_vecs, strict=True)}
        inserted = await indexer.upsert(embedded, sparse_by_id=sparse_by_id)

    stale = {
        coll: [pid for pid in ids if pid not in new_ids]
        for coll, ids in old_buckets.items()
    }
    stale = {coll: ids for coll, ids in stale.items() if ids}
    deleted = 0
    if stale:
        deleted = await indexer.delete_points_by_ids(stale)

    if track_status:
        _write_manifest(
            registry=registry,
            product_id=product_id,
            source_key=source_key,
            resource=resource,
            content=content,
            chunk_ids=sorted(new_ids),
            embedding_version=embedding_version,
        )

    log.info(
        "incremental %s: deleted=%d indexed=%d",
        resource.uri,
        deleted,
        inserted,
    )
    return IncrementalResult(chunks_deleted=deleted, chunks_indexed=inserted)


def _write_manifest(
    *,
    registry,
    product_id: str,
    source_key: str,
    resource: ResourceRef,
    content: str,
    chunk_ids: list[str],
    embedding_version: str,
) -> None:
    """Manifest write is last so the registry never claims un-indexed chunks."""
    now = datetime.now(UTC).isoformat()
    prior_rows = {
        row["resourceUri"]: row
        for row in registry.list_resource_manifests(product_id, source_key)
    }
    prior = prior_rows.get(resource.uri, {})
    registry.upsert_resource_manifest(
        {
            "product": product_id,
            "sourceKey": source_key,
            "resourceUri": resource.uri,
            "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "mime": resource.mime,
            "sizeBytes": resource.size_bytes,
            "lastSeenSync": prior.get("lastSeenSync", f"daemon:{now}"),
            "chunkIds": chunk_ids,
            "indexedAt": now,
            "embeddingVersion": embedding_version or prior.get("embeddingVersion", ""),
            "enrichmentVersion": prior.get("enrichmentVersion", ""),
            "enrichmentStatus": prior.get("enrichmentStatus", ""),
            "graphExtractionVersion": prior.get("graphExtractionVersion", ""),
            "graphStatus": prior.get("graphStatus", ""),
            "graphFactIds": prior.get("graphFactIds", []),
            "graphIndexedAt": prior.get("graphIndexedAt", ""),
            "indexStatus": "indexed",
            "indexStatusAt": now,
        }
    )
