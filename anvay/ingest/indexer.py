"""Qdrant indexer — named vectors, per-product payload isolation.

Two collections (per anvay.yaml `vector_store.collections`):

  anvay_code   stores code chunks; named vectors: {dense, bm25}
  anvay_text   stores doc chunks;  named vectors: {dense, bm25}

`dense`  — configured embedding model dimensionality, cosine
`bm25`   — fastembed Qdrant/bm25 sparse encoder; Qdrant applies IDF server-side

Tenant isolation via `product_id` payload filter on every query/scroll/delete.
A keyword index on `product_id` makes these filters fast.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import exceptions as qex
from qdrant_client.http import models as qm

from anvay.ingest.models import Chunk, EmbeddedChunk
from anvay.retrieval.sparse import SparseVector, aencode_query

# Default dense embedding dimensionality. Configurable per-collection on create.
DEFAULT_VECTOR_DIM = 2048

_CODE_COLLECTION = "anvay_code"
_TEXT_COLLECTION = "anvay_text"
_QDRANT_RETRY_DELAYS = (1.0, 3.0, 8.0)

log = logging.getLogger(__name__)
T = TypeVar("T")


class IndexerError(RuntimeError):
    pass


class SourceRefPayload(BaseModel):
    product_id: str
    source_key: str
    source_id: str
    resource_uri: str
    anchor: str
    start_line: int | None = None
    end_line: int | None = None


def _point_batches(
    points: Sequence[qm.PointStruct], batch_size: int
) -> list[Sequence[qm.PointStruct]]:
    return [points[i : i + batch_size] for i in range(0, len(points), batch_size)]


def _exception_detail(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    return str(exc) or repr(exc)


class Indexer:
    """Async Qdrant wrapper. Construct once per process."""

    requires_sparse_vectors = True

    def __init__(
        self,
        url: str = "http://localhost:6333",
        *,
        code_collection: str = _CODE_COLLECTION,
        text_collection: str = _TEXT_COLLECTION,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        timeout_s: int = 120,
        upsert_batch_size: int = 16,
        quantization_enabled: bool = True,
        quantization_type: str = "turboquant",
        quantization_bits: str = "bits4",
        quantization_always_ram: bool = True,
    ):
        self.client = AsyncQdrantClient(
            url=url, check_compatibility=False, timeout=timeout_s
        )
        self._code = code_collection
        self._text = text_collection
        self._dim = vector_dim
        self._upsert_batch_size = max(1, upsert_batch_size)
        self._quantization_enabled = quantization_enabled
        self._quantization_type = quantization_type
        self._quantization_bits = quantization_bits
        self._quantization_always_ram = quantization_always_ram

    async def aclose(self) -> None:
        await self.client.close()

    async def health(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------ setup

    async def ensure_collections(self) -> None:
        """Idempotent: create collections with named vectors + custom shard key."""
        collections = await self._retry_qdrant(
            "get_collections", self.client.get_collections
        )
        existing = {c.name for c in collections.collections}
        for name in (self._code, self._text):
            if name not in existing:
                await self._retry_qdrant(
                    f"create_collection:{name}",
                    lambda name=name: self.client.create_collection(
                        collection_name=name,
                        vectors_config={
                            "dense": qm.VectorParams(
                                size=self._dim,
                                distance=qm.Distance.COSINE,
                                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
                                quantization_config=self._dense_quantization_config(),
                            ),
                        },
                        sparse_vectors_config={
                            "bm25": qm.SparseVectorParams(
                                modifier=qm.Modifier.IDF,
                            ),
                        },
                    ),
                )
            await self._ensure_payload_indexes(name)

    # ------------------------------------------------------------ write

    async def upsert(
        self,
        embedded: Sequence[EmbeddedChunk],
        *,
        sparse_by_id: dict[str, SparseVector] | None = None,
        source_key: str | None = None,
        content_hash_by_id: dict[str, str] | None = None,
        graph_node_ids_by_id: dict[str, list[str]] | None = None,
        entity_ids_by_id: dict[str, list[str]] | None = None,
        source_ref_by_id: dict[str, SourceRefPayload] | None = None,
        citation_anchor_by_id: dict[str, str] | None = None,
        graph_extraction_version_by_id: dict[str, str] | None = None,
        artifact_type_by_id: dict[str, str] | None = None,
        neighbor_chunk_ids_by_id: dict[str, list[str]] | None = None,
        embedding_version: str | None = None,
        indexed_at: str | None = None,
    ) -> int:
        """Upsert dense + (optional) sparse vectors per chunk."""
        if not embedded:
            return 0
        sparse_by_id = sparse_by_id or {}
        content_hash_by_id = content_hash_by_id or {}
        graph_node_ids_by_id = graph_node_ids_by_id or {}
        entity_ids_by_id = entity_ids_by_id or {}
        source_ref_by_id = source_ref_by_id or {}
        citation_anchor_by_id = citation_anchor_by_id or {}
        graph_extraction_version_by_id = graph_extraction_version_by_id or {}
        artifact_type_by_id = artifact_type_by_id or {}
        neighbor_chunk_ids_by_id = neighbor_chunk_ids_by_id or {}
        buckets: dict[tuple[str, str], list[qm.PointStruct]] = {}
        for ec in embedded:
            coll = self._code if ec.vector_name == "dense_code" else self._text
            point = self._to_point(
                ec,
                sparse_by_id.get(ec.chunk.id),
                source_key=source_key,
                content_hash=content_hash_by_id.get(ec.chunk.id),
                graph_node_ids=graph_node_ids_by_id.get(ec.chunk.id),
                entity_ids=entity_ids_by_id.get(ec.chunk.id),
                source_ref=source_ref_by_id.get(ec.chunk.id),
                citation_anchor=citation_anchor_by_id.get(ec.chunk.id),
                graph_extraction_version=graph_extraction_version_by_id.get(ec.chunk.id),
                artifact_type=artifact_type_by_id.get(ec.chunk.id),
                neighbor_chunk_ids=neighbor_chunk_ids_by_id.get(ec.chunk.id),
                embedding_version=embedding_version,
                indexed_at=indexed_at,
            )
            buckets.setdefault((coll, ec.chunk.product_id), []).append(point)

        n = 0
        for (coll, _product_id), points in buckets.items():
            for batch in _point_batches(points, self._upsert_batch_size):
                await self._retry_qdrant(
                    f"upsert:{coll}",
                    lambda coll=coll, batch=batch: self.client.upsert(
                        collection_name=coll,
                        points=batch,
                    ),
                )
                n += len(batch)
        return n

    async def delete_points_by_ids(
        self, point_ids: Sequence[str] | dict[str, Sequence[str]]
    ) -> int:
        """Delete points by ID from one or both collections."""
        if isinstance(point_ids, dict):
            buckets = point_ids
        else:
            buckets = {self._code: point_ids, self._text: point_ids}

        deleted = 0
        for coll, ids in buckets.items():
            unique_ids = sorted(set(ids))
            if not unique_ids:
                continue
            await self._retry_qdrant(
                f"delete_points:{coll}",
                lambda coll=coll, unique_ids=unique_ids: self.client.delete(
                    collection_name=coll,
                    points_selector=qm.PointIdsList(points=list(unique_ids)),
                ),
            )
            deleted += len(unique_ids)
        return deleted

    async def update_payloads(
        self,
        chunks: Sequence[Chunk],
        *,
        graph_node_ids_by_id: dict[str, list[str]] | None = None,
        entity_ids_by_id: dict[str, list[str]] | None = None,
        source_ref_by_id: dict[str, SourceRefPayload] | None = None,
        citation_anchor_by_id: dict[str, str] | None = None,
        graph_extraction_version_by_id: dict[str, str] | None = None,
        artifact_type_by_id: dict[str, str] | None = None,
        neighbor_chunk_ids_by_id: dict[str, list[str]] | None = None,
    ) -> int:
        """Patch graph-derived payload metadata without rewriting vectors."""
        graph_node_ids_by_id = graph_node_ids_by_id or {}
        entity_ids_by_id = entity_ids_by_id or {}
        source_ref_by_id = source_ref_by_id or {}
        citation_anchor_by_id = citation_anchor_by_id or {}
        graph_extraction_version_by_id = graph_extraction_version_by_id or {}
        artifact_type_by_id = artifact_type_by_id or {}
        neighbor_chunk_ids_by_id = neighbor_chunk_ids_by_id or {}

        updated = 0
        for chunk in chunks:
            payload: dict[str, object] = {}
            if chunk.id in graph_node_ids_by_id:
                payload["graph_node_ids"] = graph_node_ids_by_id[chunk.id]
            if chunk.id in entity_ids_by_id:
                payload["entity_ids"] = entity_ids_by_id[chunk.id]
            if chunk.id in source_ref_by_id:
                payload["source_ref"] = source_ref_by_id[chunk.id].model_dump(mode="json")
            if chunk.id in citation_anchor_by_id:
                payload["citation_anchor"] = citation_anchor_by_id[chunk.id]
            if chunk.id in graph_extraction_version_by_id:
                payload["graph_extraction_version"] = graph_extraction_version_by_id[chunk.id]
            if chunk.id in artifact_type_by_id:
                payload["artifact_type"] = artifact_type_by_id[chunk.id]
            if chunk.id in neighbor_chunk_ids_by_id:
                payload["neighbor_chunk_ids"] = neighbor_chunk_ids_by_id[chunk.id]
            if not payload:
                continue
            collection = self._code if chunk.kind.value == "code" else self._text
            await self._retry_qdrant(
                f"set_payload:{collection}",
                lambda collection=collection, payload=payload, chunk_id=chunk.id: self.client.set_payload(
                    collection_name=collection,
                    payload=payload,
                    points=[chunk_id],
                ),
            )
            updated += 1
        return updated

    # ------------------------------------------------------------ read

    async def search_dense(
        self,
        *,
        product_id: str,
        query_vector: list[float],
        vector_name: str,
        top_k: int = 50,
    ) -> list[dict]:
        coll = self._code if vector_name == "dense_code" else self._text
        return await self._search(
            collection=coll,
            product_id=product_id,
            query=query_vector,
            using="dense",
            top_k=top_k,
        )

    async def search_sparse(
        self,
        *,
        product_id: str,
        sparse: SparseVector | None = None,
        query: str | None = None,
        vector_kind: str,
        top_k: int = 50,
    ) -> list[dict]:
        """Sparse BM25 search. `vector_kind` is 'code' or 'text'."""
        if sparse is None:
            if query is None:
                raise ValueError("search_sparse requires sparse or query")
            sparse = await aencode_query(query)
        coll = self._code if vector_kind == "code" else self._text
        return await self._search(
            collection=coll,
            product_id=product_id,
            query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
            using="bm25",
            top_k=top_k,
        )

    async def search_by_graph_nodes(
        self,
        *,
        product_id: str,
        graph_node_ids: Sequence[str],
        vector_kind: str | None = None,
        top_k: int = 50,
    ) -> list[dict]:
        """Return chunks directly attached to graph nodes, product-scoped."""
        ids = sorted({gid for gid in graph_node_ids if gid})
        if not ids:
            return []
        collections = []
        if vector_kind in (None, "code"):
            collections.append(self._code)
        if vector_kind in (None, "text"):
            collections.append(self._text)
        hits: list[dict] = []
        for coll in collections:
            points, _ = await self._retry_qdrant(
                f"scroll_graph_nodes:{coll}",
                lambda coll=coll: self.client.scroll(
                    collection_name=coll,
                    scroll_filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="product_id",
                                match=qm.MatchValue(value=product_id),
                            ),
                            qm.FieldCondition(
                                key="graph_node_ids",
                                match=qm.MatchAny(any=ids),
                            ),
                        ]
                    ),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            hits.extend(
                {
                    "id": pt.id,
                    "score": 1.0,
                    "payload": pt.payload,
                    "collection": coll,
                }
                for pt in points
            )
        return hits[:top_k]

    async def _search(
        self,
        *,
        collection: str,
        product_id: str,
        query,
        using: str,
        top_k: int,
    ) -> list[dict]:
        result = await self._retry_qdrant(
            f"query_points:{collection}",
            lambda: self.client.query_points(
                collection_name=collection,
                query=query,
                using=using,
                limit=top_k,
                query_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="product_id", match=qm.MatchValue(value=product_id)
                        )
                    ]
                ),
                with_payload=True,
            ),
        )
        return [
            {"id": pt.id, "score": pt.score, "payload": pt.payload}
            for pt in result.points
        ]

    async def existing_point_ids(
        self, ids: Sequence[str], *, batch_size: int = 500
    ) -> set[str]:
        """Which of `ids` actually exist (in either collection). For manifest reconciliation."""
        unique_ids = sorted(set(ids))
        found: set[str] = set()
        for coll in (self._code, self._text):
            for i in range(0, len(unique_ids), batch_size):
                batch = unique_ids[i : i + batch_size]
                points = await self._retry_qdrant(
                    f"retrieve:{coll}",
                    lambda coll=coll, batch=batch: self.client.retrieve(
                        collection_name=coll,
                        ids=batch,
                        with_payload=False,
                        with_vectors=False,
                    ),
                )
                found.update(str(p.id) for p in points)
        return found

    async def count(self, *, product_id: str, vector_kind: str) -> int:
        coll = self._code if vector_kind == "code" else self._text
        res = await self._retry_qdrant(
            f"count:{coll}",
            lambda: self.client.count(
                collection_name=coll,
                count_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="product_id", match=qm.MatchValue(value=product_id)
                        )
                    ]
                ),
                exact=True,
            ),
        )
        return res.count

    async def iter_chunk_payloads(
        self,
        *,
        product_id: str,
        vector_kind: str,
        batch_size: int = 256,
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yield indexed chunk payloads for one product/kind without vectors."""
        coll = self._code if vector_kind == "code" else self._text
        product_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                )
            ]
        )
        offset = None
        while True:
            points, offset = await self._retry_qdrant(
                f"scroll:{coll}",
                lambda offset=offset: self.client.scroll(
                    collection_name=coll,
                    scroll_filter=product_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            for point in points:
                yield str(point.id), point.payload or {}
            if offset is None:
                break

    async def ids_by_resource(
        self, *, product_id: str, resource_uri: str
    ) -> dict[str, list[str]]:
        """All point IDs for a resource, keyed by collection name. Paginates the
        scroll so resources with >1024 points are fully covered."""
        resource_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                ),
                qm.FieldCondition(
                    key="resource_uri",
                    match=qm.MatchValue(value=resource_uri),
                ),
            ]
        )
        found: dict[str, list[str]] = {}
        for coll in (self._code, self._text):
            ids: list[str] = []
            offset = None
            while True:
                scrolled, offset = await self._retry_qdrant(
                    f"scroll_resource_ids:{coll}",
                    lambda coll=coll, offset=offset: self.client.scroll(
                        collection_name=coll,
                        scroll_filter=resource_filter,
                        limit=1024,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    ),
                )
                ids.extend(str(pt.id) for pt in scrolled)
                if offset is None:
                    break
            if ids:
                found[coll] = ids
        return found

    async def chunks_at_anchors(
        self,
        *,
        product_id: str,
        resource_uris: Sequence[str],
    ) -> dict[str, list[dict]]:
        """All chunk payloads for the given files, grouped by resource_uri.

        Callers verify (start_line ≤ line ≤ end_line) client-side — one scroll
        per file bounds the round-trips for citation verification."""
        out: dict[str, list[dict]] = {}
        for uri in dict.fromkeys(resource_uris):
            if not uri:
                continue
            file_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="product_id", match=qm.MatchValue(value=product_id)
                    ),
                    qm.FieldCondition(
                        key="resource_uri", match=qm.MatchValue(value=uri)
                    ),
                ]
            )
            rows: list[dict] = []
            for coll in (self._code, self._text):
                offset = None
                while True:
                    points, offset = await self._retry_qdrant(
                        f"scroll_anchor:{coll}",
                        lambda coll=coll, offset=offset, flt=file_filter: self.client.scroll(
                            collection_name=coll,
                            scroll_filter=flt,
                            limit=512,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        ),
                    )
                    rows.extend((pt.payload or {}) for pt in points)
                    if offset is None:
                        break
            if rows:
                out[uri] = rows
        return out

    async def chunks_by_symbol_ids(
        self,
        *,
        product_id: str,
        symbol_ids: Sequence[str],
        limit_per_symbol: int = 4,
    ) -> list[dict]:
        """All chunks sharing any of the given symbol_ids (declaration chunk,
        doc spill, split sub-chunks). One batched scroll per collection."""
        unique = [s for s in dict.fromkeys(symbol_ids) if s]
        if not unique:
            return []
        sym_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                ),
                qm.FieldCondition(
                    key="symbol_id", match=qm.MatchAny(any=list(unique))
                ),
            ]
        )
        out: list[dict] = []
        per_symbol: dict[str, int] = {}
        for coll in (self._code, self._text):
            points, _ = await self._retry_qdrant(
                f"scroll_symbols:{coll}",
                lambda coll=coll: self.client.scroll(
                    collection_name=coll,
                    scroll_filter=sym_filter,
                    limit=max(len(unique) * limit_per_symbol, 16),
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            for pt in points:
                payload = pt.payload or {}
                sym = str(payload.get("symbol_id") or "")
                if per_symbol.get(sym, 0) >= limit_per_symbol:
                    continue
                per_symbol[sym] = per_symbol.get(sym, 0) + 1
                out.append({"id": str(pt.id), "payload": payload})
        return out

    async def chunks_by_ids(
        self,
        *,
        product_id: str,
        chunk_ids: Sequence[str],
        limit: int | None = None,
    ) -> list[dict]:
        """Fetch chunk points by id from both collections, product-scoped.

        Used by depth-1 graph-local retrieval to pull neighbor chunks named in
        the `neighbor_chunk_ids` payload directly, without a graph traverse.
        Ids absent from the index (e.g. stale neighbors after a neighbor
        reindex changed its uuid5 id) are silently dropped."""
        unique = [cid for cid in dict.fromkeys(chunk_ids) if cid]
        if not unique:
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for coll in (self._code, self._text):
            points = await self._retry_qdrant(
                f"retrieve_chunks:{coll}",
                lambda coll=coll: self.client.retrieve(
                    collection_name=coll,
                    ids=list(unique),
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            for pt in points:
                pid = str(pt.id)
                payload = pt.payload or {}
                if pid in seen or payload.get("product_id") != product_id:
                    continue
                seen.add(pid)
                out.append(
                    {"id": pid, "score": 1.0, "payload": payload, "collection": coll}
                )
        return out[:limit] if limit else out

    async def delete_by_resource(
        self, *, product_id: str, resource_uri: str
    ) -> list[str]:
        """Delete all points for a resource. Returns the deleted IDs for cache
        invalidation hooks."""
        buckets = await self.ids_by_resource(
            product_id=product_id, resource_uri=resource_uri
        )
        if not buckets:
            return []
        await self.delete_points_by_ids(buckets)
        return [pid for ids in buckets.values() for pid in ids]

    async def delete_by_product(self, *, product_id: str) -> dict[str, int]:
        """Delete all points for a product from code/text collections."""
        counts: dict[str, int] = {}
        existing = await self._existing_collections()
        product_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                )
            ]
        )
        for coll in (self._code, self._text):
            if coll not in existing:
                counts[coll] = 0
                continue
            before = await self._retry_qdrant(
                f"count_delete_product:{coll}",
                lambda coll=coll: self.client.count(
                    collection_name=coll,
                    count_filter=product_filter,
                    exact=True,
                ),
            )
            if before.count:
                await self._retry_qdrant(
                    f"delete_product:{coll}",
                    lambda coll=coll: self.client.delete(
                        collection_name=coll,
                        points_selector=qm.FilterSelector(filter=product_filter),
                    ),
                )
            counts[coll] = before.count
        return counts

    async def count_by_product(self, *, product_id: str) -> dict[str, int]:
        """Count product points in code/text collections.

        Missing collections are empty from the product deletion perspective.
        """
        counts: dict[str, int] = {}
        existing = await self._existing_collections()
        product_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                )
            ]
        )
        for coll in (self._code, self._text):
            if coll not in existing:
                counts[coll] = 0
                continue
            before = await self._retry_qdrant(
                f"count_product:{coll}",
                lambda coll=coll: self.client.count(
                    collection_name=coll,
                    count_filter=product_filter,
                    exact=True,
                ),
            )
            counts[coll] = before.count
        return counts

    async def _existing_collections(self) -> set[str]:
        collections = await self._retry_qdrant(
            "get_collections", self.client.get_collections
        )
        return {collection.name for collection in collections.collections}

    async def _retry_qdrant(
        self, operation: str, call: Callable[[], Awaitable[T]]
    ) -> T:
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_QDRANT_RETRY_DELAYS, None), start=1):
            try:
                return await call()
            except (httpx.HTTPError, qex.ResponseHandlingException) as e:
                last_exc = e
                if delay is None:
                    break
                detail = _exception_detail(e)
                log.warning(
                    "qdrant %s attempt %d failed; retrying in %.0fs: %s",
                    operation,
                    attempt,
                    delay,
                    detail,
                )
                await asyncio.sleep(delay)
        raise IndexerError(
            f"qdrant {operation} failed after retries: {_exception_detail(last_exc)}"
        ) from last_exc

    async def _ensure_payload_indexes(self, collection: str) -> None:
        indexes = {
            "product_id": qm.PayloadSchemaType.KEYWORD,
            "resource_uri": qm.PayloadSchemaType.KEYWORD,
            "graph_node_ids": qm.PayloadSchemaType.KEYWORD,
            "entity_ids": qm.PayloadSchemaType.KEYWORD,
            "artifact_type": qm.PayloadSchemaType.KEYWORD,
            "neighbor_chunk_ids": qm.PayloadSchemaType.KEYWORD,
            "source_ref.resource_uri": qm.PayloadSchemaType.KEYWORD,
            "symbol_id": qm.PayloadSchemaType.KEYWORD,
            "start_line": qm.PayloadSchemaType.INTEGER,
            "end_line": qm.PayloadSchemaType.INTEGER,
        }
        for field_name, field_schema in indexes.items():
            try:
                await self._retry_qdrant(
                    "create_payload_index",
                    lambda collection=collection, field_name=field_name, field_schema=field_schema: self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=field_schema,
                    ),
                )
            except Exception as e:
                if "already" not in str(e).lower():
                    raise

    # ------------------------------------------------------------ helpers

    def _dense_quantization_config(self) -> qm.QuantizationConfig | None:
        if not self._quantization_enabled:
            return None
        if self._quantization_type.lower() not in {"turboquant", "turbo"}:
            raise IndexerError(
                f"unsupported vector_store.quantization.type: {self._quantization_type}"
            )
        bits_by_name = {
            "bits1": qm.TurboQuantBitSize.BITS1,
            "bits1_5": qm.TurboQuantBitSize.BITS1_5,
            "bits2": qm.TurboQuantBitSize.BITS2,
            "bits4": qm.TurboQuantBitSize.BITS4,
        }
        try:
            bits = bits_by_name[self._quantization_bits.lower()]
        except KeyError as e:
            raise IndexerError(
                "unsupported vector_store.quantization.bits: "
                f"{self._quantization_bits}; expected one of "
                f"{', '.join(sorted(bits_by_name))}"
            ) from e
        return qm.TurboQuantization(
            turbo=qm.TurboQuantQuantizationConfig(
                always_ram=self._quantization_always_ram,
                bits=bits,
            )
        )

    def _to_point(
        self,
        ec: EmbeddedChunk,
        sparse: SparseVector | None,
        *,
        source_key: str | None = None,
        content_hash: str | None = None,
        graph_node_ids: list[str] | None = None,
        entity_ids: list[str] | None = None,
        source_ref: SourceRefPayload | None = None,
        citation_anchor: str | None = None,
        graph_extraction_version: str | None = None,
        artifact_type: str | None = None,
        neighbor_chunk_ids: list[str] | None = None,
        embedding_version: str | None = None,
        indexed_at: str | None = None,
    ) -> qm.PointStruct:
        c: Chunk = ec.chunk
        vectors: dict[str, object] = {"dense": ec.vector}
        if sparse is not None and sparse.indices:
            vectors["bm25"] = qm.SparseVector(
                indices=sparse.indices, values=sparse.values
            )
        return qm.PointStruct(
            id=c.id,
            vector=vectors,
            payload={
                "product_id": c.product_id,
                "resource_uri": c.resource.uri,
                "source_id": c.resource.source_id,
                "source_key": source_key,
                "content_hash": content_hash,
                "embedding_version": embedding_version,
                "indexed_at": indexed_at,
                "mime": c.resource.mime,
                "kind": c.kind.value,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "context_path": c.context_path,
                "symbol_id": c.symbol_id,
                "signature": c.signature,
                "content": c.content,
                "graph_node_ids": graph_node_ids or [],
                "entity_ids": entity_ids or [],
                "source_ref": source_ref.model_dump(mode="json") if source_ref else {},
                "citation_anchor": citation_anchor or c.anchor,
                "graph_extraction_version": graph_extraction_version,
                "artifact_type": artifact_type or c.kind.value,
                "neighbor_chunk_ids": neighbor_chunk_ids or [],
            },
        )
