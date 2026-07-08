"""Retrieval pipeline: dense + sparse -> RRF merge -> rerank.

Three stages, no fallbacks beyond rerank-soft-fail. Add complexity only when
an eval set proves it moves the number.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from anvay.config import AnvayConfig
from anvay.ingest.embedder import EmbedderClient
from anvay.ingest.indexer_factory import create_indexer
from anvay.retrieval.hybrid import Hit, rrf_merge
from anvay.retrieval.reranker import RerankerClient
from anvay.retrieval.sparse import aencode_query

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    reranked: bool = True
    seed_count: int = 0
    filtered_by_gate: int = 0
    best_score_before_gate: float | None = None


@dataclass
class RetrievalContext:
    embedder: EmbedderClient
    indexer: Any
    reranker: RerankerClient
    config: AnvayConfig

    @classmethod
    def from_config(cls, config: AnvayConfig) -> RetrievalContext:
        return cls(
            embedder=EmbedderClient.from_cfg(config.models.embedding),
            indexer=create_indexer(config),
            reranker=RerankerClient.from_cfg(config.models.reranker),
            config=config,
        )

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await self.reranker.aclose()
        await self.indexer.aclose()


async def retrieve(
    *,
    ctx: RetrievalContext,
    product_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["auto", "code", "text"] = "auto",
    graph_node_ids: Sequence[str] | None = None,
) -> RetrievalResult:
    """Run the pipeline. Caller is responsible for `ctx.aclose()`."""
    if mode == "code":
        vector_kinds = ["code"]
    elif mode == "text":
        vector_kinds = ["text"]
    else:
        vector_kinds = ["code", "text"]

    query_vectors = await _embed_query_vectors(ctx, query, vector_kinds)

    seed_set = await _hybrid_search(
        ctx=ctx,
        product_id=product_id,
        query_vectors=query_vectors,
        sparse_query=query,
        vector_kinds=vector_kinds,
        graph_node_ids=graph_node_ids,
    )

    if not seed_set:
        return RetrievalResult(hits=[], reranked=False, seed_count=0)

    reranked = True
    try:
        rerank_inputs = [_to_doc(h) for h in seed_set]
        ranking = await ctx.reranker.rerank(query, rerank_inputs, top_k=top_k)
        final_hits = [
            Hit(
                id=seed_set[r.index].id,
                score=r.score,
                payload=seed_set[r.index].payload,
                source=seed_set[r.index].source,
            )
            for r in ranking
        ]
    except Exception as e:
        log.warning("rerank failed, falling back to fused order: %s", e)
        final_hits = seed_set[:top_k]
        reranked = False

    gate = ctx.config.ingestion.quality_gate_threshold
    if reranked:
        final_hits, filtered_by_gate, best_score = _apply_quality_gate(final_hits, gate)
    else:
        filtered_by_gate = 0
        best_score = max((h.score for h in final_hits), default=None)

    return RetrievalResult(
        hits=final_hits,
        reranked=reranked,
        seed_count=len(seed_set),
        filtered_by_gate=filtered_by_gate,
        best_score_before_gate=best_score,
    )


async def _hybrid_search(
    *,
    ctx: RetrievalContext,
    product_id: str,
    query_vectors: dict[str, list[float]],
    sparse_query: str,
    vector_kinds: list[str],
    graph_node_ids: Sequence[str] | None = None,
) -> list[Hit]:
    """Dense + BM25 per modality, then RRF fuse to top-20 seed set."""
    sparse_vec = None
    if getattr(ctx.indexer, "requires_sparse_vectors", True):
        sparse_vec = await aencode_query(sparse_query)

    async def _dense(kind: str) -> list[Hit]:
        name = "dense_code" if kind == "code" else "dense_text"
        raw = await ctx.indexer.search_dense(
            product_id=product_id,
            query_vector=query_vectors[kind],
            vector_name=name,
            top_k=50,
        )
        return [
            Hit(id=r["id"], score=r["score"], payload=r["payload"] or {}, source="dense")
            for r in raw
        ]

    async def _sparse(kind: str) -> list[Hit]:
        raw = await ctx.indexer.search_sparse(
            product_id=product_id,
            sparse=sparse_vec,
            query=sparse_query,
            vector_kind=kind,
            top_k=50,
        )
        return [
            Hit(id=r["id"], score=r["score"], payload=r["payload"] or {}, source="bm25")
            for r in raw
        ]

    async def _graph(kind: str) -> list[Hit]:
        if not graph_node_ids or not hasattr(ctx.indexer, "search_by_graph_nodes"):
            return []
        raw = await ctx.indexer.search_by_graph_nodes(
            product_id=product_id,
            graph_node_ids=list(graph_node_ids),
            vector_kind=kind,
            top_k=50,
        )
        return [
            Hit(id=r["id"], score=r["score"], payload=r["payload"] or {}, source="graph")
            for r in raw
        ]

    dense_w, sparse_w = _shape_weights(sparse_query)
    rankings: list[list[Hit]] = []
    weights: list[float] = []
    for kind in vector_kinds:
        d, s, g = await asyncio.gather(_dense(kind), _sparse(kind), _graph(kind))
        rankings.extend([d, s, g])
        weights.extend([dense_w, sparse_w, 1.0])
    return rrf_merge(rankings, top_k=20, weights=weights)


# Symbol/path shape: identifiers, dotted paths, file extensions, snake/camelCase.
_SYMBOL_SHAPE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"       # a.b dotted
    r"|[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|sql|ya?ml|toml|md)\b"  # file paths
    r"|[a-z0-9]+_[a-z0-9_]+"                                  # snake_case
    r"|[a-z]+[A-Z][A-Za-z0-9]*"                               # camelCase
)


def _shape_weights(query: str) -> tuple[float, float]:
    """Bias RRF fusion by query shape: symbol/path-like queries lean on BM25
    (exact lexical), natural-language questions lean on dense. Returns
    (dense_weight, sparse_weight). Deterministic — a cheap regex, gate-able."""
    has_symbol = bool(_SYMBOL_SHAPE_RE.search(query))
    words = re.findall(r"[A-Za-z']+", query)
    nl_words = [w for w in words if len(w) >= 3]
    looks_nl = len(nl_words) >= 5 and not has_symbol
    if has_symbol and not looks_nl:
        return 0.8, 1.2
    if looks_nl:
        return 1.2, 0.8
    return 1.0, 1.0


async def _embed_query_vectors(
    ctx: RetrievalContext, query: str, vector_kinds: list[str]
) -> dict[str, list[float]]:
    async def _one(kind: str) -> tuple[str, list[float]]:
        name = "dense_code" if kind == "code" else "dense_text"
        vec = await ctx.embedder.embed_query(query, vector=name)  # type: ignore[arg-type]
        return kind, vec

    pairs = await asyncio.gather(*[_one(kind) for kind in vector_kinds])
    return dict(pairs)


def _to_doc(hit: Hit) -> str:
    payload = hit.payload
    anchor = f'{payload.get("resource_uri","?")}:{payload.get("start_line","?")}'
    ctx_path = payload.get("context_path") or ""
    head = f"[{anchor}]" + (f" {ctx_path}" if ctx_path else "")
    body = payload.get("content", "")
    return f"{head}\n{body}"


def _apply_quality_gate(hits: list[Hit], gate: float) -> tuple[list[Hit], int, float | None]:
    best_score = max((h.score for h in hits), default=None)
    kept = [h for h in hits if h.score >= gate]
    return kept, len(hits) - len(kept), best_score
