"""Pytest wrapper around the eval harness.

Marked `@pytest.mark.eval` so it's skipped by default — `pytest -m eval` to
run. Requires Qdrant + embedder + reranker to be reachable and a product
index to already be populated. The product id is taken from
`ANVAY_EVAL_PRODUCT` (defaults to the dataset's `_meta.ingested_product_id`).

The thresholds in `queries.json` (`_meta.min_recall_at_10`, `_meta.min_mrr`)
are deliberately conservative — they should track real progress on the
retrieval pipeline. Bump them when the pipeline materially improves.
"""

from __future__ import annotations

import os

import httpx
import pytest

from anvay.config import AnvayConfig

from .harness import load_queries, run_ablation, run_eval


@pytest.mark.eval
async def test_retrieval_quality() -> None:
    config = _load_config_or_skip()
    meta, queries = load_queries()
    product_id = os.environ.get("ANVAY_EVAL_PRODUCT") or meta.get("ingested_product_id")
    if not product_id:
        pytest.skip("no product_id configured (set ANVAY_EVAL_PRODUCT or _meta.ingested_product_id)")

    _skip_unless_infra_reachable(config)

    report = await run_eval(config=config, product_id=product_id, top_k=10, queries=queries)
    print("\n" + report.render())

    floor_recall = float(meta.get("min_recall_at_10", 0.0))
    floor_mrr = float(meta.get("min_mrr", 0.0))
    assert report.recall_at_k >= floor_recall, (
        f"recall@10 = {report.recall_at_k:.3f} below floor {floor_recall}"
    )
    assert report.mrr >= floor_mrr, f"MRR = {report.mrr:.3f} below floor {floor_mrr}"


@pytest.mark.eval
async def test_graph_channel_helps_on_relational_slice() -> None:
    """Ablation: the graph channel must not regress the relational/graph slice.

    Enforces AGENTS.md's rule that the graph is a navigation layer justified by
    an eval-set win. Requires a live graph store in addition to the retrieval
    stack; skips cleanly when either is down.
    """
    from anvay.graph.store import create_graph_store

    config = _load_config_or_skip()
    meta, queries = load_queries()
    product_id = os.environ.get("ANVAY_EVAL_PRODUCT") or meta.get("ingested_product_id")
    if not product_id:
        pytest.skip("no product_id configured (set ANVAY_EVAL_PRODUCT or _meta.ingested_product_id)")

    _skip_unless_infra_reachable(config)

    graph_store = create_graph_store(config)
    if not await graph_store.health():
        await graph_store.aclose()
        pytest.skip("graph store unreachable")
    try:
        report = await run_ablation(
            config=config,
            product_id=product_id,
            graph_store=graph_store,
            top_k=10,
            tags=("relational", "graph"),
            queries=queries,
        )
    finally:
        await graph_store.aclose()
    print("\n" + report.render())

    assert report.delta_recall >= 0, f"graph regressed recall: {report.render()}"
    assert report.delta_mrr >= 0, f"graph regressed MRR: {report.render()}"
    assert report.delta_ndcg >= 0, f"graph regressed nDCG: {report.render()}"


# ---------------------------------------------------------------- helpers


def _load_config_or_skip() -> AnvayConfig:
    try:
        return AnvayConfig.load("anvay.yaml")
    except FileNotFoundError as e:
        pytest.skip(f"anvay.yaml not found: {e}")


def _skip_unless_infra_reachable(config: AnvayConfig) -> None:
    """Probe each upstream Qdrant + embedder + reranker is up before running
    the full eval. Cheap connection checks; skip cleanly when anything's down.
    """
    targets = [
        ("qdrant", config.vector_store.url),
        ("embedder", config.models.embedding.url or "http://localhost:8080"),
        ("reranker", config.models.reranker.url or "http://localhost:8081"),
    ]
    for name, url in targets:
        try:
            with httpx.Client(timeout=2.0) as c:
                c.get(url)
        except httpx.HTTPError as e:
            pytest.skip(f"{name} unreachable at {url}: {e}")
