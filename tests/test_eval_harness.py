"""Unit tests for the unified eval harness scoring helpers (offline)."""

from __future__ import annotations

import pytest

from anvay.retrieval.evidence import EvidenceCandidate
from evals.harness import (
    EvalRunArtifact,
    ModeMetrics,
    ProductResult,
    Thresholds,
    _aggregate,
    _is_graph,
    _ItemScore,
    _match_file,
    _opt_mean,
    _passes,
    _retrieved_labels,
    render_markdown,
    resolve_products,
)


def _cand(file: str, channel: str = "hybrid", graph_ids: list[str] | None = None) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=f"c::{file}",
        channel=channel,
        role="overview",
        file=file,
        line=1,
        graph_node_ids=graph_ids or [],
    )


def test_match_file_suffix_and_basename() -> None:
    relevant = {"anvay/retrieval/hybrid.py"}
    assert _match_file("/abs/anvay/retrieval/hybrid.py", relevant) == "anvay/retrieval/hybrid.py"
    assert _match_file("/abs/other/hybrid.py", {"hybrid.py"}) == "hybrid.py"
    assert _match_file("/abs/other/sparse.py", relevant) is None
    assert _match_file("", relevant) is None


def test_retrieved_labels_ordered_unique() -> None:
    cands = [
        _cand("/abs/anvay/retrieval/hybrid.py"),
        _cand("/abs/anvay/retrieval/hybrid.py"),  # dup -> collapsed
        _cand("/abs/anvay/retrieval/sparse.py"),
    ]
    labels = _retrieved_labels(cands, {"anvay/retrieval/hybrid.py"})
    assert labels[0] == "anvay/retrieval/hybrid.py"
    assert len(labels) == len(set(labels))


def test_is_graph() -> None:
    assert _is_graph(_cand("a.py", channel="graph"))
    assert _is_graph(_cand("a.py", graph_ids=["n1"]))
    assert not _is_graph(_cand("a.py", channel="hybrid"))


def test_opt_mean_skips_none() -> None:
    assert _opt_mean([1.0, None, 0.0]) == 0.5
    assert _opt_mean([None, None]) is None


def _score(recall: float, ndcg: float, faith: float | None) -> _ItemScore:
    return _ItemScore(
        id="i",
        query="q",
        category="conceptual",
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        reciprocal_rank=1.0,
        latency_ms=10.0,
        n_candidates=5,
        graph_used=False,
        top_files=[],
        faithfulness=faith,
    )


def test_aggregate_and_misses() -> None:
    scores = [_score(1.0, 1.0, 0.9), _score(0.0, 0.0, None)]
    m = _aggregate("auto", scores, top_k=10)
    assert m.n == 2
    assert m.recall_at_k == 0.5
    assert m.faithfulness == 0.9  # None skipped
    assert m.misses == ["q"]  # the recall==0 item


def test_passes_gates_on_deterministic_and_llm_metrics() -> None:
    t = Thresholds()
    good = ModeMetrics(
        mode="auto", n=15,
        recall_at_k=0.9, ndcg_at_k=0.9, mrr=0.9,
        answer_correctness=0.65, context_recall=0.57,
    )
    assert _passes(good, t)
    # each deterministic metric can sink the gate
    assert not _passes(good.model_copy(update={"recall_at_k": 0.1}), t)
    assert not _passes(good.model_copy(update={"ndcg_at_k": 0.1}), t)
    assert not _passes(good.model_copy(update={"mrr": 0.1}), t)
    # answer_correctness and context_recall now gate when present
    assert not _passes(good.model_copy(update={"answer_correctness": 0.1}), t)
    assert not _passes(good.model_copy(update={"context_recall": 0.1}), t)
    # faithfulness and context_precision remain diagnostic (no gate even at 0.0)
    assert _passes(good.model_copy(update={"faithfulness": 0.0, "context_precision": 0.0}), t)
    # None means judge unavailable — skip LLM gate rather than fail
    assert _passes(good.model_copy(update={"answer_correctness": None, "context_recall": None}), t)


def test_resolve_products() -> None:
    assert {p.product_id for p in resolve_products(["all"])} >= {"anvay", "zod", "guava"}
    assert [p.product_id for p in resolve_products(["zod"])] == ["zod"]
    with pytest.raises(ValueError):
        resolve_products(["nope"])


def test_render_markdown_smoke() -> None:
    art = EvalRunArtifact(
        run_id="r1",
        generated_at="2026-01-01T00:00:00Z",
        config_path="anvay.yaml",
        config_fingerprint={"judge_model": "m"},
        top_k=10,
        limit=None,
        modes=["auto", "rewrite"],
        thresholds=Thresholds(),
        products=[
            ProductResult(
                product_id="anvay",
                n=1,
                passed=True,
                modes=[
                    ModeMetrics(mode="auto", n=1, recall_at_k=0.8, ndcg_at_k=0.7),
                    ModeMetrics(mode="rewrite", n=1, recall_at_k=0.9, ndcg_at_k=0.75),
                ],
            )
        ],
        output_dir="out",
    )
    md = render_markdown(art)
    assert "Anvay Eval Run r1" in md
    assert "vs `auto`" in md  # ablation delta block rendered
