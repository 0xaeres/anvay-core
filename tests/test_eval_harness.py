"""Unit tests for the eval harness's pure scoring helpers (no infra needed)."""

from __future__ import annotations

from anvay.retrieval.hybrid import Hit
from tests.eval.harness import (
    EvalReport,
    QueryResult,
    load_queries,
    matches_expected,
)


def _hit(uri: str, *, start: int = 1, end: int | None = None) -> Hit:
    return Hit(
        id="x",
        score=1.0,
        payload={"resource_uri": uri, "start_line": start, "end_line": end or start},
        source="dense",
    )


# ---------- matches_expected ------------------------------------------------


def test_matches_expected_by_file_suffix() -> None:
    h = _hit("/abs/path/to/anvay/retrieval/sparse.py")
    assert matches_expected(h, [{"file": "anvay/retrieval/sparse.py"}])


def test_matches_expected_rejects_unrelated_file() -> None:
    h = _hit("anvay/retrieval/sparse.py")
    assert not matches_expected(h, [{"file": "anvay/retrieval/reranker.py"}])


def test_matches_expected_any_of_listed_files() -> None:
    h = _hit("anvay/retrieval/hybrid.py")
    assert matches_expected(
        h,
        [
            {"file": "anvay/retrieval/sparse.py"},
            {"file": "anvay/retrieval/hybrid.py"},
        ],
    )


def test_matches_expected_line_range_overlap() -> None:
    h = _hit("anvay/foo.py", start=50, end=60)
    assert matches_expected(h, [{"file": "anvay/foo.py", "line_start": 55, "line_end": 70}])
    assert not matches_expected(h, [{"file": "anvay/foo.py", "line_start": 100, "line_end": 110}])


def test_matches_expected_no_line_range_means_whole_file() -> None:
    h = _hit("anvay/foo.py", start=999, end=1000)
    assert matches_expected(h, [{"file": "anvay/foo.py"}])


def test_matches_expected_empty_uri_never_matches() -> None:
    h = Hit(id="x", score=1.0, payload={"resource_uri": ""}, source="dense")
    assert not matches_expected(h, [{"file": "anvay/foo.py"}])


# ---------- EvalReport metrics ----------------------------------------------


def _qr(rank: int | None, query: str = "q") -> QueryResult:
    return QueryResult(query=query, top_k_hits=[], first_match_rank=rank, tags=[])


def test_recall_at_k_all_hit() -> None:
    report = EvalReport(results=[_qr(1), _qr(5), _qr(2)], top_k=10)
    assert report.recall_at_k == 1.0
    # MRR = (1/1 + 1/5 + 1/2) / 3
    assert abs(report.mrr - ((1.0 + 0.2 + 0.5) / 3)) < 1e-9


def test_recall_at_k_partial() -> None:
    report = EvalReport(results=[_qr(1), _qr(None), _qr(None)], top_k=10)
    assert report.recall_at_k == 1 / 3
    assert report.mrr == 1.0 / 3


def test_recall_at_k_zero_when_all_miss() -> None:
    report = EvalReport(results=[_qr(None), _qr(None)], top_k=10)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_render_lists_misses() -> None:
    r1 = _qr(1, query="found")
    r2 = _qr(None, query="missed-one")
    out = EvalReport(results=[r1, r2], top_k=10).render()
    assert "recall@10:" in out
    assert "missed-one" in out
    assert "found" not in out  # misses-only block


def test_render_when_no_misses() -> None:
    out = EvalReport(results=[_qr(1)], top_k=10).render()
    assert "everything landed in top-K" in out


# ---------- dataset shape ---------------------------------------------------


def test_queries_json_is_well_formed() -> None:
    _, queries = load_queries()
    assert isinstance(queries, list)
    assert len(queries) >= 30, "eval set should have at least 30 queries"
    for q in queries:
        assert "query" in q and isinstance(q["query"], str) and q["query"].strip()
        assert "expected" in q and isinstance(q["expected"], list) and q["expected"]
        for ex in q["expected"]:
            assert "file" in ex and ex["file"].endswith((".py", ".md", ".yaml", ".json"))


def test_queries_json_meta_has_thresholds() -> None:
    meta, _ = load_queries()
    assert "min_recall_at_10" in meta
    assert "min_mrr" in meta
    assert 0.0 <= meta["min_recall_at_10"] <= 1.0
    assert 0.0 <= meta["min_mrr"] <= 1.0


# ---------- graph ablation (no infra; retrieve_evidence is monkeypatched) ----


def test_filter_by_tags_selects_graph_slice() -> None:
    from tests.eval.harness import filter_by_tags

    queries = [
        {"query": "a", "tags": ["graph", "relational"]},
        {"query": "b", "tags": ["ingest"]},
        {"query": "c", "tags": ["RELATIONAL"]},
    ]
    selected = filter_by_tags(queries, ("relational",))
    assert [q["query"] for q in selected] == ["a", "c"]


def test_ablation_report_graph_helps_logic() -> None:
    from tests.eval.harness import AblationReport, EvalReport, QueryResult

    def _report(hit: bool) -> EvalReport:
        return EvalReport(
            results=[
                QueryResult(
                    query="q",
                    top_k_hits=[],
                    first_match_rank=1 if hit else None,
                    tags=["graph"],
                    relevance=[hit],
                    expected_count=1,
                )
            ],
            top_k=10,
        )

    helps = AblationReport(tags=["graph"], with_graph=_report(True), without_graph=_report(False))
    assert helps.delta_recall == 1.0
    assert helps.graph_helps is True

    regresses = AblationReport(
        tags=["graph"], with_graph=_report(False), without_graph=_report(True)
    )
    assert regresses.graph_helps is False


async def test_run_ablation_diffs_graph_on_vs_off(monkeypatch) -> None:
    from types import SimpleNamespace

    from anvay.retrieval.evidence import EvidenceCandidate
    from tests.eval import harness

    async def fake_retrieve_evidence(**kwargs):
        # Graph on -> resolve the expected file; graph off -> miss.
        if kwargs.get("graph_store") is not None:
            candidates = [
                EvidenceCandidate(
                    chunk_id="g1",
                    channel="graph",
                    role="relationship",
                    score=1.0,
                    file="anvay/graph/store.py",
                    line=1,
                )
            ]
        else:
            candidates = []
        return SimpleNamespace(candidates=candidates)

    monkeypatch.setattr(harness, "retrieve_evidence", fake_retrieve_evidence)
    queries = [
        {"query": "how does traversal reach neighbors", "expected": [{"file": "anvay/graph/store.py"}], "tags": ["graph"]},
        {"query": "unrelated ingest question", "expected": [{"file": "x.py"}], "tags": ["ingest"]},
    ]
    report = await harness.run_ablation(
        config=SimpleNamespace(),
        product_id="p",
        graph_store=object(),
        top_k=10,
        tags=("graph",),
        queries=queries,
        ctx=object(),  # injected: no RetrievalContext built, no infra touched
    )
    assert report.without_graph.total == 1  # only the graph-tagged query
    assert report.without_graph.recall_at_k == 0.0
    assert report.with_graph.recall_at_k == 1.0
    assert report.graph_helps is True
    assert "Δ+1.000" in report.render()
