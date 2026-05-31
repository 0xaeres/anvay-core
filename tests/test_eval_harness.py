"""Unit tests for the eval harness's pure scoring helpers (no infra needed)."""

from __future__ import annotations

from nexus.retrieval.hybrid import Hit
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
    h = _hit("/abs/path/to/nexus/retrieval/sparse.py")
    assert matches_expected(h, [{"file": "nexus/retrieval/sparse.py"}])


def test_matches_expected_rejects_unrelated_file() -> None:
    h = _hit("nexus/retrieval/sparse.py")
    assert not matches_expected(h, [{"file": "nexus/retrieval/reranker.py"}])


def test_matches_expected_any_of_listed_files() -> None:
    h = _hit("nexus/retrieval/hybrid.py")
    assert matches_expected(
        h,
        [
            {"file": "nexus/retrieval/sparse.py"},
            {"file": "nexus/retrieval/hybrid.py"},
        ],
    )


def test_matches_expected_line_range_overlap() -> None:
    h = _hit("nexus/foo.py", start=50, end=60)
    assert matches_expected(h, [{"file": "nexus/foo.py", "line_start": 55, "line_end": 70}])
    assert not matches_expected(h, [{"file": "nexus/foo.py", "line_start": 100, "line_end": 110}])


def test_matches_expected_no_line_range_means_whole_file() -> None:
    h = _hit("nexus/foo.py", start=999, end=1000)
    assert matches_expected(h, [{"file": "nexus/foo.py"}])


def test_matches_expected_empty_uri_never_matches() -> None:
    h = Hit(id="x", score=1.0, payload={"resource_uri": ""}, source="dense")
    assert not matches_expected(h, [{"file": "nexus/foo.py"}])


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
