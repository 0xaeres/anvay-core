"""Unit tests for the RAG eval runner's context recall scoring.

These tests exercise _heuristic_context_recall and _any_uri_covers without
any network dependencies.  They pin the correct suffix-match behaviour and
guard against regressions back to the old substring-in-joined-string approach.
"""

from __future__ import annotations

from evals.common import GoldenItem
from evals.run_ragas import _any_uri_covers, _heuristic_context_recall

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _item(*expected_files: str) -> GoldenItem:
    return GoldenItem(
        id="t",
        query="q",
        expected_answer="",
        expected_skill="",
        complexity="simple",
        expected_files=list(expected_files),
    )


def _hit(resource_uri: str) -> object:
    """Minimal hit-like object with a payload dict."""

    class _Hit:
        def __init__(self) -> None:
            self.payload = {"resource_uri": resource_uri}

    return _Hit()


# --------------------------------------------------------------------------- #
# _any_uri_covers
# --------------------------------------------------------------------------- #


def test_any_uri_covers_exact_suffix() -> None:
    uris = ["/abs/path/anvay/retrieval/sparse.py"]
    assert _any_uri_covers("anvay/retrieval/sparse.py", uris)


def test_any_uri_covers_full_match() -> None:
    uris = ["anvay/retrieval/sparse.py"]
    assert _any_uri_covers("anvay/retrieval/sparse.py", uris)


def test_any_uri_covers_wrong_file() -> None:
    uris = ["anvay/retrieval/reranker.py"]
    assert not _any_uri_covers("anvay/retrieval/sparse.py", uris)


def test_any_uri_covers_no_false_positive_from_prefix() -> None:
    """'sparse.py' must NOT match 'sparse_encoder.py' — the old bug."""
    uris = ["anvay/retrieval/sparse_encoder.py"]
    assert not _any_uri_covers("sparse.py", uris)


def test_any_uri_covers_empty_uris() -> None:
    assert not _any_uri_covers("anvay/foo.py", [])


def test_any_uri_covers_empty_uri_string_skipped() -> None:
    assert not _any_uri_covers("anvay/foo.py", ["", "  "])


def test_any_uri_covers_multiple_uris_first_matches() -> None:
    uris = ["anvay/retrieval/sparse.py", "anvay/retrieval/reranker.py"]
    assert _any_uri_covers("anvay/retrieval/sparse.py", uris)


def test_any_uri_covers_multiple_uris_second_matches() -> None:
    uris = ["anvay/retrieval/reranker.py", "anvay/retrieval/sparse.py"]
    assert _any_uri_covers("anvay/retrieval/sparse.py", uris)


# --------------------------------------------------------------------------- #
# _heuristic_context_recall
# --------------------------------------------------------------------------- #


def test_recall_no_expected_files_is_one() -> None:
    item = _item()
    assert _heuristic_context_recall(item, []) == 1.0


def test_recall_all_expected_files_covered() -> None:
    item = _item("anvay/retrieval/sparse.py", "anvay/retrieval/reranker.py")
    hits = [
        _hit("anvay/retrieval/sparse.py"),
        _hit("anvay/retrieval/reranker.py"),
    ]
    assert _heuristic_context_recall(item, hits) == 1.0


def test_recall_partial_coverage() -> None:
    item = _item("anvay/retrieval/sparse.py", "anvay/retrieval/reranker.py")
    hits = [_hit("anvay/retrieval/sparse.py")]
    assert _heuristic_context_recall(item, hits) == 0.5


def test_recall_zero_when_no_hits_match() -> None:
    item = _item("anvay/retrieval/sparse.py")
    hits = [_hit("anvay/retrieval/reranker.py")]
    assert _heuristic_context_recall(item, hits) == 0.0


def test_recall_no_false_positive_from_old_substring_bug() -> None:
    """'sparse.py' must NOT be credited when only 'sparse_encoder.py' was retrieved."""
    item = _item("anvay/retrieval/sparse.py")
    hits = [_hit("anvay/retrieval/sparse_encoder.py")]
    assert _heuristic_context_recall(item, hits) == 0.0


def test_recall_absolute_path_uri_matches_relative_expected() -> None:
    item = _item("anvay/retrieval/sparse.py")
    hits = [_hit("/home/runner/work/anvay/anvay/retrieval/sparse.py")]
    assert _heuristic_context_recall(item, hits) == 1.0


def test_recall_case_insensitive() -> None:
    item = _item("ANVAY/Retrieval/Sparse.py")
    hits = [_hit("anvay/retrieval/sparse.py")]
    assert _heuristic_context_recall(item, hits) == 1.0


def test_recall_empty_hits() -> None:
    item = _item("anvay/retrieval/sparse.py")
    assert _heuristic_context_recall(item, []) == 0.0


def test_recall_hit_with_none_payload() -> None:
    class _NullHit:
        payload = None

    item = _item("anvay/retrieval/sparse.py")
    assert _heuristic_context_recall(item, [_NullHit()]) == 0.0


def test_recall_three_of_three_files() -> None:
    item = _item("a.py", "b.py", "c.py")
    hits = [_hit("a.py"), _hit("b.py"), _hit("c.py")]
    assert _heuristic_context_recall(item, hits) == 1.0


def test_recall_two_of_three_files() -> None:
    item = _item("a.py", "b.py", "c.py")
    hits = [_hit("a.py"), _hit("c.py")]
    assert abs(_heuristic_context_recall(item, hits) - 2 / 3) < 1e-9
