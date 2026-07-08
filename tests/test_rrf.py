from anvay.retrieval.hybrid import Hit, rrf_merge


def _h(id_: str, source: str = "dense") -> Hit:
    return Hit(id=id_, score=0.0, payload={"id": id_}, source=source)


def test_rrf_merges_disjoint_rankings() -> None:
    dense = [_h("a"), _h("b"), _h("c")]
    sparse = [_h("d", "bm25"), _h("e", "bm25"), _h("f", "bm25")]
    merged = rrf_merge([dense, sparse], k=60, top_k=10)
    ids = [h.id for h in merged]
    # All six should appear, and the first item of each ranking ties at rank 1.
    assert set(ids) == {"a", "b", "c", "d", "e", "f"}
    # Equal-rank items have the same RRF score
    assert abs(merged[0].score - merged[1].score) < 1e-9


def test_rrf_boosts_items_in_both_rankings() -> None:
    dense = [_h("shared"), _h("dense_only"), _h("c")]
    sparse = [_h("shared", "bm25"), _h("sparse_only", "bm25"), _h("c", "bm25")]
    merged = rrf_merge([dense, sparse], k=60, top_k=10)
    # 'shared' appears at rank 1 in both → highest fused score
    assert merged[0].id == "shared"
    # 'shared' source includes both contributors
    assert "dense" in merged[0].source and "bm25" in merged[0].source


def test_rrf_respects_top_k() -> None:
    long = [_h(f"x{i}") for i in range(100)]
    merged = rrf_merge([long], top_k=5)
    assert len(merged) == 5
    assert [h.id for h in merged] == [f"x{i}" for i in range(5)]


def test_rrf_weights_bias_fusion() -> None:
    dense = [_h("d1"), _h("d2")]
    sparse = [_h("s1", "bm25"), _h("s2", "bm25")]
    # Sparse weighted 2x → its rank-1 item beats dense rank-1.
    merged = rrf_merge([dense, sparse], weights=[1.0, 2.0], top_k=10)
    assert merged[0].id == "s1"
    # Symmetric: dense heavier → dense rank-1 wins.
    merged = rrf_merge([dense, sparse], weights=[2.0, 1.0], top_k=10)
    assert merged[0].id == "d1"


def test_rrf_weights_default_to_uniform() -> None:
    dense = [_h("a"), _h("b")]
    sparse = [_h("c", "bm25")]
    with_none = rrf_merge([dense, sparse], top_k=10)
    with_ones = rrf_merge([dense, sparse], weights=[1.0, 1.0], top_k=10)
    assert [h.id for h in with_none] == [h.id for h in with_ones]


def test_rrf_weights_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="weights length"):
        rrf_merge([[_h("a")], [_h("b")]], weights=[1.0], top_k=5)


def test_shape_weights_classifies_query_shape() -> None:
    from anvay.retrieval.pipeline import _shape_weights

    # Symbol / path / snake_case → lean sparse.
    assert _shape_weights("getUserById") == (0.8, 1.2)
    assert _shape_weights("anvay/retrieval/pipeline.py") == (0.8, 1.2)
    assert _shape_weights("user_id validation") == (0.8, 1.2)
    # Natural-language question → lean dense.
    assert _shape_weights("how does the authentication flow work end to end") == (1.2, 0.8)
    # Short ambiguous → neutral.
    assert _shape_weights("auth flow") == (1.0, 1.0)
