from __future__ import annotations

from types import SimpleNamespace

import pytest

import anvay.mcp_server.tools as mcp_tools
from anvay.mcp_server.tools import (
    ToolState,
    _extract_section,
    _merge_adjacent_hits,
    _pack_hits,
    _render_evidence_set,
)
from anvay.retrieval.evidence import (
    EvidenceCandidate,
    EvidenceCoverage,
    EvidenceSet,
    QueryUnderstanding,
    RetrievalTrace,
)


def _hit(*, file="a.py", line=1, end=None, score=0.5, content="body text"):
    return {
        "score": score,
        "source": "hybrid",
        "anchor": f"{file}:{line}",
        "context_path": None,
        "content": content,
        "_file": file,
        "_line": line,
        "_end_line": end if end is not None else line,
    }


# ---------------------------------------------------------------- packing


def test_pack_hits_respects_token_budget_with_anchor_tail() -> None:
    hits = [_hit(line=i, score=1.0 - i * 0.1, content="x" * 400) for i in range(1, 6)]
    packed = _pack_hits(
        [{k: v for k, v in h.items()} for h in hits],
        detail="excerpt",
        max_response_tokens=250,  # ~2 hits of 400 chars (100 tokens each)
    )
    with_content = [h for h in packed if h["content"]]
    anchor_only = [h for h in packed if not h["content"]]
    assert len(with_content) == 2
    assert len(anchor_only) == 3
    # Anchors survive for the tail so the client can read the file directly.
    assert all(h["anchor"] for h in anchor_only)


def test_pack_hits_anchor_detail_strips_all_content() -> None:
    packed = _pack_hits([_hit()], detail="anchor")
    assert packed[0]["content"] is None
    assert "_file" not in packed[0]


def test_pack_hits_excerpt_caps_per_hit_content() -> None:
    packed = _pack_hits([_hit(content="y" * 2000)], detail="excerpt")
    assert len(packed[0]["content"]) <= 701 + 1


def test_pack_hits_full_keeps_whole_content() -> None:
    packed = _pack_hits([_hit(content="y" * 2000)], detail="full", max_response_tokens=4000)
    assert len(packed[0]["content"]) == 2000


# ---------------------------------------------------------------- span merge


def test_merge_adjacent_hits_unions_overlapping_spans() -> None:
    hits = [
        _hit(line=1, end=10, score=0.9, content="first span"),
        _hit(line=12, end=20, score=0.7, content="second span"),  # gap 2 ≤ 3
        _hit(file="other.py", line=1, end=5, score=0.8, content="other file"),
    ]
    merged = _merge_adjacent_hits(hits)
    a_py = [h for h in merged if h["anchor"].startswith("a.py")]
    assert len(a_py) == 1
    assert a_py[0]["score"] == 0.9
    assert a_py[0]["merged_spans"] == ["1-10", "12-20"]
    assert "first span" in a_py[0]["content"]
    assert "second span" in a_py[0]["content"]
    assert len([h for h in merged if h["anchor"].startswith("other.py")]) == 1


def test_merge_adjacent_hits_keeps_distant_spans_separate() -> None:
    hits = [
        _hit(line=1, end=5, score=0.9),
        _hit(line=50, end=60, score=0.8),
    ]
    merged = _merge_adjacent_hits(hits)
    assert len(merged) == 2
    assert all("merged_spans" not in h for h in merged)


# ---------------------------------------------------------------- evidence render


def _evidence_set() -> EvidenceSet:
    return EvidenceSet(
        product_id="demo",
        query="how does auth work",
        understanding=QueryUnderstanding(query="how does auth work", shape="global"),
        candidates=[
            EvidenceCandidate(
                chunk_id="c1",
                channel="hybrid",
                role="definition",
                score=0.9,
                file="auth.py",
                line=10,
                end_line=20,
                excerpt="def authenticate(): ...",
                metadata={"kind": "code"},
            )
        ],
        coverage=EvidenceCoverage(sufficient=True),
        trace=[RetrievalTrace(channel="hybrid", query="q", hits=1)],
    )


def test_render_evidence_set_hides_trace_and_metadata_by_default() -> None:
    out = _render_evidence_set(_evidence_set())
    assert "trace" not in out
    assert "metadata" not in out["hits"][0]
    assert "graph_node_ids" not in out["hits"][0]
    assert out["hits"][0]["content"] == "def authenticate(): ..."


def test_render_evidence_set_debug_includes_trace_and_metadata() -> None:
    out = _render_evidence_set(_evidence_set(), debug=True)
    assert len(out["trace"]) == 1
    assert out["hits"][0]["metadata"] == {"kind": "code"}


# ---------------------------------------------------------------- get_skill section


_SKILL_BODY = (
    "# Master\n\nIntro paragraph.\n\n"
    "## Setup\n\nInstall the things.\n\n### Sub-detail\n\nMore setup.\n\n"
    "## Deployment\n\nShip the things.\n"
)


def test_extract_section_returns_h2_subtree() -> None:
    out = _extract_section(_SKILL_BODY, "Setup")
    assert out.startswith("## Setup")
    assert "Sub-detail" in out
    assert "Deployment" not in out


def test_extract_section_case_insensitive_and_missing() -> None:
    assert _extract_section(_SKILL_BODY, "deployment").startswith("## Deployment")
    assert _extract_section(_SKILL_BODY, "nonexistent") is None


# ---------------------------------------------------------------- warmup


@pytest.mark.asyncio
async def test_warmup_swallows_all_failures(monkeypatch) -> None:
    class ExplodingEmbedder:
        async def embed_query(self, *a, **kw):
            raise RuntimeError("embedder down")

    class ExplodingReranker:
        async def rerank(self, *a, **kw):
            raise RuntimeError("reranker down")

    state = ToolState(product="demo", config=SimpleNamespace())
    state._ctx = SimpleNamespace(
        embedder=ExplodingEmbedder(), reranker=ExplodingReranker()
    )

    async def exploding_sparse():
        raise RuntimeError("sparse down")

    monkeypatch.setattr(mcp_tools, "_warm_sparse", exploding_sparse)
    await state.warmup()  # must not raise
