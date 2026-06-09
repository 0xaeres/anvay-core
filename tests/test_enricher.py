"""Tests for the contextual enricher (HQE for code, Anthropic CR for docs)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from nexus.ingest.enricher import (
    _DOC_TRUNCATE_CHARS,
    ContextualEnricher,
    _truncate_doc,
)
from nexus.ingest.models import Chunk, ChunkKind, ResourceRef

# ---------- helpers ---------------------------------------------------------


def _chunk(content: str, kind: ChunkKind, *, uri: str = "f.py", ctx: str = "") -> Chunk:
    return Chunk(
        product_id="p",
        resource=ResourceRef(source_id="s", uri=uri, mime="text/plain"),
        content=content,
        start_line=1,
        end_line=content.count("\n") + 1,
        kind=kind,
        context_path=ctx,
    )


def _build(handler) -> ContextualEnricher:
    """Return an enricher whose SDK client uses MockTransport(handler)."""
    transport = httpx.MockTransport(handler)
    enricher = ContextualEnricher(
        base_url="http://test.local/v1",
        model="m",
        api_key="k",
        enrich_code=True,
        enrich_docs=True,
        concurrency=4,
    )
    enricher._chat_client._http_client = httpx.AsyncClient(
        transport=transport,
        timeout=5.0,
    )
    enricher._chat_client._client = AsyncOpenAI(
        api_key="k",
        base_url="http://test.local/v1",
        max_retries=0,
        http_client=enricher._chat_client._http_client,
    )
    return enricher


def _ok(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
    )


# ---------- dispatch on chunk kind ------------------------------------------


def test_code_chunk_calls_hqe_prompt() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return _ok("Q: How do I verify a JWT?\nQ: Token check?\nQ: Auth helper?")

    enricher = _build(handler)
    c = _chunk("def verify(token): ...", ChunkKind.CODE, uri="auth.py")
    out = asyncio.run(enricher.enrich([c]))
    asyncio.run(enricher.aclose())

    assert len(seen) == 1
    assert "Q:" in seen[0]
    assert out[0].context_summary and out[0].context_summary.startswith("Q:")


def test_doc_chunk_calls_contextual_retrieval_prompt() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return _ok("This chunk is the API-key rotation section of the auth guide.")

    enricher = _build(handler)
    c = _chunk("Rotate the key with the rotate_key() helper.", ChunkKind.DOC, uri="docs/auth.md")
    doc = "# Auth Guide\n\n## API Keys\n\n### Rotating\n\nRotate the key with the rotate_key() helper."
    out = asyncio.run(enricher.enrich([c], doc_contents={"docs/auth.md": doc}))
    asyncio.run(enricher.aclose())

    assert len(seen) == 1
    assert "<document>" in seen[0]
    assert "<chunk>" in seen[0]
    assert "situate this chunk" in seen[0]
    assert out[0].context_summary == "This chunk is the API-key rotation section of the auth guide."


def test_doc_chunk_without_full_doc_skips_llm_call() -> None:
    """Falls back gracefully — `text_for_embedding()` still has context_path."""
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return _ok("should not be called")

    enricher = _build(handler)
    c = _chunk("Some doc content.", ChunkKind.DOC, uri="docs/x.md")
    out = asyncio.run(enricher.enrich([c]))  # no doc_contents
    asyncio.run(enricher.aclose())

    assert seen == []
    assert out[0].context_summary is None


def test_enrich_docs_flag_off_skips_doc_calls() -> None:
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return _ok("x")

    enricher = _build(handler)
    enricher.enrich_docs = False
    c = _chunk("Body", ChunkKind.DOC, uri="docs/x.md")
    out = asyncio.run(enricher.enrich([c], doc_contents={"docs/x.md": "Body"}))
    asyncio.run(enricher.aclose())

    assert seen == []
    assert out[0].context_summary is None


def test_enrich_code_flag_off_skips_code_calls() -> None:
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return _ok("x")

    enricher = _build(handler)
    enricher.enrich_code = False
    c = _chunk("def f(): pass", ChunkKind.CODE)
    out = asyncio.run(enricher.enrich([c]))
    asyncio.run(enricher.aclose())

    assert seen == []
    assert out[0].context_summary is None


def test_mixed_batch_routes_each_chunk_correctly() -> None:
    payloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        payloads.append(body)
        if "<document>" in body:
            return _ok("doc-context")
        return _ok("Q: q1\nQ: q2\nQ: q3")

    enricher = _build(handler)
    code_c = _chunk("def foo(): ...", ChunkKind.CODE)
    doc_c = _chunk("Para about foo.", ChunkKind.DOC, uri="d.md")
    out = asyncio.run(
        enricher.enrich([code_c, doc_c], doc_contents={"d.md": "Para about foo."})
    )
    asyncio.run(enricher.aclose())

    assert {c.context_summary for c in out} == {"doc-context", "Q: q1\nQ: q2\nQ: q3"}
    # One HQE + one CR call.
    assert any("<document>" in p for p in payloads)
    assert any("Q:" in p and "<document>" not in p for p in payloads)


# ---------- transport robustness --------------------------------------------


def test_non_200_response_leaves_chunk_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    enricher = _build(handler)
    c = _chunk("def f(): ...", ChunkKind.CODE)
    out = asyncio.run(enricher.enrich([c]))
    asyncio.run(enricher.aclose())
    assert out[0].context_summary is None


def test_empty_choices_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    enricher = _build(handler)
    c = _chunk("def f(): ...", ChunkKind.CODE)
    out = asyncio.run(enricher.enrich([c]))
    asyncio.run(enricher.aclose())
    assert out[0].context_summary is None


# ---------- _truncate_doc ---------------------------------------------------


def test_truncate_doc_passthrough_under_cap() -> None:
    doc = "small body"
    assert _truncate_doc(doc, around_chunk="body") == doc


def test_truncate_doc_centres_window_around_chunk() -> None:
    marker = "UNIQUE_MARKER_TEXT_THAT_IS_LONG_ENOUGH"
    body = ("filler line\n" * 5000) + marker + ("\nmore filler\n" * 5000)
    assert len(body) > _DOC_TRUNCATE_CHARS

    out = _truncate_doc(body, around_chunk=marker)
    assert len(out) <= _DOC_TRUNCATE_CHARS + 40  # plus the "[truncated]" markers
    assert marker in out


def test_truncate_doc_unfound_chunk_falls_back_to_head() -> None:
    body = "a" * (_DOC_TRUNCATE_CHARS + 1000)
    out = _truncate_doc(body, around_chunk="nope-not-here-token")
    assert out.startswith("a")
    assert "truncated" in out


# ---------- empty input is a no-op ------------------------------------------


def test_enrich_empty_list_returns_empty() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call")

    enricher = _build(handler)
    out = asyncio.run(enricher.enrich([]))
    asyncio.run(enricher.aclose())
    assert out == []


# silence unused-import warnings under future python
_ = pytest
