"""Literal-ish grep over indexed chunk payloads.

Council repair uses this before semantic retrieval so citation repair can find
exact anchors from the same chunk universe later checked by evals.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable

from nexus.council.state import EvidenceChunk

_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+", re.IGNORECASE)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "anti",
    "are",
    "before",
    "check",
    "citation",
    "citations",
    "concrete",
    "data",
    "does",
    "draft",
    "evidence",
    "exact",
    "file",
    "flow",
    "from",
    "gotchas",
    "heading",
    "into",
    "line",
    "needs",
    "only",
    "ownership",
    "path",
    "product",
    "repair",
    "section",
    "skill",
    "system",
    "that",
    "this",
    "tier",
    "use",
    "validation",
    "when",
    "with",
}


async def grep_indexed_chunks(
    *,
    indexer,
    product_id: str,
    query: str,
    limit: int = 8,
    batch_size: int = 256,
) -> list[EvidenceChunk]:
    """Search indexed chunk payloads for query terms and return citation anchors."""
    terms = _query_terms(query)
    if not terms or limit <= 0:
        return []

    matches: list[EvidenceChunk] = []
    for kind in ("code", "text"):
        async for chunk_id, payload in _iter_payloads(
            indexer, product_id=product_id, vector_kind=kind, batch_size=batch_size
        ):
            hit = _match_payload(chunk_id, payload, terms)
            if hit is not None:
                matches.append(hit)

    deduped: dict[tuple[str, str], EvidenceChunk] = {}
    for match in sorted(matches, key=lambda item: item.score, reverse=True):
        key = (match.chunk_id, match.file)
        if key not in deduped:
            deduped[key] = match
    return list(deduped.values())[:limit]


async def sample_indexed_chunks(
    *,
    indexer,
    product_id: str,
    limit: int = 8,
    batch_size: int = 64,
) -> list[EvidenceChunk]:
    """Return first indexed chunks for citation fallback when literal grep misses."""
    if limit <= 0:
        return []
    out: list[EvidenceChunk] = []
    seen: set[str] = set()
    for kind in ("code", "text"):
        async for chunk_id, payload in _iter_payloads(
            indexer, product_id=product_id, vector_kind=kind, batch_size=batch_size
        ):
            if chunk_id in seen:
                continue
            hit = _payload_to_evidence(chunk_id, payload, score=0.01)
            if hit is None:
                continue
            seen.add(chunk_id)
            out.append(hit)
            if len(out) >= limit:
                return out
    return out


async def _iter_payloads(
    indexer, *, product_id: str, vector_kind: str, batch_size: int
) -> AsyncIterator[tuple[str, dict]]:
    if not hasattr(indexer, "iter_chunk_payloads"):
        return
    async for chunk_id, payload in indexer.iter_chunk_payloads(
        product_id=product_id,
        vector_kind=vector_kind,
        batch_size=batch_size,
    ):
        yield chunk_id, payload


def _match_payload(
    chunk_id: str, payload: dict, terms: frozenset[str]
) -> EvidenceChunk | None:
    file_ = str(payload.get("resource_uri") or "")
    content = str(payload.get("content") or "")
    context = str(payload.get("context_path") or "")
    if not file_ or not content:
        return None

    haystack = f"{file_}\n{context}\n{content}".lower()
    matched = [term for term in terms if term in haystack]
    if not matched:
        return None

    lines = content.splitlines() or [content]
    best_offset = 0
    best_line = lines[0] if lines else ""
    best_line_score = -1
    for idx, line in enumerate(lines):
        line_score = _term_count(line, matched)
        if line_score > best_line_score:
            best_offset = idx
            best_line = line
            best_line_score = line_score

    try:
        start_line = int(payload.get("start_line") or 1)
    except (TypeError, ValueError):
        start_line = 1
    context_bonus = _term_count(f"{file_} {context}", matched)
    score = float((len(matched) * 10) + max(best_line_score, 0) + context_bonus)
    return EvidenceChunk(
        chunk_id=chunk_id,
        file=file_,
        line=start_line + best_offset,
        score=score,
        excerpt=_excerpt(lines, best_offset, fallback=best_line),
    )


def _payload_to_evidence(
    chunk_id: str, payload: dict, *, score: float
) -> EvidenceChunk | None:
    file_ = str(payload.get("resource_uri") or "")
    content = str(payload.get("content") or "")
    if not file_ or not content:
        return None
    try:
        start_line = int(payload.get("start_line") or 1)
    except (TypeError, ValueError):
        start_line = 1
    lines = content.splitlines() or [content]
    return EvidenceChunk(
        chunk_id=chunk_id,
        file=file_,
        line=start_line,
        score=score,
        excerpt=_excerpt(lines, 0, fallback=lines[0] if lines else ""),
    )


def _query_terms(query: str) -> frozenset[str]:
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(query.lower()):
        token = token.strip("._-:/")
        if len(token) < 4 or token in _STOPWORDS:
            continue
        terms.add(token)
    return frozenset(terms)


def _term_count(text: str, terms: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _excerpt(lines: list[str], offset: int, *, fallback: str) -> str:
    if not lines:
        return fallback.strip()[:500]
    start = max(0, offset - 1)
    end = min(len(lines), offset + 2)
    return "\n".join(line.strip() for line in lines[start:end]).strip()[:500]
