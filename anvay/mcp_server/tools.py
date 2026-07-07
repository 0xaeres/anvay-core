"""MCP tool implementations.

Each tool is an async function `(state: ToolState, **kwargs) -> dict`. The
server module wraps them and JSON-serialises the return for the TextContent
response.

Two layers:
  Guidance — find_skills, get_skill, report_outcome
  Context — query_code_context, hybrid_search_corpus
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

from pydantic import BaseModel

from anvay.config import AnvayConfig
from anvay.council.queue import ProposalQueue
from anvay.council.skill_catalog import fixed_skill_name, product_slug
from anvay.graph.models import GraphRAGQuery
from anvay.graph.rag import answer_graph_rag
from anvay.graph.store import create_graph_store
from anvay.llm.client import ChatClient
from anvay.retrieval.chunk_grep import grep_indexed_chunks
from anvay.retrieval.evidence import EvidenceSet, retrieve_evidence
from anvay.retrieval.pipeline import RetrievalContext, retrieve
from anvay.skills.models import Skill
from anvay.skills.store import SkillStore

log = logging.getLogger(__name__)

DEFAULT_RESPONSE_TOKENS = 4000
_EXCERPT_CHAR_CAP = 700  # per-hit content cap at detail="excerpt"


async def _warm_sparse() -> None:
    from anvay.retrieval.sparse import aencode_query

    await aencode_query("warmup")


class ToolError(BaseModel):
    error: str
    product_id: str | None = None


class SkillSummary(BaseModel):
    id: str
    name: str
    tier: str
    confidence: float
    summary: str


class FindSkillsResponse(BaseModel):
    query: str
    context: str
    current_file: str | None = None
    filtered_from: int
    skills: list[SkillSummary]


class SkillResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    product: str
    tier: str
    parent: str | None = None
    related: list[str]
    coverage: dict[str, Any]
    version: int
    confidence: float
    eval_status: str
    eval_summary: str
    eval_failures: list[str]
    quality_score: float
    signals_used: list[str]
    applies_to: dict[str, Any]
    provenance: dict[str, Any]
    body: str


class OutcomeRecord(BaseModel):
    skill_name: str
    succeeded: bool
    notes: str
    ts: float
    signal_id: str


class ReportOutcomeResponse(BaseModel):
    ok: bool
    received: OutcomeRecord


class RetrievalHitResponse(BaseModel):
    score: float
    source: str
    anchor: str
    context_path: str | None = None
    content: str | None = None
    merged_spans: list[str] | None = None


class RetrievalResponse(BaseModel):
    reranked: bool
    hits: list[RetrievalHitResponse]


class GrepHitResponse(BaseModel):
    score: float
    anchor: str
    content: str
    source: str


class GrepResponse(BaseModel):
    query: str
    hits: list[GrepHitResponse]


@dataclass
class ToolState:
    """Shared, lazily-initialised handles for the lifetime of the MCP server."""

    product: str
    config: AnvayConfig
    _ctx: RetrievalContext | None = None
    _graph_store: object | None = None
    _graph_store_loaded: bool = False
    _graphrag_chat: ChatClient | None = None
    _store: SkillStore | None = None
    _queue: ProposalQueue | None = None
    _outcomes: list[dict] = field(default_factory=list)

    @property
    def ctx(self) -> RetrievalContext:
        if self._ctx is None:
            self._ctx = RetrievalContext.from_config(self.config)
        return self._ctx

    @property
    def graph_store(self):
        if not self._graph_store_loaded:
            self._graph_store = create_graph_store(self.config)
            self._graph_store_loaded = True
        return self._graph_store

    @property
    def graphrag_chat(self) -> ChatClient:
        if self._graphrag_chat is None:
            self._graphrag_chat = ChatClient.from_cfg(
                self.config.models.council,
                role="graphrag",
            )
        return self._graphrag_chat

    @property
    def store(self) -> SkillStore:
        if self._store is None:
            root = Path(self.config.hierarchy_root)
            if not root.is_absolute():
                root = Path.cwd() / root
            self._store = SkillStore(root)
        return self._store

    @property
    def queue(self) -> ProposalQueue:
        if self._queue is None:
            self._queue = ProposalQueue(Path(self.config.storage.proposal_queue))
        return self._queue

    async def warmup(self) -> None:
        """Best-effort model warm-up so the first interactive call doesn't pay
        embedder/reranker/fastembed cold-start inside the latency budget. Every
        step soft-fails — a down Qdrant or embedder must not kill the server."""
        try:
            ctx = self.ctx
        except Exception:
            log.debug("warmup: retrieval context unavailable", exc_info=True)
            return
        for step_name, step in (
            ("embedder", lambda: ctx.embedder.embed_query("warmup", vector="dense_text")),
            ("sparse", lambda: _warm_sparse()),
            ("reranker", lambda: ctx.reranker.rerank("warmup", ["warmup document"], top_k=1)),
        ):
            try:
                await step()
            except Exception as e:
                log.debug("warmup %s failed (non-fatal): %s", step_name, e)

    async def aclose(self) -> None:
        closers = []
        if self._ctx is not None:
            closers.append(self._ctx.aclose)
        if self._graph_store is not None and hasattr(self._graph_store, "aclose"):
            closers.append(self._graph_store.aclose)
        if self._graphrag_chat is not None:
            closers.append(self._graphrag_chat.aclose)
        for close in closers:
            try:
                await close()
            except Exception:
                log.exception("tool state close failed")


# ---------------------------------------------------------------- guidance tools


async def find_skills(
    state: ToolState,
    *,
    query: str,
    context: str = "general",
    current_file: str | None = None,
    top_k: int = 5,
) -> dict:
    """Return ranked skill summaries for a query+context.

    Selective serving:
      1. `current_file` filters by `applies_to.files` glob match.
      2. `context` (when not "general") filters by exact `applies_to.contexts` tag.
      3. Top-K survivors are ranked by lexical overlap + confidence.
    """
    all_skills = state.store.iter_skills()
    if not all_skills:
        return FindSkillsResponse(
            query=query,
            context=context,
            current_file=current_file,
            filtered_from=0,
            skills=[],
        ).model_dump(mode="json")

    product_skills = [s for s in all_skills if s.product == state.product]
    master_skills = [s for s in product_skills if s.tier == "product_master"]
    canonical_master = fixed_skill_name(product_slug(state.product), "skill")
    candidates = [
        s
        for s in product_skills
        if _matches_file_globs(current_file, s.applies_to.files)
        and _matches_context(context, s.applies_to.contexts)
        and s.tier != "product_master"
    ]

    ql = (query + " " + context).lower()
    q_tokens = {t for t in _tokens(ql) if len(t) >= 3}
    scored: list[tuple[float, Skill]] = []
    for s in candidates:
        haystack = (
            f"{s.name} {s.description} {' '.join(s.applies_to.contexts or [])} {s.body}"
        ).lower()
        h_tokens = set(_tokens(haystack))
        if not q_tokens:
            score = s.confidence
        else:
            overlap = len(q_tokens & h_tokens)
            score = overlap / max(len(q_tokens), 1) + 0.2 * s.confidence
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    masters = sorted(
        master_skills,
        key=lambda s: (0 if s.name == canonical_master else 1, -s.confidence, s.name),
    )[:1]
    remaining = max(top_k - len(masters), 0)
    top = [*masters, *[s for _, s in scored[:remaining] if s not in masters]]

    out: list[SkillSummary] = []
    for s in top:
        out.append(
            SkillSummary(
                id=s.id,
                name=s.name,
                tier=s.tier,
                confidence=s.confidence,
                summary=s.description or _first_paragraph(s.body),
            )
        )
    return FindSkillsResponse(
        query=query,
        context=context,
        current_file=current_file,
        filtered_from=len(product_skills),
        skills=out,
    ).model_dump(mode="json")


async def get_skill(state: ToolState, *, name: str, section: str | None = None) -> dict:
    """Return the skill body + frontmatter. When `section` names an H2 heading,
    only that heading's subtree is returned — keeps big master skills cheap for
    clients that need one procedure, not the whole document."""
    for s in state.store.iter_skills():
        if s.product != state.product:
            continue
        if s.name == name:
            out = s.model_dump(mode="json")
            out["id"] = s.id
            if section:
                extracted = _extract_section(s.body, section)
                if extracted is None:
                    available = _section_headings(s.body)
                    return ToolError(
                        error=(
                            f"section not found: {section!r}. "
                            f"Available sections: {available}"
                        )
                    ).model_dump(exclude_none=True)
                out["body"] = extracted
            return SkillResponse.model_validate(out).model_dump(mode="json")
    return ToolError(error=f"skill not found: {name}").model_dump(exclude_none=True)


async def report_outcome(
    state: ToolState,
    *,
    skill_name: str,
    succeeded: bool,
    notes: str = "",
) -> dict:
    """Persist an outcome signal for future skill improvement."""
    record = {
        "skill_name": skill_name,
        "succeeded": succeeded,
        "notes": notes,
        "ts": time.time(),
    }
    state._outcomes.append(record)
    signal_id = state.queue.record_skill_signal(
        product_id=state.product,
        source_type="mcp_outcome",
        skill_name=skill_name,
        text=notes or ("Skill succeeded." if succeeded else "Skill failed."),
        metadata={"succeeded": succeeded, "ts": record["ts"]},
    )
    record["signal_id"] = signal_id
    log.info("outcome reported: %s", record)
    return ReportOutcomeResponse(ok=True, received=record).model_dump(mode="json")


# ---------------------------------------------------------------- context tools


async def query_code_context(
    state: ToolState,
    *,
    symbol: str,
    file_glob: str = "**/*",
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
) -> dict:
    """Cheap symbol lookup — runs the retrieval pipeline in code-only mode."""
    result = await retrieve(
        ctx=state.ctx,
        product_id=state.product,
        query=symbol,
        top_k=10,
        mode="code",
    )
    rendered = _render_retrieval(
        result, detail=detail, max_response_tokens=max_response_tokens
    )
    if file_glob and file_glob != "**/*":
        rendered["hits"] = [
            hit for hit in rendered["hits"] if _matches_file_globs(hit["anchor"].split(":", 1)[0], [file_glob])
        ]
    return RetrievalResponse.model_validate(rendered).model_dump(mode="json")


async def grep_corpus(
    state: ToolState,
    *,
    query: str,
    product_id: str | None = None,
    top_k: int = 8,
) -> dict:
    """Exact-ish indexed chunk grep. Product-scoped."""
    if product_id is not None and product_id != state.product:
        return ToolError(error="cross-product corpus search is not allowed").model_dump(
            exclude_none=True
        )
    hits = await grep_indexed_chunks(
        indexer=state.ctx.indexer,
        product_id=state.product,
        query=query,
        limit=top_k,
    )
    return GrepResponse(
        query=query,
        hits=[
            GrepHitResponse(
                score=hit.score,
                anchor=f"{hit.file}:{hit.line}",
                content=hit.excerpt,
                source="grep",
            )
            for hit in hits
        ],
    ).model_dump(mode="json")


async def hybrid_search_corpus(
    state: ToolState,
    *,
    query: str,
    product_id: str | None = None,
    top_k: int = 5,
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
) -> dict:
    """Hybrid retrieval (dense + BM25 + rerank) against the indexed corpus."""
    if product_id is not None and product_id != state.product:
        return ToolError(error="cross-product corpus search is not allowed").model_dump(
            exclude_none=True
        )
    pid = state.product
    result = await retrieve(
        ctx=state.ctx, product_id=pid, query=query, top_k=top_k, mode="auto"
    )
    rendered = _render_retrieval(
        result, detail=detail, max_response_tokens=max_response_tokens
    )
    return RetrievalResponse.model_validate(rendered).model_dump(mode="json")


async def evidence_search_corpus(
    state: ToolState,
    *,
    query: str,
    product_id: str | None = None,
    top_k: int = 10,
    current_file: str | None = None,
    mode: str = "auto",
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
    debug: bool = False,
) -> dict:
    """EvidenceGraphRAG retrieval across hybrid, grep, repo-map, graph, and skills."""
    if product_id is not None and product_id != state.product:
        return ToolError(error="cross-product corpus search is not allowed").model_dump(
            exclude_none=True
        )
    result = await retrieve_evidence(
        ctx=state.ctx,
        graph_store=state.graph_store,
        product_id=state.product,
        query=query,
        top_k=top_k,
        current_file=current_file,
        query_mode=mode,  # type: ignore[arg-type]
        skills=[s for s in state.store.iter_skills() if s.product == state.product],
        budget_ms=state.config.retrieval.interactive_budget_ms,
    )
    return _render_evidence_set(
        result, detail=detail, max_response_tokens=max_response_tokens, debug=debug
    )


async def ask_product_graph(
    state: ToolState,
    *,
    query: str,
    history: list[dict] | None = None,
    current_file: str | None = None,
    max_depth: int = 3,
    top_k: int = 8,
    mode: str = "auto",
    synthesize: bool = True,
) -> dict:
    """Generic product GraphRAG: graph expansion + cited evidence + answer."""
    chat = state.graphrag_chat if synthesize else None
    answer = await answer_graph_rag(
        ctx=state.ctx,
        graph_store=state.graph_store,
        chat=chat,
        product_id=state.product,
        request=GraphRAGQuery(
            query=query,
            history=history or [],
            current_file=current_file,
            mode=mode,  # type: ignore[arg-type]
            max_depth=max_depth,
            top_k=top_k,
        ),
    )
    return answer.model_dump(mode="json")


# ---------------------------------------------------------------- resource helpers


async def skill_hierarchy(state: ToolState) -> dict:
    return {
        "product": state.product,
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "tier": s.tier,
                "confidence": s.confidence,
            }
            for s in state.store.iter_skills()
            if s.product == state.product
        ],
    }


async def skill_markdown(state: ToolState, *, name: str) -> str:
    for s in state.store.iter_skills():
        if s.product != state.product:
            continue
        if s.name == name:
            return s.body
    raise ValueError(f"skill not found: {name}")


async def corpus_summary(state: ToolState, *, product_id: str) -> dict:
    if product_id != state.product:
        return ToolError(
            product_id=product_id,
            error="cross-product corpus access is not allowed",
        ).model_dump(exclude_none=True)
    indexer = state.ctx.indexer
    try:
        code_count = await indexer.count(product_id=product_id, vector_kind="code")
        text_count = await indexer.count(product_id=product_id, vector_kind="text")
    except Exception as e:
        log.warning("corpus count failed: %s", e)
        return {"product_id": product_id, "error": str(e)}
    return {
        "product_id": product_id,
        "chunk_count": code_count + text_count,
        "code_chunk_count": code_count,
        "doc_chunk_count": text_count,
        "source_count": 0,
    }


# ---------------------------------------------------------------- helpers


def _matches_file_globs(file_path: str | None, globs: list[str]) -> bool:
    """Match `applies_to.files` globs against a repo-relative path.

    Skill authors should prefer recursive patterns such as `**/*.py`; those
    preserve the same intent under Python 3.13 `PurePath.full_match()` and the
    older `PurePath.match()` fallback.
    """
    if not globs:
        return True
    if file_path is None:
        return True
    p = PurePath(file_path)
    # Keep this helper usable in older local envs even though CI targets 3.13+.
    full_match = getattr(p, "full_match", None)
    if full_match is not None:
        return any(full_match(g) for g in globs)
    return any(p.match(g) for g in globs)


def _matches_context(requested: str, skill_contexts: list[str]) -> bool:
    if not skill_contexts:
        return True
    if not requested or requested == "general":
        return True
    return requested in skill_contexts


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _section_headings(body: str) -> list[str]:
    return [
        line.removeprefix("## ").strip()
        for line in body.splitlines()
        if line.startswith("## ")
    ]


def _extract_section(body: str, section: str) -> str | None:
    """Return the H2 subtree whose heading matches `section` (case-insensitive
    substring match), or None."""
    lines = body.splitlines()
    want = section.strip().lower()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and want in line.removeprefix("## ").strip().lower():
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def _first_paragraph(body: str) -> str:
    for block in body.split("\n\n"):
        stripped = block.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:300]
    return ""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _merge_adjacent_hits(hits: list[dict], *, max_gap_lines: int = 3) -> list[dict]:
    """Merge overlapping/adjacent hits from the same file into one span.

    Keeps the highest score and the first anchor; the union of covered line
    ranges is reported in `merged_spans` so citations stay line-precise."""
    by_file: dict[str, list[dict]] = {}
    passthrough: list[dict] = []
    for hit in hits:
        file, line, end = hit.get("_file"), hit.get("_line"), hit.get("_end_line")
        if not file or line is None:
            passthrough.append(hit)
            continue
        hit["_end_line"] = end if end is not None else line
        by_file.setdefault(file, []).append(hit)

    merged: list[dict] = list(passthrough)
    for _file, group in by_file.items():
        group.sort(key=lambda h: h["_line"])
        current = dict(group[0])
        spans = [(current["_line"], current["_end_line"])]
        for nxt in group[1:]:
            if nxt["_line"] <= current["_end_line"] + max_gap_lines:
                if nxt["_end_line"] > current["_end_line"]:
                    current["_end_line"] = nxt["_end_line"]
                    if nxt.get("content") and nxt["content"] not in (current.get("content") or ""):
                        current["content"] = (
                            f"{current.get('content') or ''}\n…\n{nxt['content']}"
                        ).strip()
                current["score"] = max(current["score"], nxt["score"])
                spans.append((nxt["_line"], nxt["_end_line"]))
            else:
                if len(spans) > 1:
                    current["merged_spans"] = [f"{s}-{e}" for s, e in spans]
                merged.append(current)
                current = dict(nxt)
                spans = [(current["_line"], current["_end_line"])]
        if len(spans) > 1:
            current["merged_spans"] = [f"{s}-{e}" for s, e in spans]
        merged.append(current)

    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged


def _pack_hits(
    hits: list[dict],
    *,
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
) -> list[dict]:
    """Budget-aware response shaping: content for the top hits while the token
    budget lasts, anchor + context_path only for the tail. `detail="anchor"`
    strips content entirely; `"full"` never truncates individual excerpts."""
    packed: list[dict] = []
    spent = 0
    for hit in hits:
        out = {k: v for k, v in hit.items() if not k.startswith("_")}
        content = out.get("content")
        if detail == "anchor":
            out["content"] = None
        elif content:
            if detail == "excerpt" and len(content) > _EXCERPT_CHAR_CAP:
                content = content[:_EXCERPT_CHAR_CAP] + "…"
            cost = _estimate_tokens(content)
            if spent + cost > max_response_tokens:
                out["content"] = None  # anchor-only tail — client can grep/read
            else:
                out["content"] = content
                spent += cost
        packed.append(out)
    return packed


def _render_retrieval(
    result,
    *,
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
) -> dict:
    hits = []
    for h in result.hits:
        payload = h.payload or {}
        hits.append(
            {
                "score": h.score,
                "source": h.source,
                "anchor": f'{payload.get("resource_uri","?")}:'
                          f'{payload.get("start_line","?")}',
                "context_path": payload.get("context_path"),
                "content": payload.get("content"),
                "_file": payload.get("resource_uri"),
                "_line": payload.get("start_line"),
                "_end_line": payload.get("end_line"),
            }
        )
    hits = _merge_adjacent_hits(hits)
    return {
        "reranked": result.reranked,
        "hits": _pack_hits(hits, detail=detail, max_response_tokens=max_response_tokens),
    }


def _render_evidence_set(
    result: EvidenceSet,
    *,
    detail: str = "excerpt",
    max_response_tokens: int = DEFAULT_RESPONSE_TOKENS,
    debug: bool = False,
) -> dict:
    hits = []
    for candidate in result.candidates:
        hit = {
            "score": candidate.score,
            "source": candidate.channel,
            "role": candidate.role,
            "anchor": candidate.anchor,
            "context_path": candidate.context_path,
            "content": candidate.excerpt,
            "_file": candidate.file or None,
            "_line": candidate.line if candidate.file else None,
            "_end_line": candidate.end_line,
        }
        if debug:
            hit["graph_node_ids"] = candidate.graph_node_ids
            hit["metadata"] = candidate.metadata
        hits.append(hit)
    hits = _merge_adjacent_hits(hits)
    out = {
        "query": result.query,
        "shape": result.understanding.shape,
        "coverage": result.coverage.model_dump(mode="json"),
        "reranked": result.reranked,
        "hits": _pack_hits(hits, detail=detail, max_response_tokens=max_response_tokens),
    }
    if debug:
        out["trace"] = [step.model_dump(mode="json") for step in result.trace]
    return out
