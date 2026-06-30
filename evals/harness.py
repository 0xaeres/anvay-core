"""Unified Anvay eval harness.

One command, one entrypoint, one report. Evaluates the *shipping* retrieval
path — :func:`anvay.retrieval.evidence.retrieve_evidence` (hybrid + grep +
repo-map + **graph-local** + summaries) — not the low-level pipeline. For each
product and each evidence ``query_mode`` it computes:

- **Retrieval (deterministic, free):** recall@k, mrr, ndcg@k from
  ``expected_files`` (reuses :mod:`evals.metrics`).
- **Answer quality (RAGAS, LLM-judged):** faithfulness, answer_correctness,
  context_precision, context_recall (via :class:`evals.ragas_adapter.RagasJudge`).
- **Diagnostics:** per-mode latency, graph-hit rate, candidate count, and the
  list of retrieval misses — the raw material for RAG-pipeline suggestions.

Only the ``auto`` evidence mode is evaluated; the query-rewrite ablation was
removed (proven not to earn its extra LLM call). The multi-mode delta rendering
remains for any future ablation.

The graph store is wired in (``create_graph_store``) so the graph channel is
exercised exactly as it is in production — it is otherwise inert.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, computed_field

from anvay.config import AnvayConfig
from anvay.llm.client import ChatClient
from anvay.retrieval.evidence import EvidenceCandidate, retrieve_evidence
from anvay.retrieval.pipeline import RetrievalContext
from evals.corpus import PRODUCTS, ProductEval
from evals.dataset import GoldenItem, load_golden
from evals.metrics import first_match_rank, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from evals.ragas_adapter import METRIC_KEYS, RagasJudge

log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("artifacts/evals")
DEFAULT_MODES = ("auto",)
DEFAULT_TOP_K = 10

# Synthesis prompt: answer strictly from retrieved excerpts so faithfulness
# measures the *context*, not the answerer's parametric knowledge.
_SYNTH_PROMPT = (
    "You are a code-search assistant. Answer the QUESTION using ONLY the "
    "provided CONTEXTS (2-4 sentences). Never introduce facts not in the "
    "contexts. If the contexts are silent, say so."
)


# ------------------------------------------------------------ thresholds


class Thresholds(BaseModel):
    """Hard gates applied to the default (first) mode of each product.

    **Deterministic retrieval metrics** (recall/ndcg/mrr) gate on pure file-match
    math — 100% reproducible, stable at n=5.

    **LLM-judged metrics** (answer_correctness, context_recall) gate on results
    measured at n=15 (guava post-fix javadoc-attachment baseline, run 20260630,
    HQE off). At n=5 these swung ~0.2 between identical runs and were diagnostic
    only; n=15 is stable enough to gate. Faithfulness and context_precision remain
    diagnostic — faithfulness is near-ceiling (>0.95) and adds little signal;
    context_precision is noisy even at n=15.

    Calibrated with margin below the post-fix per-product mins:
    - Deterministic: observed mins recall 0.90, ndcg 0.79, mrr 0.72 (all products).
    - LLM: guava n=15 post-fix: answer_correctness 0.65, context_recall 0.57.
      Thresholds set 0.15 / 0.12 below. Ratchet up as corpora improve.
    """

    recall_at_k: float = 0.75
    ndcg_at_k: float = 0.70
    mrr: float = 0.60
    answer_correctness: float = 0.50
    context_recall: float = 0.45


# ------------------------------------------------------------ report shapes


class ModeMetrics(BaseModel):
    mode: str
    n: int
    # retrieval (deterministic)
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    # answer quality (RAGAS; None when the judge produced no score)
    faithfulness: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    # diagnostics
    avg_latency_ms: float = 0.0
    avg_candidates: float = 0.0
    graph_hit_rate: float = 0.0
    misses: list[str] = Field(default_factory=list)


class ProductResult(BaseModel):
    product_id: str
    n: int
    modes: list[ModeMetrics]
    passed: bool


class EvalRunArtifact(BaseModel):
    run_id: str
    generated_at: str
    config_path: str
    config_fingerprint: dict
    top_k: int
    limit: int | None
    modes: list[str]
    thresholds: Thresholds
    products: list[ProductResult]
    output_dir: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.products)


# ------------------------------------------------------------ per-item


class _ItemScore(BaseModel):
    id: str
    query: str
    category: str
    recall_at_k: float
    ndcg_at_k: float
    reciprocal_rank: float
    latency_ms: float
    n_candidates: int
    graph_used: bool
    top_files: list[str]
    answer: str = ""
    faithfulness: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


# ------------------------------------------------------------ runner


async def run_eval(
    *,
    config: AnvayConfig,
    config_path: Path,
    products: list[ProductEval],
    modes: tuple[str, ...] = DEFAULT_MODES,
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    thresholds: Thresholds | None = None,
    judge_model: str | None = None,
) -> EvalRunArtifact:
    thresholds = thresholds or Thresholds()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = RetrievalContext.from_config(config)
    judge = RagasJudge(config, judge_model=judge_model)
    judge_model_used = judge.model
    answerer = ChatClient.from_cfg(config.models.council, role="eval_answerer")
    graph_store = _make_graph_store(config)

    product_results: list[ProductResult] = []
    try:
        for product in products:
            result = await _run_product(
                product=product,
                config=config,
                ctx=ctx,
                judge=judge,
                answerer=answerer,
                graph_store=graph_store,
                modes=modes,
                top_k=top_k,
                limit=limit,
                thresholds=thresholds,
                run_dir=run_dir,
            )
            if result is not None:
                product_results.append(result)
    finally:
        await answerer.aclose()
        await judge.aclose()
        await ctx.aclose()
        await _close_graph_store(graph_store)

    artifact = EvalRunArtifact(
        run_id=run_id,
        generated_at=datetime.now(UTC).isoformat(),
        config_path=str(config_path),
        config_fingerprint=_config_fingerprint(config, judge_model=judge_model_used),
        top_k=top_k,
        limit=limit,
        modes=list(modes),
        thresholds=thresholds,
        products=product_results,
        output_dir=str(run_dir),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render_markdown(artifact), encoding="utf-8")
    return artifact


async def _run_product(
    *,
    product: ProductEval,
    config: AnvayConfig,
    ctx: RetrievalContext,
    judge: RagasJudge,
    answerer: ChatClient,
    graph_store: object | None,
    modes: tuple[str, ...],
    top_k: int,
    limit: int | None,
    thresholds: Thresholds,
    run_dir: Path,
) -> ProductResult | None:
    if product.golden_path is None or not Path(product.golden_path).exists():
        log.warning("product %s has no golden dataset; skipping", product.product_id)
        return None
    golden = load_golden(Path(product.golden_path))
    if limit:
        golden = golden[:limit]
    if not golden:
        return None

    mode_metrics: list[ModeMetrics] = []
    detail: dict[str, list[dict]] = {}
    for mode in modes:
        scores = await asyncio.gather(
            *[
                _score_item(
                    item,
                    product_id=product.product_id,
                    ctx=ctx,
                    judge=judge,
                    answerer=answerer,
                    graph_store=graph_store,
                    top_k=top_k,
                    mode=mode,
                )
                for item in golden
            ]
        )
        mode_metrics.append(_aggregate(mode, scores, top_k))
        detail[mode] = [s.model_dump() for s in scores]

    (run_dir / f"{product.product_id}.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    passed = _passes(mode_metrics[0], thresholds) if mode_metrics else False
    return ProductResult(
        product_id=product.product_id,
        n=len(golden),
        modes=mode_metrics,
        passed=passed,
    )


async def _score_item(
    item: GoldenItem,
    *,
    product_id: str,
    ctx: RetrievalContext,
    judge: RagasJudge,
    answerer: ChatClient,
    graph_store: object | None,
    top_k: int,
    mode: str,
) -> _ItemScore:
    started = time.perf_counter()
    evidence = await retrieve_evidence(
        ctx=ctx,
        product_id=product_id,
        query=item.query,
        top_k=top_k,
        graph_store=graph_store,
        query_mode=mode,  # type: ignore[arg-type]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    candidates = evidence.candidates

    # --- retrieval metrics (deterministic) ---
    relevant = {f.lower() for f in item.expected_files}
    labels = _retrieved_labels(candidates, relevant)
    recall = recall_at_k(labels, relevant, top_k) if relevant else 1.0
    ndcg = ndcg_at_k(labels, relevant, top_k) if relevant else 1.0
    rank = first_match_rank(labels, list(relevant), lambda lab, rel: lab in set(rel))
    rr = reciprocal_rank(rank)
    graph_used = any(_is_graph(c) for c in candidates)

    # --- answer quality (RAGAS) ---
    # Score the candidates the pipeline actually delivers (top_k). An earlier
    # cost cut to 4 measured <half of them and artificially suppressed
    # context_recall; 8 restores measurement fidelity. Judge cache + --limit keep
    # the (per-context) cost in check.
    contexts = [c.excerpt for c in candidates if c.excerpt][:8]
    answer = ""
    ragas = None
    if contexts:
        answer = await _synthesize(answerer, item.query, contexts)
        ragas = await judge.score(
            question=item.query,
            answer=answer,
            reference=item.expected_answer,
            contexts=contexts,
        )

    return _ItemScore(
        id=item.id,
        query=item.query,
        category=item.category,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        reciprocal_rank=rr,
        latency_ms=latency_ms,
        n_candidates=len(candidates),
        graph_used=graph_used,
        top_files=[c.file for c in candidates[:5] if c.file],
        answer=answer,
        faithfulness=ragas.faithfulness if ragas else None,
        answer_correctness=ragas.answer_correctness if ragas else None,
        context_precision=ragas.context_precision if ragas else None,
        context_recall=ragas.context_recall if ragas else None,
    )


async def _synthesize(answerer: ChatClient, question: str, contexts: list[str]) -> str:
    msg = [
        {"role": "system", "content": _SYNTH_PROMPT},
        {
            "role": "user",
            "content": f"QUESTION:\n{question}\n\nCONTEXTS:\n"
            + "\n---\n".join(c[:1000] for c in contexts[:6]),
        },
    ]
    resp = await answerer.chat(msg, temperature=0.0, max_tokens=400)
    return resp.content.strip()


# ------------------------------------------------------------ scoring helpers


def _retrieved_labels(candidates: list[EvidenceCandidate], relevant: set[str]) -> list[str]:
    """Ordered-unique labels: matched expected file when a candidate hits one,
    else the candidate's own file (so non-matches still occupy a rank)."""
    labels: list[str] = []
    for cand in candidates:
        match = _match_file(cand.file, relevant)
        labels.append(match or (cand.file or cand.chunk_id).lower())
    return list(dict.fromkeys(labels))


def _match_file(uri: str, relevant: set[str]) -> str | None:
    """Suffix match a retrieved file URI against expected paths (longest first)."""
    normalized = (uri or "").lower().split(":", 1)[0].strip("/")
    if not normalized:
        return None
    name = normalized.rsplit("/", 1)[-1]
    for path in sorted(relevant, key=lambda p: (-len(p), p)):
        expected = path.strip("/")
        if normalized == expected or normalized.endswith(f"/{expected}"):
            return expected
        if "/" not in expected and name == expected:
            return expected
    return None


def _is_graph(c: EvidenceCandidate) -> bool:
    return c.channel == "graph" or bool(c.graph_node_ids)


def _aggregate(mode: str, scores: list[_ItemScore], top_k: int) -> ModeMetrics:
    n = len(scores)
    if n == 0:
        return ModeMetrics(mode=mode, n=0)
    return ModeMetrics(
        mode=mode,
        n=n,
        recall_at_k=round(mean([s.recall_at_k for s in scores]), 4),
        mrr=round(mean([s.reciprocal_rank for s in scores]), 4),
        ndcg_at_k=round(mean([s.ndcg_at_k for s in scores]), 4),
        faithfulness=_opt_mean(s.faithfulness for s in scores),
        answer_correctness=_opt_mean(s.answer_correctness for s in scores),
        context_precision=_opt_mean(s.context_precision for s in scores),
        context_recall=_opt_mean(s.context_recall for s in scores),
        avg_latency_ms=round(mean([s.latency_ms for s in scores]), 1),
        avg_candidates=round(mean([float(s.n_candidates) for s in scores]), 1),
        graph_hit_rate=round(mean([1.0 if s.graph_used else 0.0 for s in scores]), 4),
        misses=[s.query for s in scores if s.recall_at_k == 0.0],
    )


def _opt_mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _passes(m: ModeMetrics, t: Thresholds) -> bool:
    # Deterministic retrieval metrics always gate.
    # LLM-judged metrics gate when a score was produced (None = judge unavailable → skip).
    # See Thresholds docstring for calibration details.
    checks = [
        m.recall_at_k >= t.recall_at_k,
        m.ndcg_at_k >= t.ndcg_at_k,
        m.mrr >= t.mrr,
        m.answer_correctness is None or m.answer_correctness >= t.answer_correctness,
        m.context_recall is None or m.context_recall >= t.context_recall,
    ]
    return all(checks)


# ------------------------------------------------------------ markdown report


def render_markdown(a: EvalRunArtifact) -> str:
    lines = [
        f"# Anvay Eval Run {a.run_id}",
        "",
        f"- Status: **{'PASS' if a.passed else 'FAIL'}**",
        f"- Generated: {a.generated_at}",
        f"- Modes: {', '.join(a.modes)} | top_k={a.top_k}"
        + (f" | limit={a.limit}" if a.limit else ""),
        f"- Judge model: `{a.config_fingerprint.get('judge_model', '?')}`",
        "",
    ]
    metric_cols = [
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "graph_hit_rate",
        "avg_latency_ms",
    ]
    for p in a.products:
        lines.append(f"## {p.product_id}  ({'PASS' if p.passed else 'FAIL'}, n={p.n})")
        lines.append("")
        header = "| mode | " + " | ".join(metric_cols) + " |"
        sep = "|" + "---|" * (len(metric_cols) + 1)
        lines += [header, sep]
        for m in p.modes:
            lines.append("| " + m.mode + " | " + " | ".join(_fmt(m, c) for c in metric_cols) + " |")
        # ablation delta vs first mode
        if len(p.modes) > 1:
            base = p.modes[0]
            lines.append("")
            lines.append(f"**Δ vs `{base.mode}`:**")
            for m in p.modes[1:]:
                lines.append("")
                lines.append(f"- `{m.mode}` vs `{base.mode}`: " + ", ".join(_delta(base, m)))
        lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(m: ModeMetrics, col: str) -> str:
    val = getattr(m, col)
    if val is None:
        return "—"
    if col == "avg_latency_ms":
        return f"{val:.0f}ms"
    return f"{val:.3f}"


def _delta(base: ModeMetrics, other: ModeMetrics) -> list[str]:
    keys = ["recall_at_k", "ndcg_at_k", *METRIC_KEYS]
    out = []
    for k in keys:
        b, o = getattr(base, k), getattr(other, k)
        if b is None or o is None:
            continue
        d = o - b
        out.append(f"{k} {d:+.3f}")
    out.append(f"latency {other.avg_latency_ms - base.avg_latency_ms:+.0f}ms")
    return out


# ------------------------------------------------------------ graph store


def _make_graph_store(config: AnvayConfig) -> object | None:
    try:
        from anvay.graph.store import create_graph_store

        return create_graph_store(config)
    except Exception as exc:  # graph channel stays inert if the store is down
        log.warning("graph store unavailable; graph channel inert: %s", exc)
        return None


async def _close_graph_store(store: object | None) -> None:
    if store is None:
        return
    closer = getattr(store, "aclose", None) or getattr(store, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("graph store close failed: %s", exc)


# ------------------------------------------------------------ fingerprint


def _redact_url(value: str) -> str:
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _config_fingerprint(config: AnvayConfig, *, judge_model: str) -> dict:
    return {
        "embedding_model": config.models.embedding.model,
        "reranker_model": config.models.reranker.model,
        "council_model": config.models.council.model,
        "judge_model": judge_model,
        "qdrant_url": _redact_url(config.vector_store.url),
    }


# ------------------------------------------------------------ resolution


def resolve_products(names: list[str]) -> list[ProductEval]:
    """Map product-id strings to registry entries; ``["all"]`` selects all."""
    if not names or names == ["all"]:
        return list(PRODUCTS.values())
    unknown = [n for n in names if n not in PRODUCTS]
    if unknown:
        raise ValueError(f"unknown product(s): {', '.join(unknown)}")
    return [PRODUCTS[n] for n in dict.fromkeys(names)]
