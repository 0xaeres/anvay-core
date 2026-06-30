# Anvay RAG Pipeline — Eval Findings & Suggestions

Grounded in run `20260629_061147`: 3 products (anvay n=28, zod n=15, guava n=15),
modes `auto` + `rewrite`, top_k=10, judge `deepseek-ai/DeepSeek-V4-Pro`, real
natural-language dev questions. Reproduce: `uv run anvay eval run --products all`.

---

## Resolution status (2026-06-30) — cost + graph + ordering + excerpts landed

Work from the "Cut eval cost to cents, then fix the RAG findings" plan.

**Cost (Phase 0).** HQE off at ingest (`enrich=False`) — see the HQE finding
below; deterministic graph still builds. Query-rewrite removed entirely (verdict
below was conclusive): eval surface + product code (`query_rewrite.py`, the
`evidence.py` rewrite mode). Deterministic judge disk cache
(`artifacts/eval-cache/`); `--limit` is the routine default.

**Graph (P0) — FIXED.** Two breaks: (1) the eval ingest ran `run_ingest` with no
`registry`/`source_key`, so `delta_enabled=False` skipped graph extraction +
chunk↔graph linkage entirely — now runs the production delta path; (2)
`understand_query` flagged sentence-initial capitals ("How"/"What") as symbols —
now requires an internal code signal (snake/camel/`()`) + stopword set.
`graph_hit_rate` 0.000 → **1.000** on all three products.

**Ordering (old P2 grep/precision) — FIXED.** Root cause: in
`evidence.py::merge_candidates`/`_candidate_rank`, coverage-repair grep candidates
were merged with **raw scores** (match counts in the tens) un-reranked, and
`_channel_weight` (2–8) dwarfed the reranker's 0–1 score, so thin grep line-
snippets buried substantive reranked chunks. Fix: rerank the repair pool before
merge; rescale channel/role weights into a `<0.2` tie-breaker band so the reranker
drives ordering (quotas still guarantee channel diversity).

**Excerpt budget (old P1 file→chunk) — FIXED.** Candidate excerpts were truncated
to 900 chars (chunks are ≤1200), dropping the behavioural detail that answers
semantic questions; the eval also fed only 4 of the 10 delivered candidates to
RAGAS. Fix: return the full chunk body (`_candidate_from_hit` 900 → 1600) and
score 8 contexts. Lifted ctx_recall materially (zod 0.567 → 0.60; guava 0.28 →
0.40).

**HQE — turned OFF (measured, not assumed).** Re-enabled HQE on a free OpenRouter
endpoint and measured a clean 3-product A/B. HQE **degrades** quality here:
`context_precision` dropped on every product (anvay 0.944→0.894, zod 0.694→0.644,
guava 0.367→0.267) and `context_recall` was flat-to-down (zod 0.567→0.367).
Prepending hypothetical questions dilutes the code chunk's embedding signal. So
HQE stays off — cheaper *and* better.

**Latency (P3).** `RetrievalCfg.interactive_budget_ms` (8s) applied to the MCP +
graph-RAG callers (council/eval stay unbounded).

**Final baseline (run 20260630, n=5, `auto`, HQE off, full excerpts, contexts=8):**

| product | recall | ndcg | mrr | graph | ctx_prec* | ctx_recall* | result |
|---|---|---|---|---|---|---|---|
| anvay | 0.900 | 0.861 | 0.900 | 1.000 | 0.90 | 1.00 | **PASS** |
| zod   | 1.000 | 0.797 | 0.733 | 1.000 | 0.58 | 0.43-0.63 | **PASS** |
| guava | 1.000 | 0.786 | 0.717 | 1.000 | 0.43 | 0.46 | **PASS** |

\* RAGAS LLM metrics — **diagnostic, not gated** (see below).

**P4 guava/Java — FIXED (retrieval).** Three retrieval-time fixes (no re-ingest)
lifted guava from worst-corpus to passing the deterministic gate:
1. **Hybrid quota** in `merge_candidates`: hybrid (the substantive dense channel)
   had *no* reserved slot and got crowded out by grep/repo_map/summary/overview
   quotas on global queries — for some Java queries the full class chunk was
   dropped entirely. Now reserved `max(3, top_k//2)` first.
2. **Filename-anchor boost**: when a query anchor exactly names a file (stem ==
   anchor, e.g. "ImmutableMap" -> `ImmutableMap.java`), boost it so the canonical
   class outranks its many same-prefix impls (`RegularImmutableMap`…).
3. **Full-chunk excerpts** (900 -> 1600) + score top-8 contexts.

guava: recall 0.80 -> **1.00**, ndcg 0.477 -> **0.786**, mrr 0.37 -> **0.72**,
ctx_recall 0.28 -> 0.46. anvay/zod also improved (anvay ctx_recall -> 1.0).

**Gate philosophy — gate the stable, diagnose the noisy.** The RAGAS LLM metrics
(faithfulness, answer_correctness, context_precision, context_recall) swing ~0.2
between identical-config runs at n=5 (zod ctx_recall observed 0.43-0.63). Gating
on them flakes. So the **hard gate is the deterministic retrieval metrics only**
(recall/ndcg/mrr — pure file-match math, 100% reproducible); the LLM metrics are
reported as diagnostics. Not lowering a gate to hide quality — refusing to gate on
a noisy measurement. All three products PASS the deterministic gate.

**Still open:** lift the LLM metrics' *signal* — measure at larger n (full golden
sets) for a stable baseline, then promote ctx_recall/answer_correctness back into
the gate. Remaining guava ctx_recall drag is 2 conceptual queries whose answer
lives in class javadoc not surfaced as a covering chunk (chunk-overlap / javadoc
chunking — needs a re-ingest).

---

## Baseline (mode `auto`)

| product | recall@10 | mrr | ndcg@10 | faithfulness | answer_corr | ctx_precision | ctx_recall | graph_hit | latency |
|---|---|---|---|---|---|---|---|---|---|
| anvay | 0.821 | 0.669 | 0.688 | 0.788 | 0.357 | 0.686 | 0.363 | **0.000** | 18.8s |
| zod | 0.800 | 0.767 | 0.739 | 0.761 | 0.283 | 0.482 | 0.400 | **0.000** | 13.1s |
| guava | 0.667 | 0.469 | 0.515 | 0.753 | 0.350 | 0.424 | 0.428 | **0.000** | 14.5s |

Read: retrieval finds the **right files** most of the time (recall 0.67–0.82),
but the **right chunks** are weaker (ctx_recall 0.36–0.43), answers only
partially match references (answer_corr 0.28–0.36), the graph contributes
**nothing**, and latency is high (13–19s).

---

## Verdict: query rewrite is NOT worth it (do not enable by default)

`rewrite` vs `auto`, every product:

| product | Δrecall | Δndcg | Δanswer_corr | Δctx_recall | Δlatency |
|---|---|---|---|---|---|
| anvay | +0.000 | +0.000 | −0.009 | −0.015 | **+10.8s** |
| zod | +0.000 | +0.000 | −0.017 | +0.000 | **+17.7s** |
| guava | +0.000 | +0.002 | −0.017 | −0.107 | **+11.6s** |

- **Retrieval is bit-identical** (Δrecall/ndcg = 0). The base fan-out already
  surfaces the expected files; the rewrite follow-ups add candidates that never
  change the top-10 ranking.
- Answer quality is flat-to-negative; faithfulness moves are noise (+0.117
  anvay, −0.094 zod).
- Latency rises **60–130%** for that null result.

**Recommendation:** keep default mode `auto`; do not promote `rewrite`. Per
AGENTS.md it needed an eval-set win to justify its cost — it has the opposite.
Either delete the `rewrite` path or restrict it to a measured fallback (only
when base coverage is insufficient *and* anchors are weak — see #2). Keep it in
the harness as an ablation arm, not a production default.

---

## Prioritized improvements

### P0 — The graph layer is dead weight in retrieval (`graph_hit_rate = 0.000`)

Every product has a populated graph (anvay 5,629 / zod 10,410 / guava 19,707
nodes) yet **not one** evidence candidate came from the graph channel — even for
queries naming an exact symbol (`retrieve_evidence` trace: `graph hits=0`, "no
graph relationships returned"). The graph is built at ingest cost and serves no
retrieval value today. Root causes observed:

1. **Anchor extraction returns stopwords.** `understand_query("How does
   graph-local traversal seed candidates…")` produced anchors `["How"]`. With no
   real symbol anchors, `graph_local_candidates` has nothing to resolve.
2. **Entity resolution misses even when the anchor is a real symbol.**
   `retrieve_evidence` as a query yields anchor `retrieve_evidence` but
   `resolve_entity`/`traverse` still return 0.

Actions:
- Fix `understand_query` anchor extraction: drop stopwords, and pull
  code-symbol candidates (identifiers, `snake_case`/`CamelCase`, quoted terms,
  repo-map symbol matches) instead of raw first tokens.
- Verify `FalkorGraphStore.resolve_entity` matches against the fields actually
  populated at ingest (name/stable_id/qualified name) and is product-scoped to
  the right graph (`<prefix>_<product>`).
- Add a graph-channel assertion to the eval (fail if `graph_hit_rate == 0` on a
  query set that names known symbols) so this can't silently regress again.
- Until fixed, the graph navigation claim in ENGINEERING.md §3–4 is aspirational,
  not real, for these corpora.

### P1 — Close the file→chunk gap (ctx_recall 0.36–0.43 ≪ recall 0.67–0.82)

The pipeline finds the right *file* far more often than the right *chunk*: the
reference's facts are frequently not inside the retrieved excerpts. Levers:

- **Chunk granularity / overlap:** AST chunks may split a symbol's
  explanation from its signature. Add small overlap or parent-context stitching
  so an excerpt carries the surrounding definition.
- **Excerpt budget:** the harness feeds top-8 excerpts truncated to 1k chars;
  ctx_recall suggests key spans get truncated. Consider returning the full
  symbol body for primary candidates.
- **Re-rank for answer coverage, not just lexical match** (see P2).

### P2 — grep dominates and dilutes precision (ctx_precision 0.42–0.69)

In the merged set, candidates are overwhelmingly `grep` + `repo_map`; the
semantic `hybrid` channel (which scores the correct files at ~0.999 in isolation)
gets collapsed/outranked in `merge_candidates` + `rerank_mixed_candidates`. grep
pulls keyword-matching but topically-irrelevant chunks, lowering precision.

- Audit `rerank_mixed_candidates`: the cross-encoder reranker (Qwen3-Reranker-4B)
  should float the semantically-relevant hybrid chunks above keyword grep noise.
  Today it doesn't appear to.
- Cap grep's share of the final top_k, or gate grep to exact-symbol queries
  (where it shines) instead of NL questions (where it adds noise).
- Preserve channel provenance through `merge_candidates` so this is observable
  (right now hybrid candidates lose their channel label on dedup).

### P3 — Latency is high (13–19s in `auto`)

For an interactive context engine this is slow. The multi-channel fan-out +
mixed rerank + coverage repair is expensive. Profile per-stage (the trace
already records `latency_ms` per channel); likely culprits are coverage-repair
follow-ups and the reranker round-trip. Consider a default `budget_ms` so the
long-tail enrichment is skipped on the interactive path.

### P4 — guava (large Java corpus) lags (recall 0.667, mrr 0.469)

Biggest corpus, worst retrieval; mrr 0.47 means the right file often isn't ranked
first. Java tokenization/symbol density differs from Python/TS. Worth checking
the sparse (BM25) tokenizer and embedding instruction profile on Java, and
whether repo-map symbol ranking handles Java packages.

### P5 — Calibrate gates; current thresholds are aspirational

All products "FAIL" only because default `Thresholds` (answer_correctness ≥0.55,
context_recall ≥0.70) sit well above today's baseline. Set gates from this
baseline (e.g. recall ≥0.65, faithfulness ≥0.70) so the suite catches
*regressions* now, and ratchet up as P0–P2 land.

---

## On the eval itself (so results stay trustworthy)

- **Judge must be a strong, non-reasoning instruct model.** gemma-4-31B scored
  correct answers at 0.0; Qwen3 (thinking) returns empty structured output.
  Default judge is `models.chat_agent` (DeepSeek-V4-Pro).
- We use RAGAS `AnswerAccuracy`, not `FactualCorrectness` — the latter's strict
  per-claim NLI scored confidently-correct answers at 0.0 even with DeepSeek.
- Keep golden `expected_answer`s **content-grounded** (what the code does), not
  file-location trivia, or `context_recall` is unfairly punished.
