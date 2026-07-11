# Anvay Eval Harness

One unified, production-grade eval for the thing that actually matters: **the
quality of the context Anvay delivers**. It scores the *shipping* retrieval path
— `anvay/retrieval/evidence.py::retrieve_evidence` (hybrid + grep + repo-map +
**graph-local** + summaries) — not a low-level approximation of it.

```bash
# install the opt-in eval deps once
uv sync --extra evals

# run everything (clones+ingests zod/guava if their index is empty)
uv run anvay eval run --products all
```

## Cost — keep a routine run in cents

The eval is engineered to be cheap to iterate on:

- **HQE is off** (`enrich=False`): the `ContextualEnricher` (3 hypothetical
  questions per code chunk + doc blurbs over ~16k chunks) was the ~$2/cycle
  driver on paid deepinfra **and** a 3-product eval showed it *degrades* quality
  here (ctx_precision down on every product, ctx_recall flat-to-down). So it
  stays off on both counts. The deterministic FalkorDB graph still builds.
- **Judge results are cached to disk** (`artifacts/eval-cache/`, keyed by
  `model+metric+inputs`). The judge runs at temp 0, so re-runs on unchanged data
  are ~free. Delete the dir to force re-scoring.
- **RAGAS scores the top 8 delivered contexts** (`context_precision` scores each
  separately — the call-heaviest metric; 8 keeps measurement faithful to the
  pipeline's top_k without scoring all 10).
- **Use `--limit`** (e.g. 10/product) for routine/CI runs; a full sweep (no
  limit) only before a release.
- **Ingest is cached** — never re-ingests a populated product without
  `--force-ingest`.

With HQE off + `--limit` + judge cache, a routine run is a few cents (judge only)
and a full DeepSeek sweep is sub-$0.50. Verify on the deepinfra dashboard.

Every run writes one directory under `artifacts/evals/<run_id>/`:

- `summary.json` — the full `EvalRunArtifact` (per product, per mode).
- `summary.md` — human-readable table + the **mode ablation delta**.
- `<product>.json` — per-item detail (answer, scores, retrieved files, latency).

## What it measures

Per golden item, per evidence `query_mode`:

| Group | Metric | How |
|---|---|---|
| Retrieval | `recall_at_k`, `mrr`, `ndcg_at_k` | deterministic, from `expected_files` (free) |
| Answer | `faithfulness` | RAGAS — answer claims grounded in retrieved contexts |
| Answer | `answer_correctness` | RAGAS `AnswerAccuracy` vs the reference answer |
| Answer | `context_precision` | RAGAS — are retrieved contexts relevant (reference-aware) |
| Answer | `context_recall` | RAGAS — is the reference covered by the contexts |
| Diagnostics | `graph_hit_rate`, `avg_candidates`, `avg_latency_ms`, `misses` | per-mode, for pipeline debugging |

RAGAS metrics come from `ragas.metrics.collections` (RAGAS 0.4) and run through
`evals/ragas_adapter.py`, which wraps our OpenAI-compatible deepinfra endpoints.

### Judge model — read this before trusting scores

RAGAS uses structured (instructor) output, so the judge **must be a capable,
non-reasoning instruct model**. Two lessons learned the hard way:

- A weak judge (e.g. `gemma-4-31B`) scored obviously-correct answers at 0.0.
- A *thinking* model (e.g. `Qwen3` reasoning) spends its whole token budget on
  hidden reasoning and returns **empty** structured output → metric failure.

The judge defaults to `models.chat_agent` (the strongest configured instruct
model). Override per run with `--judge-model <model>`. We also dropped RAGAS
`FactualCorrectness` (strict per-claim NLI scored correct answers at 0 even with
a strong judge) in favour of `AnswerAccuracy`.

## Query-rewrite ablation — removed

The harness used to score `--modes auto,rewrite` for a per-metric delta. The
verdict was conclusive (rewrite did not earn its extra LLM call), so the mode is
gone — only `auto` is evaluated. See `evals/SUGGESTIONS.md` for the data.

## Corpus

Products live in `evals/corpus.py`. Each has one golden file
`evals/products/<id>/golden.jsonl` of **realistic, natural-language dev
questions** grounded in verified source files (not single-symbol lexical
queries):

| Product | Source | Ingest |
|---|---|---|
| `anvay` | this repo's `anvay/` package | local |
| `zod` | `colinhacks/zod` → `packages/zod/src/v3` | shallow clone |
| `guava` | `google/guava` → `…/common/collect` | shallow clone |

Ingest runs with `enrich=False` (HQE off — see **Cost** above) and builds the
deterministic **FalkorDB graph**. Re-ingest is skipped when a product already has
points (`--force-ingest` to override).

## Golden schema

```json
{"id": "...", "query": "natural-language question",
 "expected_files": ["path/that/answers/it"],
 "expected_answer": "reference answer, grounded in those files",
 "category": "architecture|how-to|debugging|conceptual|api",
 "complexity": "simple|medium|hard"}
```

`expected_files` → retrieval metrics; `expected_answer` → RAGAS reference.
Keep references **content-grounded** (what the code does), not file-location
trivia — a reference the corpus can't actually answer tanks `context_recall`.

## CLI

```bash
uv run anvay eval run \
  --products guava,zod \      # or "all"
  --top-k 10 \
  --limit 10 \                # routine/smoke runs; drop for a full sweep
  --judge-model deepseek-ai/DeepSeek-V4-Pro \
  --no-ingest                 # query the live index as-is
```

Exit code is non-zero if any product fails its thresholds
(`evals/harness.py::Thresholds`), so it doubles as a CI gate.
