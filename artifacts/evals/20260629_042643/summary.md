# Anvay Eval Run 20260629_042643

- Status: FAIL
- Generated: 2026-06-29T04:48:52.956382+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260629_042643`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `anvay` | PASS | recall_at_k=0.8772, mrr=0.4696, ndcg_at_k=1.373 | `artifacts/evals/20260629_042643/retrieval.json` |
| rag | `anvay` | FAIL | n=15, faithfulness=0.9333, answer_correctness=0.9667, context_recall=0.4667 | `artifacts/evals/20260629_042643/rag.json` |
