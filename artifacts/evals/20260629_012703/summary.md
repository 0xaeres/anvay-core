# Anvay Eval Run 20260629_012703

- Status: FAIL
- Generated: 2026-06-29T01:44:49.405674+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260629_012703`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `anvay` | PASS | recall_at_k=0.8772, mrr=0.4794, ndcg_at_k=1.41 | `artifacts/evals/20260629_012703/retrieval.json` |
| rag | `anvay` | FAIL | n=15, faithfulness=0.9333, answer_correctness=0.9667, context_recall=0.4667 | `artifacts/evals/20260629_012703/rag.json` |
