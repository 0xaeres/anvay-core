# Anvay Eval Run 20260629_014650

- Status: FAIL
- Generated: 2026-06-29T01:50:09.689170+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260629_014650`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `zod` | PASS | recall_at_k=0.9412, mrr=0.5349, ndcg_at_k=0.6629 | `artifacts/evals/20260629_014650/retrieval.json` |
| rag | `zod` | FAIL | n=17, faithfulness=1, answer_correctness=0.8353, context_recall=0.4216 | `artifacts/evals/20260629_014650/rag.json` |
