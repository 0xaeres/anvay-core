# Anvay Eval Run 20260629_045413

- Status: FAIL
- Generated: 2026-06-29T05:00:46.654099+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260629_045413`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `zod` | PASS | recall_at_k=0.9412, mrr=0.5741, ndcg_at_k=0.6941 | `artifacts/evals/20260629_045413/retrieval.json` |
| rag | `zod` | FAIL | n=17, faithfulness=1, answer_correctness=0.8941, context_recall=0.4216 | `artifacts/evals/20260629_045413/rag.json` |
