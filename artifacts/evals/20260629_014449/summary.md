# Anvay Eval Run 20260629_014449

- Status: FAIL
- Generated: 2026-06-29T01:46:50.537965+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260629_014449`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `guava` | PASS | recall_at_k=1, mrr=0.8843, ndcg_at_k=1.718 | `artifacts/evals/20260629_014449/retrieval.json` |
| rag | `guava` | FAIL | n=17, faithfulness=0.8471, answer_correctness=0.8353, context_recall=0.8824 | `artifacts/evals/20260629_014449/rag.json` |
