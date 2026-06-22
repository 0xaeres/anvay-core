# Anvay Eval Run 20260621_125331

- Status: PASS
- Generated: 2026-06-21T12:55:37.246338+00:00
- Config: `anvay.yaml`
- Output: `artifacts/evals/20260621_125331`

| Suite | Product | Status | Metrics | Output |
|---|---|---|---|---|
| retrieval | `synthetic` | PASS | recall_at_k=1, mrr=1 | `artifacts/evals/20260621_125331/retrieval.json` |
| rag | `synthetic` | PASS | n=10, faithfulness=1, answer_relevancy=0.98, context_recall=1 | `artifacts/evals/20260621_125331/rag.json` |
| code | `synthetic` | PASS | n=10, ndcg_at_10=1.061, recall_at_10=1.133, pairwise_preference_accuracy=1, pairwise_n=10 | `artifacts/evals/20260621_125331/code.json` |
