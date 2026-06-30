# Anvay Eval Run 20260629_061147

- Status: **FAIL**
- Generated: 2026-06-29T06:25:08.389446+00:00
- Modes: auto, rewrite | top_k=10
- Judge model: `deepseek-ai/DeepSeek-V4-Pro`

## anvay  (FAIL, n=28)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.821 | 0.669 | 0.688 | 0.788 | 0.357 | 0.686 | 0.363 | 0.000 | 18772ms |
| rewrite | 0.821 | 0.669 | 0.688 | 0.905 | 0.348 | 0.675 | 0.348 | 0.000 | 29616ms |

**Δ vs `auto`:**

- `rewrite` vs `auto`: recall_at_k +0.000, ndcg_at_k +0.000, faithfulness +0.117, answer_correctness -0.009, context_precision -0.011, context_recall -0.015, latency +10844ms

## zod  (FAIL, n=15)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.800 | 0.767 | 0.739 | 0.761 | 0.283 | 0.482 | 0.400 | 0.000 | 13116ms |
| rewrite | 0.800 | 0.767 | 0.739 | 0.667 | 0.267 | 0.502 | 0.400 | 0.000 | 30781ms |

**Δ vs `auto`:**

- `rewrite` vs `auto`: recall_at_k +0.000, ndcg_at_k +0.000, faithfulness -0.094, answer_correctness -0.017, context_precision +0.020, context_recall +0.000, latency +17665ms

## guava  (FAIL, n=15)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.667 | 0.469 | 0.515 | 0.753 | 0.350 | 0.424 | 0.428 | 0.000 | 14506ms |
| rewrite | 0.667 | 0.471 | 0.517 | 0.811 | 0.333 | 0.439 | 0.321 | 0.000 | 26100ms |

**Δ vs `auto`:**

- `rewrite` vs `auto`: recall_at_k +0.000, ndcg_at_k +0.002, faithfulness +0.058, answer_correctness -0.017, context_precision +0.015, context_recall -0.107, latency +11593ms

