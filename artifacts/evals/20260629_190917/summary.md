# Anvay Eval Run 20260629_190917

- Status: **FAIL**
- Generated: 2026-06-29T19:12:55.009188+00:00
- Modes: auto | top_k=10 | limit=5
- Judge model: `deepseek-ai/DeepSeek-V4-Pro`

## zod  (PASS, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1.000 | 0.840 | 0.877 | 0.764 | 0.500 | 0.667 | 0.600 | 1.000 | 12892ms |

## guava  (FAIL, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.800 | 0.373 | 0.477 | 0.767 | 0.450 | 0.324 | 0.400 | 1.000 | 7200ms |

