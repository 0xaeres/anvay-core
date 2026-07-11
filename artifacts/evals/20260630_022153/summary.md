# Anvay Eval Run 20260630_022153

- Status: **FAIL**
- Generated: 2026-06-30T02:26:02.903879+00:00
- Modes: auto | top_k=10 | limit=5
- Judge model: `deepseek-ai/DeepSeek-V4-Pro`

## zod  (PASS, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1.000 | 0.833 | 0.871 | 0.956 | 0.700 | 0.573 | 0.633 | 1.000 | 9337ms |

## guava  (FAIL, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.800 | 0.517 | 0.586 | 0.920 | 0.500 | 0.364 | 0.460 | 1.000 | 22255ms |

