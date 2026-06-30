# Anvay Eval Run 20260630_023034

- Status: **FAIL**
- Generated: 2026-06-30T02:36:02.852565+00:00
- Modes: auto | top_k=10 | limit=5
- Judge model: `deepseek-ai/DeepSeek-V4-Pro`

## anvay  (PASS, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 0.900 | 0.900 | 0.861 | 0.933 | 0.650 | 0.896 | 1.000 | 1.000 | 28902ms |

## zod  (FAIL, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1.000 | 0.733 | 0.797 | 0.916 | 0.750 | 0.583 | 0.433 | 1.000 | 12636ms |

## guava  (FAIL, n=5)

| mode | recall_at_k | mrr | ndcg_at_k | faithfulness | answer_correctness | context_precision | context_recall | graph_hit_rate | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| auto | 1.000 | 0.717 | 0.786 | 0.967 | 0.500 | 0.431 | 0.460 | 1.000 | 6330ms |

