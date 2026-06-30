---
name: prompt-cache-layout
description: Convention for ordering LLM prompts so DeepInfra prefix caching fires
metadata:
  type: project
---

DeepInfra does automatic prefix caching (free, discounts cached **input** tokens only — not output). To benefit, prompts must put static/stable content as the prefix, variable content last.

Convention in this repo:
- Static-per-tier content (role instructions, citation rules, mandatory template, format invariants) lives in the **system message**.
- Per-session variable payload (repo map, summaries, signals, evidence, product/topic/name) lives in the **user message**, largest content last.

Applied to council nodes in `anvay/council/agents/skill.py`: synthesizer, repair (caches the template across its ≤3 attempts), eval-repair.

Enricher (`anvay/ingest/enricher.py`) contextual retrieval: every chunk of one doc shares the `<document>` prefix. Scheduling warms the prefix with the doc's first chunk, then fans the rest out against the warm cache (don't revert to plain `asyncio.gather` over all chunks — it races and re-bills the 7.5-30k-token doc per chunk).

Graph extractor (`anvay/graph/llm_extractor.py`) system message = static schema; already cache-optimal.

**Why:** synthesizer runs DeepSeek-V4-Pro (the cost driver, ~$0.11/run was the trigger); output tokens dominate so caching is an input-side secondary win — the model downgrade is the primary lever.
**How to apply:** when editing any council/enricher/graph prompt, keep static text in the system/prefix position; never lead a user message with variable data. Re-run `pytest -m eval` after any prompt restructure. See [[prompt-cache-layout]].
