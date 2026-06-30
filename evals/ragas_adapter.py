"""RAGAS bridge for Anvay evals.

Wraps our OpenAI-compatible deepinfra endpoints as a RAGAS *instructor* LLM and
exposes the four answer-quality metrics we gate on. Single source of truth for
every RAGAS call so the harness never re-derives provider wiring.

We use RAGAS 0.4's ``ragas.metrics.collections`` metrics, which expose a clean
per-sample ``ascore(...)`` coroutine — no ``Dataset``/``evaluate()`` plumbing.
All four chosen metrics are **LLM-only** (no embeddings dependency):

- ``faithfulness``        — are answer claims grounded in the retrieved contexts?
- ``answer_correctness``  — AnswerAccuracy: holistic rating of the answer vs the
  reference. (We deliberately do **not** use FactualCorrectness: its strict
  per-claim NLI scored confidently-correct answers at 0.0 even with a strong
  judge, because terminology differences break claim entailment.)
- ``context_precision``   — are the retrieved contexts relevant (reference-aware)?
- ``context_recall``      — is the reference covered by the retrieved contexts?

Every metric is fail-soft: a judge error yields ``None`` for that metric rather
than crashing the run.

**Judge model matters.** RAGAS metrics rely on structured (instructor) output,
so the judge must be a capable, *non-reasoning* instruct model: a thinking model
(e.g. Qwen3 reasoning) spends its whole token budget on hidden reasoning and
returns empty structured output. The judge defaults to ``models.chat_agent``
(the strongest configured instruct model) and can be overridden per run; it
reuses the evaluator/council provider credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from anvay.config import AnvayConfig

log = logging.getLogger(__name__)

# Judge runs at temp 0 → deterministic, so a successful (model, metric, inputs)
# score is cacheable to disk. Dev re-runs on unchanged data become ~free.
DEFAULT_CACHE_DIR = Path("artifacts/eval-cache")

# Metric keys produced by :meth:`RagasJudge.score`. Kept in one place so the
# harness, report, and thresholds agree on naming.
METRIC_KEYS = (
    "faithfulness",
    "answer_correctness",
    "context_precision",
    "context_recall",
)


@dataclass
class RagasScores:
    faithfulness: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


class RagasJudge:
    """Holds the RAGAS instructor LLM + metric instances for a run."""

    def __init__(
        self,
        config: AnvayConfig,
        *,
        judge_model: str | None = None,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
    ) -> None:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerAccuracy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

        # Provider credentials come from evaluator/council; the judge *model*
        # defaults to the strongest configured instruct model (chat_agent).
        creds = config.models.evaluator or config.models.council
        self.model = judge_model or config.models.chat_agent.model or creds.model
        self._client = AsyncOpenAI(base_url=creds.base_url, api_key=creds.api_key)
        llm = llm_factory(self.model, provider="openai", client=self._client)

        self._faithfulness = Faithfulness(llm=llm)
        self._answer_correctness = AnswerAccuracy(llm=llm)
        self._context_precision = ContextPrecisionWithReference(llm=llm)
        self._context_recall = ContextRecall(llm=llm)

        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        await self._client.close()

    async def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str,
        contexts: list[str],
    ) -> RagasScores:
        """Score one sample. Each metric is independent and fail-soft."""
        scores = RagasScores()
        if contexts and answer:
            scores.faithfulness = await self._safe(
                "faithfulness",
                {"q": question, "a": answer, "ctx": contexts},
                lambda: self._faithfulness.ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts
                ),
            )
        if reference and answer:
            scores.answer_correctness = await self._safe(
                "answer_correctness",
                {"q": question, "a": answer, "ref": reference},
                lambda: self._answer_correctness.ascore(
                    user_input=question, response=answer, reference=reference
                ),
            )
        if reference and contexts:
            scores.context_precision = await self._safe(
                "context_precision",
                {"q": question, "ref": reference, "ctx": contexts},
                lambda: self._context_precision.ascore(
                    user_input=question, reference=reference, retrieved_contexts=contexts
                ),
            )
            scores.context_recall = await self._safe(
                "context_recall",
                {"q": question, "ref": reference, "ctx": contexts},
                lambda: self._context_recall.ascore(
                    user_input=question, retrieved_contexts=contexts, reference=reference
                ),
            )
        return scores

    def _cache_path(self, name: str, inputs: dict) -> Path | None:
        if self._cache_dir is None:
            return None
        payload = json.dumps(
            {"model": self.model, "metric": name, "inputs": inputs},
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{name}_{digest}.json"

    async def _safe(
        self,
        name: str,
        inputs: dict,
        coro_factory: Callable[[], Awaitable],
    ) -> float | None:
        path = self._cache_path(name, inputs)
        if path is not None and path.exists():
            try:
                return float(json.loads(path.read_text(encoding="utf-8"))["value"])
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("bad eval cache entry %s; recomputing", path)
        try:
            result = await coro_factory()
        except Exception as exc:  # fail-soft: one metric must not sink the run
            log.warning("ragas metric %s failed: %s", name, exc)
            return None
        value = getattr(result, "value", result)
        try:
            score = float(value)
        except (TypeError, ValueError):
            log.warning("ragas metric %s returned non-numeric %r", name, value)
            return None
        if path is not None:
            try:
                path.write_text(json.dumps({"value": score}), encoding="utf-8")
            except OSError as exc:  # pragma: no cover - defensive
                log.warning("could not write eval cache %s: %s", path, exc)
        return score
