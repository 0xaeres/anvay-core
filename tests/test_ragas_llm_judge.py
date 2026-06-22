"""Unit tests for the RAGAS eval runner — LLM judge CoT parsing.

Covers the _llm_judge helper's ability to extract the `reasoning` field
(primary) or fall back to the legacy `notes` field, plus boundary cases.
No network calls needed.
"""

from __future__ import annotations

import pytest

from evals.judges import llm
from evals.judges.llm import (
    JudgeScore,
    PairwiseJudgment,
    evaluator_client,
    judge_answer_correctness,
    judge_faithfulness,
    judge_pairwise_preference,
    judge_score,
)
from evals.run_ragas import _llm_judge

# --------------------------------------------------------------------------- #
# Minimal stub judge
# --------------------------------------------------------------------------- #


class _StubJudge:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.messages = []

    async def chat_json(self, messages, **kwargs):
        self.messages.append(messages)
        return self._payload, None


class _FailingJudge:
    async def chat_json(self, messages, **kwargs):
        raise RuntimeError("timeout")


class _Config:
    class _Models:
        evaluator = None
        council = object()

    models = _Models()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_judge_extracts_reasoning_field() -> None:
    """New CoT schema: reasoning field is preferred over notes."""
    judge = _StubJudge(
        {"reasoning": "Claim 1 is supported. Claim 2 is absent.", "score": 0.5, "verdict": "partial"}
    )
    score, notes = await _llm_judge(judge, "system", "user")
    assert score == 0.5
    assert "Claim 1 is supported" in notes


@pytest.mark.asyncio
async def test_llm_judge_falls_back_to_notes_for_legacy_responses() -> None:
    """Backwards-compatible: old `notes` field still works if reasoning is absent."""
    judge = _StubJudge({"score": 0.8, "notes": "answer is grounded"})
    score, notes = await _llm_judge(judge, "system", "user")
    assert score == 0.8
    assert "grounded" in notes


@pytest.mark.asyncio
async def test_llm_judge_prefers_reasoning_over_notes_when_both_present() -> None:
    judge = _StubJudge(
        {"reasoning": "detailed reasoning", "score": 0.9, "notes": "short note"}
    )
    _, notes = await _llm_judge(judge, "system", "user")
    assert "detailed reasoning" in notes


@pytest.mark.asyncio
async def test_llm_judge_clamps_score_above_one() -> None:
    judge = _StubJudge({"reasoning": "ok", "score": 1.5})
    score, _ = await _llm_judge(judge, "system", "user")
    assert score == 1.0


@pytest.mark.asyncio
async def test_llm_judge_clamps_score_below_zero() -> None:
    judge = _StubJudge({"reasoning": "bad", "score": -0.3})
    score, _ = await _llm_judge(judge, "system", "user")
    assert score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_handles_missing_score() -> None:
    judge = _StubJudge({"reasoning": "no score field"})
    score, _ = await _llm_judge(judge, "system", "user")
    assert score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_returns_zero_on_exception() -> None:
    score, notes = await _llm_judge(_FailingJudge(), "system", "user")
    assert score == 0.0
    assert "judge error" in notes


def test_evaluator_client_uses_evaluator_or_council(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_from_cfg(model_cfg, *, role: str):
        seen["model_cfg"] = model_cfg
        seen["role"] = role
        return "client"

    monkeypatch.setattr(llm.ChatClient, "from_cfg", fake_from_cfg)

    assert evaluator_client(_Config(), role="ragas") == "client"
    assert seen == {"model_cfg": _Config.models.council, "role": "ragas"}


@pytest.mark.asyncio
async def test_judge_score_returns_typed_score() -> None:
    judge = _StubJudge({"reasoning": "ok", "score": "0.75", "verdict": "partial"})

    result = await judge_score(judge, "system", "user")

    assert result == JudgeScore(score=0.75, reasoning="ok", verdict="partial")


@pytest.mark.asyncio
async def test_judge_faithfulness_joins_context_list() -> None:
    judge = _StubJudge({"reasoning": "grounded", "score": 1.0, "verdict": "faithful"})

    result = await judge_faithfulness(
        judge,
        question="q",
        answer="a",
        contexts=["ctx1", "ctx2"],
    )

    assert result.score == 1.0
    assert "ctx1\n---\nctx2" in judge.messages[-1][1]["content"]


@pytest.mark.asyncio
async def test_judge_answer_correctness_defaults_missing_expected_answer() -> None:
    judge = _StubJudge({"reasoning": "partial", "score": 0.5, "verdict": "partial"})

    result = await judge_answer_correctness(
        judge,
        question="q",
        answer="a",
        expected_answer="",
    )

    assert result.score == 0.5
    assert "EXPECTED_ANSWER:\n(not provided)" in judge.messages[-1][1]["content"]


@pytest.mark.asyncio
async def test_judge_pairwise_preference_accepts_context_list() -> None:
    judge = _StubJudge({"reasoning": "A better", "choice": "A"})

    result = await judge_pairwise_preference(
        judge,
        question="q",
        contexts=["ctx1", "ctx2"],
        expected_answer="good",
        anti_answer="bad",
        expected_is_a=True,
    )

    assert result == PairwiseJudgment(
        expected_preferred=True,
        reasoning="A better",
        choice="A",
    )
    assert "ctx1\n---\nctx2" in judge.messages[-1][1]["content"]
