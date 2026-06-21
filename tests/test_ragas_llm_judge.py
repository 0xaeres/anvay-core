"""Unit tests for the RAGAS eval runner — LLM judge CoT parsing.

Covers the _llm_judge helper's ability to extract the `reasoning` field
(primary) or fall back to the legacy `notes` field, plus boundary cases.
No network calls needed.
"""

from __future__ import annotations

import pytest

from evals.run_ragas import _llm_judge

# --------------------------------------------------------------------------- #
# Minimal stub judge
# --------------------------------------------------------------------------- #


class _StubJudge:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def chat_json(self, messages, **kwargs):
        return self._payload, None


class _FailingJudge:
    async def chat_json(self, messages, **kwargs):
        raise RuntimeError("timeout")


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
