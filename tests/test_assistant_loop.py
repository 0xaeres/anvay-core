"""AssistantLoop — the agent loop, driven by a scripted fake LLM."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nexus.assistant.capabilities import ToolContext, build_registry
from nexus.assistant.connector_port import FakeConnectorPort
from nexus.assistant.loop import AssistantLoop
from nexus.assistant.models import Conversation, ProposalStatus
from nexus.assistant.store import AssistantStore
from nexus.llm.client import TokenUsage


class FakeLLM:
    """Returns scripted JSON action objects. Used as both loop LLM and planner."""

    def __init__(self, scripted: list[dict]):
        self.scripted = list(scripted)
        self.call_count = 0

    async def chat_json(self, messages, **kw):
        self.call_count += 1
        if not self.scripted:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.scripted.pop(0), TokenUsage(prompt=10, completion=5)


def _ctx(tmp_path: Path, llm: FakeLLM) -> tuple[ToolContext, AssistantStore]:
    store = AssistantStore(tmp_path / "assistant.db")
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    ctx = ToolContext(
        product_id="my-api",
        user_id="alice",
        conversation_id=conv.id,
        store=store,
        read_port=FakeConnectorPort(),
        planner=llm,
        retrieval=None,
        skill_store=None,
    )
    return ctx, store


def test_read_tool_then_final_answer(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"action": "call_tool", "tool": "get_jira_issue", "args": {"key": "PROJ-2"}},
            {"action": "final", "answer": "PROJ-2 is In Progress."},
        ]
    )
    ctx, _store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry())

    result = asyncio.run(loop.run_turn(ctx=ctx, history=[], user_text="status of PROJ-2?"))

    assert result.reply == "PROJ-2 is In Progress."
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "get_jira_issue"
    assert result.action_proposal is None


def test_propose_jira_changes_creates_pending_proposal(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            # 1. loop decides to draft Jira changes
            {
                "action": "call_tool",
                "tool": "propose_jira_changes",
                "args": {"issue_key": "PROJ-1", "instruction": "split into 2 subtasks"},
            },
            # 2. the planner LLM call inside the tool returns a plan
            {
                "plan": [
                    {"op": "create_subtask", "args": {"summary": "Part A"}, "summary": "A"},
                    {"op": "create_subtask", "args": {"summary": "Part B"}, "summary": "B"},
                ]
            },
            # 3. loop emits the final answer
            {"action": "final", "answer": "Drafted 2 subtasks for PROJ-1 — confirm to apply."},
        ]
    )
    ctx, store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry())

    result = asyncio.run(
        loop.run_turn(ctx=ctx, history=[], user_text="break PROJ-1 into subtasks")
    )

    assert result.action_proposal is not None
    assert result.action_proposal.status is ProposalStatus.PENDING
    assert len(result.action_proposal.plan) == 2
    # The proposal was persisted and is pending — nothing was executed.
    pending = store.list_proposals(product_id="my-api", status="pending")
    assert len(pending) == 1
    assert pending[0].id == result.action_proposal.id


def test_unknown_tool_is_surfaced_not_fatal(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"action": "call_tool", "tool": "delete_everything", "args": {}},
            {"action": "final", "answer": "I can't do that."},
        ]
    )
    ctx, _store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry())

    result = asyncio.run(loop.run_turn(ctx=ctx, history=[], user_text="drop the db"))

    assert result.reply == "I can't do that."
    assert "error" in result.tool_calls[0]["result"]


def test_run_turn_emits_progress_events(tmp_path: Path) -> None:
    events: list[dict] = []

    async def on_event(ev: dict) -> None:
        events.append(ev)

    llm = FakeLLM(
        [
            {"action": "call_tool", "tool": "get_jira_issue", "args": {"key": "P-1"}},
            {"action": "final", "answer": "done"},
        ]
    )
    ctx, _store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry())

    asyncio.run(loop.run_turn(ctx=ctx, history=[], user_text="x", on_event=on_event))

    types = [e["type"] for e in events]
    assert types == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "get_jira_issue"
    assert events[1]["ok"] is True


def test_run_turn_without_on_event_is_unaffected(tmp_path: Path) -> None:
    # The sync / MCP callers pass no sink — the loop must behave exactly as before.
    llm = FakeLLM([{"action": "final", "answer": "hi"}])
    ctx, _store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry())
    result = asyncio.run(loop.run_turn(ctx=ctx, history=[], user_text="hello"))
    assert result.reply == "hi"


def test_iteration_cap_is_enforced(tmp_path: Path) -> None:
    # Always calls a tool, never finalises — must stop at the cap.
    llm = FakeLLM(
        [{"action": "call_tool", "tool": "get_jira_issue", "args": {"key": "X-1"}}] * 10
    )
    ctx, _store = _ctx(tmp_path, llm)
    loop = AssistantLoop(llm=llm, registry=build_registry(), max_iterations=3)

    result = asyncio.run(loop.run_turn(ctx=ctx, history=[], user_text="loop forever"))

    assert result.iterations == 3
    assert "budget" in result.reply
