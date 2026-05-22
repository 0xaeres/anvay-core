"""MCP server assistant tools — the coding-agent channel (Increment 3)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nexus.assistant.connector_port import FakeConnectorPort
from nexus.assistant.models import (
    ActionProposal,
    ActionStep,
    ActionTarget,
    Conversation,
)
from nexus.assistant.store import AssistantStore
from nexus.llm.client import TokenUsage
from nexus.mcp_server.assistant_tools import (
    AssistantToolState,
    assistant_ask,
    assistant_confirm_action,
    assistant_get_jira_issue,
    assistant_list_actions,
    assistant_search_confluence,
)


class FakeLLM:
    """Scripted JSON-action LLM — doubles as the loop LLM and the planner."""

    def __init__(self, scripted: list[dict]):
        self.scripted = list(scripted)

    async def chat_json(self, messages, **kw):
        if not self.scripted:
            raise AssertionError("FakeLLM exhausted")
        return self.scripted.pop(0), TokenUsage(prompt=5, completion=5)

    async def aclose(self) -> None:
        pass


def _state(tmp_path: Path, *, scripted: list[dict] | None = None) -> AssistantToolState:
    store = AssistantStore(tmp_path / "assistant.db")
    return AssistantToolState(
        product="my-api",
        user="alice",
        _store=store,
        _connector=FakeConnectorPort(),
        _skill_store=None,
        llm_factory=(lambda: FakeLLM(scripted or [])) if scripted is not None else None,
    )


def test_get_jira_issue(tmp_path: Path) -> None:
    state = _state(tmp_path)
    out = asyncio.run(assistant_get_jira_issue(state, key="PROJ-7"))
    assert out["issue"]["key"] == "PROJ-7"


def test_search_confluence(tmp_path: Path) -> None:
    state = _state(tmp_path)
    out = asyncio.run(assistant_search_confluence(state, query="oncall", space="ENG"))
    assert out["results"]
    assert out["results"][0]["space"] == "ENG"


def test_ask_runs_loop_and_creates_conversation(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        scripted=[
            {"action": "call_tool", "tool": "get_jira_issue", "args": {"key": "PROJ-1"}},
            {"action": "final", "answer": "PROJ-1 is In Progress."},
        ],
    )
    out = asyncio.run(assistant_ask(state, query="status of PROJ-1?"))
    assert out["reply"] == "PROJ-1 is In Progress."
    assert out["conversation_id"]
    assert out["tool_calls"] == ["get_jira_issue"]
    # The conversation + messages were persisted.
    msgs = state.store.list_messages(out["conversation_id"])
    assert [m.role.value for m in msgs] == ["user", "assistant"]


def test_ask_surfaces_a_drafted_action_proposal(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        scripted=[
            {
                "action": "call_tool",
                "tool": "propose_jira_changes",
                "args": {"issue_key": "PROJ-1", "instruction": "add a subtask"},
            },
            {"plan": [{"op": "create_subtask", "args": {"summary": "X"}, "summary": "X"}]},
            {"action": "final", "answer": "Drafted 1 subtask — confirm to apply."},
        ],
    )
    out = asyncio.run(assistant_ask(state, query="add a subtask to PROJ-1"))
    assert out["action_proposal"] is not None
    assert out["action_proposal"]["status"] == "pending"


def test_list_and_confirm_action(tmp_path: Path) -> None:
    state = _state(tmp_path)
    conv = state.store.create_conversation(
        Conversation(product_id="my-api", user_id="alice")
    )
    proposal = ActionProposal(
        conversation_id=conv.id,
        product_id="my-api",
        requested_by="alice",
        target=ActionTarget(system="jira", key="PROJ-1"),
        instruction="split it",
        plan=[ActionStep(op="create_subtask", args={"summary": "A"}, summary="A")],
    )
    state.store.save_proposal(proposal)

    listed = asyncio.run(assistant_list_actions(state, status="pending"))
    assert len(listed["actions"]) == 1

    confirmed = asyncio.run(assistant_confirm_action(state, proposal_id=proposal.id))
    assert confirmed["status"] == "executed"
    # no longer pending
    assert asyncio.run(assistant_list_actions(state, status="pending"))["actions"] == []


def test_confirm_unknown_action_returns_error(tmp_path: Path) -> None:
    state = _state(tmp_path)
    out = asyncio.run(assistant_confirm_action(state, proposal_id="act_does_not_exist"))
    assert "error" in out


def test_ask_with_unknown_conversation_returns_error(tmp_path: Path) -> None:
    state = _state(tmp_path, scripted=[])
    out = asyncio.run(
        assistant_ask(state, query="hi", conversation_id="conv_missing")
    )
    assert "error" in out
