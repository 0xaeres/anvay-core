"""AssistantStore — conversation, message, and action-proposal persistence."""

from __future__ import annotations

from pathlib import Path

from nexus.assistant.models import (
    ActionProposal,
    ActionStep,
    ActionTarget,
    Conversation,
    ConversationMessage,
    MessageRole,
    ProposalStatus,
)
from nexus.assistant.store import AssistantStore


def _store(tmp_path: Path) -> AssistantStore:
    return AssistantStore(tmp_path / "assistant.db")


def test_conversation_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conv = store.create_conversation(
        Conversation(product_id="my-api", user_id="alice", channel="ui")
    )
    got = store.get_conversation(conv.id)
    assert got is not None
    assert got.product_id == "my-api"
    assert got.user_id == "alice"
    assert store.list_conversations(product_id="my-api")[0].id == conv.id
    assert store.list_conversations(product_id="other") == []


def test_messages_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    store.add_message(
        ConversationMessage(
            conversation_id=conv.id, role=MessageRole.USER, content="hello"
        )
    )
    store.add_message(
        ConversationMessage(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="hi there",
        )
    )
    msgs = store.list_messages(conv.id)
    assert [m.role for m in msgs] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert msgs[0].content == "hello"


def test_touch_conversation_sets_title_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    store.touch_conversation(conv.id, title="First question")
    store.touch_conversation(conv.id, title="Second question")
    assert store.get_conversation(conv.id).title == "First question"  # type: ignore[union-attr]


def test_action_proposal_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    proposal = ActionProposal(
        conversation_id=conv.id,
        product_id="my-api",
        requested_by="alice",
        target=ActionTarget(system="jira", key="PROJ-1"),
        instruction="split into subtasks",
        plan=[
            ActionStep(op="create_subtask", args={"summary": "A"}, summary="Create A"),
            ActionStep(op="transition", args={"to": "In Progress"}, summary="Move it"),
        ],
        preview="1. [create_subtask] Create A\n2. [transition] Move it",
    )
    store.save_proposal(proposal)

    got = store.get_proposal(proposal.id)
    assert got is not None
    assert got.status is ProposalStatus.PENDING
    assert len(got.plan) == 2
    assert got.plan[0].op == "create_subtask"
    assert got.target.system == "jira"


def test_proposal_status_filter_and_update(tmp_path: Path) -> None:
    store = _store(tmp_path)
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    proposal = ActionProposal(
        conversation_id=conv.id,
        product_id="my-api",
        requested_by="alice",
        target=ActionTarget(system="confluence", key="page-1"),
        instruction="fix typo",
    )
    store.save_proposal(proposal)

    assert len(store.list_proposals(product_id="my-api", status="pending")) == 1

    ok = store.update_proposal(
        proposal.id,
        status=ProposalStatus.EXECUTED,
        confirmed_by="bob",
        result={"steps": []},
    )
    assert ok
    updated = store.get_proposal(proposal.id)
    assert updated is not None
    assert updated.status is ProposalStatus.EXECUTED
    assert updated.confirmed_by == "bob"
    assert updated.executed_at is not None
    assert store.list_proposals(product_id="my-api", status="pending") == []
