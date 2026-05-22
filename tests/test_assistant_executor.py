"""execute_proposal — the confirm-and-apply path for ActionProposals."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nexus.assistant.connector_port import FakeConnectorPort
from nexus.assistant.executor import execute_proposal
from nexus.assistant.models import (
    ActionProposal,
    ActionStep,
    ActionTarget,
    Conversation,
    ProposalStatus,
)
from nexus.assistant.store import AssistantStore


class _BoomPort:
    """An ActPort whose first write fails — exercises the failure path."""

    async def execute_step(self, step, *, target, as_user):
        raise RuntimeError("connector exploded")


def _proposal(store: AssistantStore) -> ActionProposal:
    conv = store.create_conversation(Conversation(product_id="my-api", user_id="alice"))
    proposal = ActionProposal(
        conversation_id=conv.id,
        product_id="my-api",
        requested_by="alice",
        target=ActionTarget(system="jira", key="PROJ-1"),
        instruction="split it",
        plan=[
            ActionStep(op="create_subtask", args={"summary": "A"}, summary="A"),
            ActionStep(op="transition", args={"to": "In Progress"}, summary="move"),
        ],
    )
    store.save_proposal(proposal)
    return proposal


def test_execute_proposal_happy_path(tmp_path: Path) -> None:
    store = AssistantStore(tmp_path / "assistant.db")
    proposal = _proposal(store)

    executed = asyncio.run(
        execute_proposal(
            proposal,
            act_port=FakeConnectorPort(),
            store=store,
            confirmed_by="bob",
        )
    )

    assert executed.status is ProposalStatus.EXECUTED
    assert executed.confirmed_by == "bob"
    assert executed.executed_at is not None
    assert executed.result is not None
    assert len(executed.result["steps"]) == 2
    # Persisted, too.
    assert store.get_proposal(proposal.id).status is ProposalStatus.EXECUTED  # type: ignore[union-attr]


def test_execute_proposal_records_failure(tmp_path: Path) -> None:
    store = AssistantStore(tmp_path / "assistant.db")
    proposal = _proposal(store)

    executed = asyncio.run(
        execute_proposal(
            proposal, act_port=_BoomPort(), store=store, confirmed_by="bob"
        )
    )

    assert executed.status is ProposalStatus.FAILED
    assert executed.error is not None
    assert "exploded" in executed.error


def test_executed_proposal_cannot_be_re_executed(tmp_path: Path) -> None:
    store = AssistantStore(tmp_path / "assistant.db")
    proposal = _proposal(store)
    asyncio.run(
        execute_proposal(
            proposal, act_port=FakeConnectorPort(), store=store, confirmed_by="bob"
        )
    )
    refreshed = store.get_proposal(proposal.id)
    assert refreshed is not None

    try:
        asyncio.run(
            execute_proposal(
                refreshed, act_port=FakeConnectorPort(), store=store, confirmed_by="bob"
            )
        )
        raise AssertionError("expected a ValueError for re-executing")
    except ValueError:
        pass
