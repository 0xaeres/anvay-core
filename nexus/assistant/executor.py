"""Executes a *confirmed* ActionProposal against the connector ActPort.

This is the only path that performs writes. It runs after a human has confirmed
the proposal via `POST /assistant/actions/{id}/confirm` — honouring Invariant 3.
Each step is attributed to the confirming user (per-user OAuth, §6).
"""

from __future__ import annotations

import logging

from nexus.assistant.connector_port import ActPort
from nexus.assistant.models import ActionProposal, ProposalStatus
from nexus.assistant.store import AssistantStore

log = logging.getLogger(__name__)


async def execute_proposal(
    proposal: ActionProposal,
    *,
    act_port: ActPort,
    store: AssistantStore,
    confirmed_by: str,
) -> ActionProposal:
    """Run every step of a confirmed proposal. Idempotency is the connector's job."""
    if proposal.status not in (ProposalStatus.PENDING, ProposalStatus.CONFIRMED):
        raise ValueError(
            f"proposal {proposal.id} is {proposal.status}, not confirmable"
        )

    store.update_proposal(
        proposal.id, status=ProposalStatus.CONFIRMED, confirmed_by=confirmed_by
    )

    step_results: list[dict] = []
    try:
        for step in proposal.plan:
            res = await act_port.execute_step(
                step, target=proposal.target, as_user=confirmed_by
            )
            step_results.append({"op": step.op, "summary": step.summary, **res})
    except Exception as e:
        log.warning("action proposal %s failed mid-execution: %s", proposal.id, e)
        store.update_proposal(
            proposal.id,
            status=ProposalStatus.FAILED,
            result={"steps": step_results},
            error=str(e),
        )
        return store.get_proposal(proposal.id)  # type: ignore[return-value]

    store.update_proposal(
        proposal.id,
        status=ProposalStatus.EXECUTED,
        result={"steps": step_results},
    )
    return store.get_proposal(proposal.id)  # type: ignore[return-value]
