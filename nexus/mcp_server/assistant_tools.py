"""Assistant tools for the Nexus MCP server — the coding-agent channel.

See docs/ASSISTANT-LAYER.md §11. Same curate-don't-proxy discipline as the rest
of the Assistant layer: a small, intent-shaped tool set. A coding agent calls
these mid-task to query / act on Jira & Confluence; the raw Atlassian catalogue
stays behind the connector boundary.

Tools (5):
  assistant_ask              — run the agent loop, get an answer (+ any draft action)
  assistant_get_jira_issue   — direct live Jira lookup
  assistant_search_confluence — direct live Confluence search
  assistant_list_actions     — list drafted action proposals
  assistant_confirm_action   — execute a drafted proposal (the only write path)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nexus.assistant.capabilities import ToolContext, build_registry
from nexus.assistant.executor import execute_proposal
from nexus.assistant.factory import build_assistant_store, build_connector_port
from nexus.assistant.loop import AssistantLoop
from nexus.assistant.models import (
    Conversation,
    ConversationMessage,
    MessageRole,
    ProposalStatus,
)
from nexus.assistant.store import AssistantStore
from nexus.config import NexusConfig
from nexus.llm.client import ChatClient
from nexus.skills.store import SkillStore

log = logging.getLogger(__name__)


@dataclass
class AssistantToolState:
    """Lazily-initialised handles for the MCP server's assistant tools.

    Bound to one (`product`, `user`) at server launch — the `user` drives
    per-user OAuth attribution for any Jira/Confluence access.
    """

    product: str
    user: str
    config: NexusConfig | None = None
    _store: AssistantStore | None = None
    _connector: object | None = None
    _skill_store: SkillStore | None = None
    # Injectable for tests; in production, a fresh ChatClient per turn.
    llm_factory: Callable[[], object] | None = None

    @property
    def store(self) -> AssistantStore:
        if self._store is None:
            assert self.config is not None
            self._store = build_assistant_store(self.config)
        return self._store

    @property
    def connector(self):
        if self._connector is None:
            assert self.config is not None
            self._connector = build_connector_port(self.config, self.store)
        return self._connector

    @property
    def skill_store(self) -> SkillStore | None:
        if self._skill_store is None and self.config is not None:
            root = Path(self.config.hierarchy_root)
            if not root.is_absolute():
                root = Path.cwd() / root
            self._skill_store = SkillStore(root)
        return self._skill_store

    def make_llm(self):
        if self.llm_factory is not None:
            return self.llm_factory()
        assert self.config is not None
        cfg = self.config.models.assistant or self.config.models.council_agents
        return ChatClient.from_cfg(cfg, role="assistant")


async def assistant_ask(
    state: AssistantToolState, *, query: str, conversation_id: str | None = None
) -> dict:
    """Run one assistant turn — the agent may read corpus/Jira/Confluence and
    draft (never apply) Jira/Confluence changes."""
    store = state.store
    if conversation_id:
        conv = store.get_conversation(conversation_id)
        if conv is None:
            return {"error": f"unknown conversation {conversation_id!r}"}
    else:
        conv = store.create_conversation(
            Conversation(product_id=state.product, user_id=state.user, channel="mcp")
        )

    history = store.list_messages(conv.id)
    store.add_message(
        ConversationMessage(
            conversation_id=conv.id, role=MessageRole.USER, content=query
        )
    )

    llm = state.make_llm()
    try:
        ctx = ToolContext(
            product_id=state.product,
            user_id=state.user,
            conversation_id=conv.id,
            store=store,
            read_port=state.connector,
            planner=llm,
            retrieval=None,  # coding agents have the dedicated corpus tools already
            skill_store=state.skill_store,
        )
        loop = AssistantLoop(llm=llm, registry=build_registry())
        result = await loop.run_turn(ctx=ctx, history=history, user_text=query)
    finally:
        aclose = getattr(llm, "aclose", None)
        if aclose is not None:
            await aclose()

    store.add_message(
        ConversationMessage(
            conversation_id=conv.id, role=MessageRole.ASSISTANT, content=result.reply
        )
    )
    store.touch_conversation(conv.id, title=query[:60])

    return {
        "conversation_id": conv.id,
        "reply": result.reply,
        "tool_calls": [tc["tool"] for tc in result.tool_calls],
        "action_proposal": (
            result.action_proposal.model_dump() if result.action_proposal else None
        ),
    }


async def assistant_get_jira_issue(state: AssistantToolState, *, key: str) -> dict:
    return {"issue": await state.connector.get_jira_issue(key, as_user=state.user)}


async def assistant_search_confluence(
    state: AssistantToolState, *, query: str, space: str | None = None
) -> dict:
    return {
        "results": await state.connector.search_confluence(
            query, space=space, as_user=state.user
        )
    }


async def assistant_list_actions(
    state: AssistantToolState, *, status: str = "pending"
) -> dict:
    return {
        "actions": [
            p.model_dump()
            for p in state.store.list_proposals(
                product_id=state.product, status=status or None
            )
        ]
    }


async def assistant_confirm_action(
    state: AssistantToolState, *, proposal_id: str
) -> dict:
    """Execute a drafted action proposal. The ONLY MCP path that performs writes —
    the calling agent must invoke this deliberately after surfacing the preview."""
    store = state.store
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        return {"error": f"unknown action proposal {proposal_id!r}"}
    if proposal.status not in (ProposalStatus.PENDING, ProposalStatus.CONFIRMED):
        return {"error": f"proposal is {proposal.status}, not confirmable"}
    executed = await execute_proposal(
        proposal, act_port=state.connector, store=store, confirmed_by=state.user
    )
    return executed.model_dump()
