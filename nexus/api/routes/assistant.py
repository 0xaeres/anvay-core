"""Assistant layer — conversational + action API. See docs/ASSISTANT-LAYER.md §10.

`POST /assistant/messages` streams the turn as Server-Sent Events: live
`tool_call` / `tool_result` progress, then a terminal `session_end` carrying the
reply and any drafted action proposal. The agent loop runs in a background task;
its `on_event` callback feeds an `asyncio.Queue` that the SSE generator drains.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Body, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from nexus.api.deps import (
    get_assistant_store,
    get_config_dep,
    get_connector_port,
    get_skill_store,
)
from nexus.assistant.capabilities import ToolContext, build_registry
from nexus.assistant.executor import execute_proposal
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
router = APIRouter(tags=["assistant"])

# Keeps streamed turns from being garbage-collected if the client disconnects
# before the background loop finishes (it still runs to completion to persist).
_background: set[asyncio.Task] = set()


def _assistant_llm(config: NexusConfig) -> ChatClient:
    cfg = config.models.assistant or config.models.council_agents
    return ChatClient.from_cfg(cfg, role="assistant")


# ---------------------------------------------------------------- conversations


@router.post("/products/{product_id}/assistant/messages")
async def post_message(
    product_id: str,
    text: str = Body(..., embed=True),
    conversation_id: str | None = Body(None, embed=True),
    actor: str = Body("admin", embed=True),
    store: AssistantStore = Depends(get_assistant_store),
    config: NexusConfig = Depends(get_config_dep),
    skill_store: SkillStore = Depends(get_skill_store),
    connector=Depends(get_connector_port),
) -> EventSourceResponse:
    """Start or continue an assistant conversation; streams one agent turn as SSE."""
    if conversation_id:
        conv = store.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation not found")
        if conv.product_id != product_id:
            raise HTTPException(status_code=403, detail="conversation belongs to another product")
    else:
        conv = store.create_conversation(
            Conversation(product_id=product_id, user_id=actor, channel="ui")
        )

    history = store.list_messages(conv.id)
    store.add_message(
        ConversationMessage(conversation_id=conv.id, role=MessageRole.USER, content=text)
    )

    async def event_stream() -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(ev: dict) -> None:
            await queue.put(("event", ev))

        async def run() -> None:
            llm = _assistant_llm(config)
            try:
                ctx = ToolContext(
                    product_id=product_id,
                    user_id=actor,
                    conversation_id=conv.id,
                    store=store,
                    read_port=connector,
                    planner=llm,
                    retrieval=None,
                    skill_store=skill_store,
                )
                loop = AssistantLoop(llm=llm, registry=build_registry())
                result = await loop.run_turn(
                    ctx=ctx, history=history, user_text=text, on_event=on_event
                )
                store.add_message(
                    ConversationMessage(
                        conversation_id=conv.id,
                        role=MessageRole.ASSISTANT,
                        content=result.reply,
                    )
                )
                store.touch_conversation(conv.id, title=text[:60])
                await queue.put(("final", result))
            except Exception as e:
                log.exception("assistant turn failed")
                await queue.put(("error", str(e)))
            finally:
                await llm.aclose()
                await queue.put((None, None))

        task = asyncio.create_task(run())
        _background.add(task)
        task.add_done_callback(_background.discard)

        yield {"event": "start", "data": json.dumps({"conversation_id": conv.id})}
        while True:
            kind, payload = await queue.get()
            if kind is None:
                break
            if kind == "event":
                yield {"event": "message", "data": json.dumps(payload)}
            elif kind == "error":
                yield {"event": "error", "data": json.dumps({"message": payload})}
            elif kind == "final":
                yield {
                    "event": "session_end",
                    "data": json.dumps(
                        {
                            "conversation_id": conv.id,
                            "reply": payload.reply,
                            "iterations": payload.iterations,
                            "action_proposal": (
                                payload.action_proposal.model_dump()
                                if payload.action_proposal
                                else None
                            ),
                        }
                    ),
                }

    return EventSourceResponse(event_stream())


@router.get("/products/{product_id}/assistant/conversations")
async def list_conversations(
    product_id: str, store: AssistantStore = Depends(get_assistant_store)
) -> dict:
    return {
        "conversations": [
            c.model_dump() for c in store.list_conversations(product_id=product_id)
        ]
    }


@router.get("/assistant/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str, store: AssistantStore = Depends(get_assistant_store)
) -> dict:
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {
        "conversation": conv.model_dump(),
        "messages": [m.model_dump() for m in store.list_messages(conversation_id)],
    }


# ---------------------------------------------------------------- action proposals


@router.get("/products/{product_id}/assistant/actions")
async def list_actions(
    product_id: str,
    status: str | None = None,
    store: AssistantStore = Depends(get_assistant_store),
) -> dict:
    return {
        "actions": [
            p.model_dump()
            for p in store.list_proposals(product_id=product_id, status=status)
        ]
    }


@router.get("/assistant/actions/{action_id}")
async def get_action(
    action_id: str, store: AssistantStore = Depends(get_assistant_store)
) -> dict:
    proposal = store.get_proposal(action_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="action proposal not found")
    return proposal.model_dump()


@router.post("/assistant/actions/{action_id}/confirm")
async def confirm_action(
    action_id: str,
    actor: str = Body("admin", embed=True),
    store: AssistantStore = Depends(get_assistant_store),
    connector=Depends(get_connector_port),
) -> dict:
    """Execute a drafted action proposal. The ONLY path that performs writes."""
    proposal = store.get_proposal(action_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="action proposal not found")
    if proposal.status not in (ProposalStatus.PENDING, ProposalStatus.CONFIRMED):
        raise HTTPException(
            status_code=409, detail=f"proposal is {proposal.status}, not confirmable"
        )
    executed = await execute_proposal(
        proposal, act_port=connector, store=store, confirmed_by=actor
    )
    return executed.model_dump()


@router.post("/assistant/actions/{action_id}/reject")
async def reject_action(
    action_id: str,
    actor: str = Body("admin", embed=True),
    store: AssistantStore = Depends(get_assistant_store),
) -> dict:
    proposal = store.get_proposal(action_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="action proposal not found")
    if proposal.status != ProposalStatus.PENDING:
        raise HTTPException(
            status_code=409, detail=f"proposal is {proposal.status}, not rejectable"
        )
    store.update_proposal(action_id, status=ProposalStatus.REJECTED, confirmed_by=actor)
    return store.get_proposal(action_id).model_dump()  # type: ignore[union-attr]


# ---------------------------------------------------------------- identity


@router.get("/products/{product_id}/assistant/identity")
async def get_identity_status(
    product_id: str,
    actor: str = "admin",
    store: AssistantStore = Depends(get_assistant_store),
    config: NexusConfig = Depends(get_config_dep),
) -> dict:
    """Whether `actor` has connected an Atlassian account for this product."""
    if not config.atlassian.enabled:
        return {
            "connected": False,
            "available": False,
            "reason": "Atlassian integration is disabled in nexus.yaml.",
        }
    if not store.identities_enabled:
        return {
            "connected": False,
            "available": False,
            "reason": "token encryption is not configured (set NEXUS_TOKEN_KEY).",
        }
    identity = store.get_identity(actor, provider="atlassian")
    if identity is None:
        return {"connected": False, "available": True}
    return {
        "connected": True,
        "available": True,
        "scope": identity.scope,
        "expires_at": identity.expires_at,
        "expired": identity.is_expired(),
    }
