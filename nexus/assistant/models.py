"""Data models for the Assistant layer — see docs/ASSISTANT-LAYER.md §8-9.

`ActionProposal` is a deliberate sibling of `SkillProposal`: a draft that does
nothing until a human confirms it. This is how the Assistant honours Invariant 3
("humans approve, agents draft").
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ActionSystem(StrEnum):
    JIRA = "jira"
    CONFLUENCE = "confluence"


class ProposalStatus(StrEnum):
    PENDING = "pending"        # drafted, awaiting human confirmation
    CONFIRMED = "confirmed"    # human said yes, execution in flight
    REJECTED = "rejected"      # human said no
    EXECUTED = "executed"      # write(s) applied successfully
    FAILED = "failed"          # execution attempted but errored


# Typed mutation ops the connector executor understands (docs/ASSISTANT-LAYER.md §4).
# Broader write coverage = more ops here, with NO growth in the agent's tool list.
JIRA_OPS = frozenset(
    {"create_subtask", "transition", "add_comment", "assign", "update_field"}
)
CONFLUENCE_OPS = frozenset({"update_page", "create_page"})
ALL_OPS = JIRA_OPS | CONFLUENCE_OPS


class ActionStep(BaseModel):
    """One typed mutation inside an ActionProposal.plan."""

    op: str
    args: dict = Field(default_factory=dict)
    summary: str = ""  # human-readable one-liner for the preview


class ActionTarget(BaseModel):
    system: ActionSystem
    key: str  # Jira issue key (PROJ-123) or Confluence page id


class ActionProposal(BaseModel):
    """A drafted change to Jira/Confluence. Inert until `confirm`ed by a human."""

    id: str = Field(default_factory=lambda: _gen_id("act"))
    conversation_id: str
    product_id: str
    requested_by: str
    target: ActionTarget
    instruction: str
    plan: list[ActionStep] = Field(default_factory=list)
    preview: str = ""
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: str = Field(default_factory=_now)
    confirmed_by: str | None = None
    executed_at: str | None = None
    result: dict | None = None
    error: str | None = None


class ConversationMessage(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("msg"))
    conversation_id: str
    role: MessageRole
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    created_at: str = Field(default_factory=_now)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("conv"))
    product_id: str
    user_id: str
    channel: str = "ui"  # ui | mcp | teams
    title: str = ""
    created_at: str = Field(default_factory=_now)
    last_active_at: str = Field(default_factory=_now)
