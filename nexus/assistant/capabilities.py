"""The curated Tier-2 tool facade — see docs/ASSISTANT-LAYER.md §4.

This is the ONLY tool list the agent's LLM ever sees (~8 intent-shaped tools).
The raw downstream MCP catalogue (Atlassian Rovo's 30-40 tools) stays behind the
connector boundary. This is "curate, don't proxy": it keeps tool-selection
accuracy high, token cost low, and the security surface small.

Broad write coverage (transitions, comments, assignments, page creation, …) is
expressed as steps inside an `ActionProposal.plan`, NOT as extra agent tools — so
this registry stays ~8 tools regardless of how much write coverage we add.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nexus.assistant.connector_port import ReadPort
from nexus.assistant.models import (
    JIRA_OPS,
    ActionProposal,
    ActionStep,
    ActionTarget,
)
from nexus.assistant.store import AssistantStore


@dataclass
class ToolContext:
    """Everything a tool handler needs. Built per assistant turn."""

    product_id: str
    user_id: str
    conversation_id: str
    store: AssistantStore
    read_port: ReadPort
    planner: Any | None = None  # ChatClient-like: async chat_json(messages)
    retrieval: Any | None = None  # RetrievalContext | None
    skill_store: Any | None = None  # SkillStore | None


ToolHandler = Callable[..., Awaitable[dict]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for args
    handler: ToolHandler
    group: str  # read | jira_action | confluence_action — for optional Tier-3 subsetting

    def spec(self) -> dict:
        """The form the agent's LLM sees in its system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# --------------------------------------------------------------------------
# READ handlers
# --------------------------------------------------------------------------


async def _search_corpus(ctx: ToolContext, *, query: str) -> dict:
    if ctx.retrieval is None:
        return {
            "unavailable": True,
            "note": "Corpus retrieval is not wired in this deployment.",
        }
    result = await ctx.retrieval.retrieve(query, product_id=ctx.product_id)
    hits = getattr(result, "hits", result)
    return {
        "query": query,
        "hits": [
            {
                "file": getattr(h, "file", None),
                "line": getattr(h, "line", None),
                "excerpt": (getattr(h, "content", "") or "")[:400],
                "score": getattr(h, "score", None),
            }
            for h in (hits or [])[:8]
        ],
    }


async def _find_skills(ctx: ToolContext, *, query: str) -> dict:
    if ctx.skill_store is None:
        return {"unavailable": True, "note": "Skill store is not wired."}
    q = query.lower()
    matches: list[dict] = []
    for skill in ctx.skill_store.iter_skills():
        hay = f"{skill.name} {getattr(skill, 'body', '')}".lower()
        if q in hay:
            matches.append(
                {
                    "name": skill.name,
                    "kind": str(getattr(skill, "kind", "")),
                    "confidence": getattr(skill, "confidence", None),
                }
            )
    return {"query": query, "skills": matches[:8]}


async def _get_jira_issue(ctx: ToolContext, *, key: str) -> dict:
    return {"issue": await ctx.read_port.get_jira_issue(key, as_user=ctx.user_id)}


async def _search_jira(ctx: ToolContext, *, query: str) -> dict:
    return {"results": await ctx.read_port.search_jira(query, as_user=ctx.user_id)}


async def _search_confluence(
    ctx: ToolContext, *, query: str, space: str | None = None
) -> dict:
    return {
        "results": await ctx.read_port.search_confluence(
            query, space=space, as_user=ctx.user_id
        )
    }


async def _get_confluence_page(ctx: ToolContext, *, page_id: str) -> dict:
    return {"page": await ctx.read_port.get_confluence_page(page_id, as_user=ctx.user_id)}


# --------------------------------------------------------------------------
# ACT handlers — propose only. They draft an ActionProposal and stop.
# Nothing is written until POST /assistant/actions/{id}/confirm.
# --------------------------------------------------------------------------

_JIRA_PLAN_SYSTEM = (
    "You are a Jira change planner. Given an issue and a user instruction, emit a "
    "JSON action plan — a list of typed mutations. Respond with ONLY a JSON object: "
    '{"plan": [{"op": "<op>", "args": {...}, "summary": "<one line>"}]}.\n'
    "Allowed ops and args:\n"
    '  create_subtask {"summary": str, "description"?: str}\n'
    '  transition     {"to": str}\n'
    '  add_comment    {"body": str}\n'
    '  assign         {"assignee": str}\n'
    '  update_field   {"field": str, "value": str}\n'
    "Keep the plan minimal and faithful to the instruction."
)

_CONFLUENCE_PLAN_SYSTEM = (
    "You are a Confluence change planner. Given a page and a user instruction, emit "
    "a JSON action plan. Respond with ONLY a JSON object: "
    '{"plan": [{"op": "<op>", "args": {...}, "summary": "<one line>"}]}.\n'
    "Allowed ops and args:\n"
    '  update_page {"body": str}                       (full new body)\n'
    '  create_page {"title": str, "body": str, "space"?: str}\n'
    "Keep the plan minimal and faithful to the instruction."
)


def _render_preview(plan: list[ActionStep]) -> str:
    if not plan:
        return "(empty plan — nothing to do)"
    lines = [f"{i + 1}. [{s.op}] {s.summary or _summarise(s)}" for i, s in enumerate(plan)]
    return "\n".join(lines)


def _summarise(step: ActionStep) -> str:
    a = step.args
    if step.op == "create_subtask":
        return f"Create subtask: {a.get('summary', '')}"
    if step.op == "transition":
        return f"Transition to: {a.get('to', '')}"
    if step.op == "add_comment":
        return f"Comment: {str(a.get('body', ''))[:80]}"
    if step.op == "assign":
        return f"Assign to: {a.get('assignee', '')}"
    if step.op == "update_field":
        return f"Set {a.get('field', '')} = {a.get('value', '')}"
    if step.op == "update_page":
        return "Update page body"
    if step.op == "create_page":
        return f"Create page: {a.get('title', '')}"
    return step.op


async def _plan_steps(ctx: ToolContext, system: str, user: str) -> list[ActionStep]:
    if ctx.planner is None:
        raise RuntimeError("no planner LLM configured for the assistant")
    parsed, _usage = await ctx.planner.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    raw = parsed.get("plan", []) if isinstance(parsed, dict) else []
    steps: list[ActionStep] = []
    for item in raw:
        if not isinstance(item, dict) or "op" not in item:
            continue
        steps.append(
            ActionStep(
                op=str(item["op"]),
                args=item.get("args", {}) or {},
                summary=str(item.get("summary", "")),
            )
        )
    return steps


async def _propose_jira_changes(
    ctx: ToolContext, *, issue_key: str, instruction: str
) -> dict:
    issue = await ctx.read_port.get_jira_issue(issue_key, as_user=ctx.user_id)
    steps = await _plan_steps(
        ctx,
        _JIRA_PLAN_SYSTEM,
        f"Issue:\n{json.dumps(issue, indent=2)}\n\nInstruction:\n{instruction}",
    )
    bad = [s.op for s in steps if s.op not in JIRA_OPS]
    if bad:
        return {"error": f"planner produced unsupported ops: {bad}"}
    proposal = ActionProposal(
        conversation_id=ctx.conversation_id,
        product_id=ctx.product_id,
        requested_by=ctx.user_id,
        target=ActionTarget(system="jira", key=issue_key),
        instruction=instruction,
        plan=steps,
        preview=_render_preview(steps),
    )
    ctx.store.save_proposal(proposal)
    return {
        "proposal_id": proposal.id,
        "status": "pending",
        "preview": proposal.preview,
        "note": "Drafted only. A human must confirm this before anything changes.",
    }


async def _propose_confluence_update(
    ctx: ToolContext, *, page_id: str, instruction: str
) -> dict:
    page = await ctx.read_port.get_confluence_page(page_id, as_user=ctx.user_id)
    steps = await _plan_steps(
        ctx,
        _CONFLUENCE_PLAN_SYSTEM,
        f"Page:\n{json.dumps(page, indent=2)}\n\nInstruction:\n{instruction}",
    )
    proposal = ActionProposal(
        conversation_id=ctx.conversation_id,
        product_id=ctx.product_id,
        requested_by=ctx.user_id,
        target=ActionTarget(system="confluence", key=page_id),
        instruction=instruction,
        plan=steps,
        preview=_render_preview(steps),
    )
    ctx.store.save_proposal(proposal)
    return {
        "proposal_id": proposal.id,
        "status": "pending",
        "preview": proposal.preview,
        "note": "Drafted only. A human must confirm this before anything changes.",
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_STR = {"type": "string"}


def build_registry() -> dict[str, Tool]:
    """The curated ~8-tool facade the agent LLM is given each turn."""
    tools = [
        Tool(
            name="search_corpus",
            description="Semantic search over the product's indexed code and docs. "
            "Use for broad 'how/where/why' questions.",
            parameters={"type": "object", "properties": {"query": _STR}, "required": ["query"]},
            handler=_search_corpus,
            group="read",
        ),
        Tool(
            name="find_skills",
            description="Look up curated Nexus skill files relevant to a topic.",
            parameters={"type": "object", "properties": {"query": _STR}, "required": ["query"]},
            handler=_find_skills,
            group="read",
        ),
        Tool(
            name="get_jira_issue",
            description="Fetch a single Jira issue by key (e.g. PROJ-123). Use for "
            "specific, current issue state.",
            parameters={"type": "object", "properties": {"key": _STR}, "required": ["key"]},
            handler=_get_jira_issue,
            group="read",
        ),
        Tool(
            name="search_jira",
            description="Search Jira issues by natural-language query.",
            parameters={"type": "object", "properties": {"query": _STR}, "required": ["query"]},
            handler=_search_jira,
            group="read",
        ),
        Tool(
            name="search_confluence",
            description="Search Confluence pages, optionally scoped to a space key.",
            parameters={
                "type": "object",
                "properties": {"query": _STR, "space": _STR},
                "required": ["query"],
            },
            handler=_search_confluence,
            group="read",
        ),
        Tool(
            name="get_confluence_page",
            description="Fetch a single Confluence page by id.",
            parameters={
                "type": "object",
                "properties": {"page_id": _STR},
                "required": ["page_id"],
            },
            handler=_get_confluence_page,
            group="read",
        ),
        Tool(
            name="propose_jira_changes",
            description="Draft a plan of Jira changes for an issue (subtasks, "
            "transitions, comments, assignments). Drafts ONLY — a human confirms "
            "separately. Never claim the change was applied.",
            parameters={
                "type": "object",
                "properties": {"issue_key": _STR, "instruction": _STR},
                "required": ["issue_key", "instruction"],
            },
            handler=_propose_jira_changes,
            group="jira_action",
        ),
        Tool(
            name="propose_confluence_update",
            description="Draft an update to a Confluence page. Drafts ONLY — a human "
            "confirms separately. Never claim the change was applied.",
            parameters={
                "type": "object",
                "properties": {"page_id": _STR, "instruction": _STR},
                "required": ["page_id", "instruction"],
            },
            handler=_propose_confluence_update,
            group="confluence_action",
        ),
    ]
    return {t.name: t for t in tools}
