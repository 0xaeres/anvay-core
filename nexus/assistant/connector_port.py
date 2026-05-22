"""Connector ports — the boundary between the assistant brain and external systems.

The brain (loop.py, capabilities.py) only ever talks to these Protocols. The real
implementation against the Atlassian remote MCP server (`AtlassianConnectorPort`)
is a later increment — until it lands, `FakeConnectorPort` keeps the whole
Assistant layer runnable, testable, and demoable.

See docs/ASSISTANT-LAYER.md §5-6.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexus.assistant.models import ActionStep, ActionTarget


class ConnectorUnavailable(RuntimeError):
    """Raised when a connector cannot serve a read/act request."""


@runtime_checkable
class ReadPort(Protocol):
    """Live read access to a source system, on behalf of a specific user."""

    async def get_jira_issue(self, key: str, *, as_user: str) -> dict: ...

    async def search_jira(self, query: str, *, as_user: str) -> list[dict]: ...

    async def search_confluence(
        self, query: str, *, space: str | None, as_user: str
    ) -> list[dict]: ...

    async def get_confluence_page(self, page_id: str, *, as_user: str) -> dict: ...


@runtime_checkable
class ActPort(Protocol):
    """Write access — executes one typed mutation, attributed to a user.

    `target` is the ActionProposal's target (e.g. the parent Jira issue), which a
    step like `create_subtask` needs but does not itself carry.
    """

    async def execute_step(
        self, step: ActionStep, *, target: ActionTarget, as_user: str
    ) -> dict: ...


class FakeConnectorPort:
    """Deterministic stand-in for the Atlassian connector.

    Returns plausible-looking data so the agent loop, the API, and (later) the UI
    can be built and demoed before the real Atlassian remote-MCP transport lands.
    Swap for `AtlassianConnectorPort` in `nexus/api/deps.py` when that arrives.
    """

    async def get_jira_issue(self, key: str, *, as_user: str) -> dict:
        return {
            "key": key,
            "summary": f"[stub] Issue {key}",
            "status": "In Progress",
            "type": "Story",
            "assignee": as_user,
            "description": (
                f"[stub] Description for {key}. Replace FakeConnectorPort with "
                "AtlassianConnectorPort for live data."
            ),
            "subtasks": [],
            "_stub": True,
        }

    async def search_jira(self, query: str, *, as_user: str) -> list[dict]:
        return [
            {"key": "STUB-1", "summary": f"[stub] result for {query!r}", "status": "To Do"},
            {"key": "STUB-2", "summary": f"[stub] result for {query!r}", "status": "Done"},
        ]

    async def search_confluence(
        self, query: str, *, space: str | None, as_user: str
    ) -> list[dict]:
        return [
            {
                "id": "stub-page-1",
                "title": f"[stub] page about {query!r}",
                "space": space or "ENG",
                "excerpt": "[stub] excerpt — wire AtlassianConnectorPort for live search.",
            }
        ]

    async def get_confluence_page(self, page_id: str, *, as_user: str) -> dict:
        return {
            "id": page_id,
            "title": f"[stub] Page {page_id}",
            "space": "ENG",
            "body": f"[stub] body of page {page_id}.",
            "version": 1,
            "_stub": True,
        }

    async def execute_step(
        self, step: ActionStep, *, target: ActionTarget, as_user: str
    ) -> dict:
        return {
            "op": step.op,
            "applied": True,
            "as_user": as_user,
            "target": target.key,
            "_stub": True,
            "detail": f"[stub] {step.op} on {target.key} would run against the real connector.",
        }
