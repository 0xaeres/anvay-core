"""AtlassianConnectorPort — the real ReadPort + ActPort backed by the Atlassian
Rovo MCP Server. See docs/ASSISTANT-LAYER.md §4-6.

This is the "connector layer" of the design. It does NOT expose Atlassian's raw
tool catalogue upward — the assistant's agent only ever sees Nexus's curated
~8-tool facade (capabilities.py). This module is the single place that maps a
curated capability / action step onto a concrete Atlassian tool call.

Per-user OAuth: every call resolves the *user's* token (refreshing silently when
expired), so writes are attributed to the real person and respect their Jira /
Confluence permissions.
"""

from __future__ import annotations

import logging

import httpx

from nexus.assistant.connector_port import ConnectorUnavailable
from nexus.assistant.models import ActionStep, ActionTarget
from nexus.assistant.store import AssistantStore
from nexus.auth.atlassian_oauth import AtlassianOAuth
from nexus.connectors.remote_mcp import (
    RemoteMCPClient,
    RemoteMCPError,
    parse_tool_result,
)

log = logging.getLogger(__name__)

# Curated capability / action op  →  concrete Atlassian Rovo MCP tool name.
# Verified against the Atlassian Rovo MCP Server "Supported tools" docs (May 2026).
# Argument shapes below are best-effort and should be re-checked against a live
# `tools/list` when first wired to a real Atlassian site.
_JIRA_GET = "getJiraIssue"
_JIRA_SEARCH = "searchJiraIssuesUsingJql"
_JIRA_CREATE = "createJiraIssue"
_JIRA_EDIT = "editJiraIssue"
_JIRA_TRANSITION = "transitionJiraIssue"
_JIRA_COMMENT = "addCommentToJiraIssue"
_CONF_GET = "getConfluencePage"
_CONF_SEARCH = "searchConfluenceUsingCql"
_CONF_CREATE = "createConfluencePage"
_CONF_UPDATE = "updateConfluencePage"


def _cql_for(query: str, space: str | None) -> str:
    safe = query.replace('"', '\\"')
    cql = f'text ~ "{safe}"'
    if space:
        cql += f' and space = "{space}"'
    return cql


class AtlassianConnectorPort:
    """Implements both ReadPort and ActPort against the Atlassian Rovo MCP Server."""

    def __init__(
        self,
        *,
        cfg,  # AtlassianCfg
        store: AssistantStore,
        oauth: AtlassianOAuth,
        client_factory=None,  # callable(token_provider) -> RemoteMCPClient — injectable for tests
    ):
        self.cfg = cfg
        self.store = store
        self.oauth = oauth
        self._client_factory = client_factory or self._default_client_factory
        self._http: httpx.AsyncClient | None = None
        self._clients: dict[str, RemoteMCPClient] = {}

    def _default_client_factory(self, token_provider) -> RemoteMCPClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0)
        return RemoteMCPClient(
            endpoint=self.cfg.mcp_url,
            token_provider=token_provider,
            http_client=self._http,
        )

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        if self._http is not None:
            await self._http.aclose()

    # ------------------------------------------------------------ token plumbing

    def _token_provider(self, user_id: str):
        """An async callable that yields a fresh access token for `user_id`,
        refreshing silently when the stored token has expired."""

        async def provide() -> str:
            identity = self.store.get_identity(user_id, provider="atlassian")
            if identity is None:
                raise ConnectorUnavailable(
                    f"{user_id} has not connected an Atlassian account — "
                    "visit Settings → Connect Atlassian."
                )
            if identity.is_expired():
                if not identity.refresh_token:
                    raise ConnectorUnavailable(
                        f"{user_id}'s Atlassian session expired — please reconnect."
                    )
                identity = await self.oauth.refresh(refresh_token=identity.refresh_token)
                self.store.save_identity(user_id, provider="atlassian", token=identity)
            return identity.access_token

        return provide

    async def _client(self, user_id: str) -> RemoteMCPClient:
        client = self._clients.get(user_id)
        if client is None:
            client = self._client_factory(self._token_provider(user_id))
            self._clients[user_id] = client
        return client

    async def _call(self, user_id: str, tool: str, args: dict) -> dict:
        client = await self._client(user_id)
        try:
            return await client.call_tool(tool, args)
        except RemoteMCPError as e:
            raise ConnectorUnavailable(f"Atlassian call {tool} failed: {e}") from e

    # ------------------------------------------------------------ ReadPort

    async def get_jira_issue(self, key: str, *, as_user: str) -> dict:
        result = await self._call(as_user, _JIRA_GET, {"issueIdOrKey": key})
        return parse_tool_result(result)

    async def search_jira(self, query: str, *, as_user: str) -> list[dict]:
        result = await self._call(as_user, _JIRA_SEARCH, {"jql": query})
        parsed = parse_tool_result(result)
        return parsed.get("issues") or parsed.get("items") or [parsed]

    async def search_confluence(
        self, query: str, *, space: str | None, as_user: str
    ) -> list[dict]:
        result = await self._call(
            as_user, _CONF_SEARCH, {"cql": _cql_for(query, space)}
        )
        parsed = parse_tool_result(result)
        return parsed.get("results") or parsed.get("items") or [parsed]

    async def get_confluence_page(self, page_id: str, *, as_user: str) -> dict:
        result = await self._call(as_user, _CONF_GET, {"pageId": page_id})
        return parse_tool_result(result)

    # ------------------------------------------------------------ ActPort

    async def execute_step(
        self, step: ActionStep, *, target: ActionTarget, as_user: str
    ) -> dict:
        tool, args = self._dispatch(step, target)
        result = await self._call(as_user, tool, args)
        return {
            "op": step.op,
            "tool": tool,
            "applied": not result.get("isError", False),
            "result": parse_tool_result(result),
        }

    @staticmethod
    def _dispatch(step: ActionStep, target: ActionTarget) -> tuple[str, dict]:
        """Map one typed mutation onto (Atlassian tool name, arguments)."""
        a = step.args
        op = step.op
        if op == "create_subtask":
            return _JIRA_CREATE, {
                "fields": {
                    "parent": {"key": target.key},
                    "summary": a.get("summary", ""),
                    "description": a.get("description", ""),
                    "issuetype": {"name": "Subtask"},
                }
            }
        if op == "transition":
            return _JIRA_TRANSITION, {
                "issueIdOrKey": target.key,
                "transition": a.get("to", ""),
            }
        if op == "add_comment":
            return _JIRA_COMMENT, {
                "issueIdOrKey": target.key,
                "body": a.get("body", ""),
            }
        if op == "assign":
            return _JIRA_EDIT, {
                "issueIdOrKey": target.key,
                "fields": {"assignee": a.get("assignee", "")},
            }
        if op == "update_field":
            return _JIRA_EDIT, {
                "issueIdOrKey": target.key,
                "fields": {a.get("field", ""): a.get("value", "")},
            }
        if op == "update_page":
            return _CONF_UPDATE, {"pageId": target.key, "body": a.get("body", "")}
        if op == "create_page":
            return _CONF_CREATE, {
                "title": a.get("title", ""),
                "body": a.get("body", ""),
                "spaceKey": a.get("space", ""),
            }
        raise ConnectorUnavailable(f"no Atlassian mapping for op {op!r}")
