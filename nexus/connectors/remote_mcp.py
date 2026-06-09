"""Remote MCP client — MCP SDK over Streamable HTTP.

`mcp_client.py` speaks stdio for local connector subprocesses. This module is
the remote HTTP transport for remote MCP servers.

The client exposes only `initialize`, `tools/list`, and `tools/call`. The
`token_provider` callable is invoked per request so credential refresh stays
outside the transport layer.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TokenProvider = Callable[[], Awaitable[str]]


class RemoteMCPError(RuntimeError):
    pass


class _BearerAuth(httpx.Auth):
    requires_request_body = False
    requires_response_body = False

    def __init__(self, token_provider: TokenProvider):
        self._token_provider = token_provider

    async def async_auth_flow(self, request: httpx.Request):
        token = await self._token_provider()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


class RemoteMCPClient:
    def __init__(
        self,
        *,
        endpoint: str,
        token_provider: TokenProvider,
    ):
        self.endpoint = endpoint
        self._token_provider = token_provider
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._initialized = False

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._session = None
        self._initialized = False

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        try:
            read, write, _session_id = await self._stack.enter_async_context(
                streamablehttp_client(
                    self.endpoint,
                    auth=_BearerAuth(self._token_provider),
                )
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            return self._session
        except Exception as e:
            raise RemoteMCPError(f"connect: {e}") from e

    async def initialize(self) -> dict:
        session = await self._ensure_session()
        try:
            result = await session.initialize()
        except Exception as e:
            raise RemoteMCPError(f"initialize: {e}") from e
        self._initialized = True
        return result.model_dump(mode="json")

    async def list_tools(self) -> list[dict]:
        if not self._initialized:
            await self.initialize()
        assert self._session is not None
        try:
            result = await self._session.list_tools()
        except Exception as e:
            raise RemoteMCPError(f"tools/list: {e}") from e
        return [tool.model_dump(mode="json") for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke a tool. Returns the raw MCP `tools/call` result envelope."""
        if not self._initialized:
            await self.initialize()
        assert self._session is not None
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            raise RemoteMCPError(f"tools/call: {e}") from e
        return result.model_dump(mode="json")


def extract_tool_text(result: dict) -> str:
    """Concatenate the text content blocks of a `tools/call` result."""
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p)


def parse_tool_result(result: dict) -> dict:
    """Best-effort: parse the tool text as JSON, else wrap it as {"text": ...}."""
    text = extract_tool_text(result)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"items": parsed}
    except json.JSONDecodeError:
        return {"text": text}
