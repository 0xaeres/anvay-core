"""Remote MCP client — MCP SDK over Streamable HTTP.

`mcp_client.py` speaks stdio for local connector subprocesses. This module is
the remote HTTP transport for remote MCP servers.

The client exposes only `initialize`, `tools/list`, and `tools/call`. The
`token_provider` callable is invoked per request so credential refresh stays
outside the transport layer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

TokenProvider = Callable[[], Awaitable[str]]


class RemoteMCPError(RuntimeError):
    pass


class _BearerAuth(httpx.Auth):
    requires_request_body = False
    requires_response_body = False

    def __init__(self, token_provider: TokenProvider):
        """
        Initialize the remote MCP client with a per-request token provider.
        
        Parameters:
            token_provider (TokenProvider): An async callable that returns a bearer token string for each request; used to populate the `Authorization: Bearer <token>` header.
        """
        self._token_provider = token_provider

    async def async_auth_flow(self, request: httpx.Request):
        """
        Attach a Bearer Authorization header to the given HTTPX request using the configured token provider and yield the modified request.
        
        Parameters:
            request (httpx.Request): The outgoing HTTP request to modify and yield.
        """
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
        """
        Create a RemoteMCPClient configured to connect to the given endpoint using the provided token provider.
        
        Parameters:
            endpoint (str): Base URL of the remote MCP Streamable HTTP endpoint.
            token_provider (TokenProvider): Async callable that returns a Bearer token string for per-request authentication.
        
        Notes:
            Initializes internal resource-management and session state; call `aclose` to release resources.
        """
        self.endpoint = endpoint
        self._token_provider = token_provider
        self._stack = AsyncExitStack()
        self._session_lock = asyncio.Lock()
        self._session: ClientSession | None = None
        self._initialized = False
        self._init_result: dict | None = None

    async def aclose(self) -> None:
        """
        Close all managed resources and reset the client's internal state.
        
        Closes the internal AsyncExitStack (shutting down transport and session resources), sets the cached session to None, and marks the client as not initialized.
        """
        await self._stack.aclose()
        self._stack = AsyncExitStack()
        self._session = None
        self._initialized = False
        self._init_result = None

    async def _ensure_session(self) -> ClientSession:
        """
        Ensure and return an active MCP ClientSession, creating and caching it if necessary.
        
        Returns:
            ClientSession: The active MCP client session used for RPC calls.
        
        Raises:
            RemoteMCPError: If establishing the streamable HTTP transport or client session fails.
        """
        if self._session is not None:
            return self._session
        async with self._session_lock:
            if self._session is not None:
                return self._session
            return await self._connect_session()

    async def _connect_session(self) -> ClientSession:
        """
        Create and cache a ClientSession using the MCP streamable HTTP transport.
        
        Returns:
            ClientSession: The newly connected session.
        
        Raises:
            RemoteMCPError: If transport or session setup fails.
        """
        try:
            http_client = await self._stack.enter_async_context(
                httpx.AsyncClient(auth=_BearerAuth(self._token_provider))
            )
            read, write, _session_id = await self._stack.enter_async_context(
                streamable_http_client(
                    self.endpoint,
                    http_client=http_client,
                )
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            return self._session
        except Exception as e:
            raise RemoteMCPError(f"connect: {e}") from e

    async def initialize(self) -> dict:
        """
        Initialize the remote MCP session and mark this client as initialized.
        
        On success, the client's initialized state is set to True.
        
        Returns:
            dict: The session initialization response serialized as a JSON-compatible dictionary.
        
        Raises:
            RemoteMCPError: If the session initialization fails.
        """
        if self._initialized and self._init_result is not None:
            return self._init_result
        async with self._session_lock:
            if self._initialized and self._init_result is not None:
                return self._init_result
            session = self._session or await self._connect_session()
            try:
                result = await session.initialize()
            except Exception as e:
                raise RemoteMCPError(f"initialize: {e}") from e
            self._init_result = result.model_dump(mode="json")
            self._initialized = True
            return self._init_result

    async def list_tools(self) -> list[dict]:
        """
        List available tools exposed by the remote MCP session.
        
        Returns:
            list[dict]: A list of tool definitions serialized as JSON-compatible dictionaries.
        """
        if not self._initialized:
            await self.initialize()
        if self._session is None:
            raise RemoteMCPError("session not initialized")
        try:
            result = await self._session.list_tools()
        except Exception as e:
            raise RemoteMCPError(f"tools/list: {e}") from e
        return [tool.model_dump(mode="json") for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Invoke a named tool on the remote MCP and obtain its result envelope.
        
        Parameters:
            name (str): Tool identifier to call.
            arguments (dict): Mapping of arguments to pass to the tool.
        
        Returns:
            dict: The MCP `tools/call` result envelope serialized as a JSON-compatible dictionary.
        
        Raises:
            RemoteMCPError: If the underlying session call or transport fails.
        """
        if not self._initialized:
            await self.initialize()
        if self._session is None:
            raise RemoteMCPError("session not initialized")
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            raise RemoteMCPError(f"tools/call: {e}") from e
        return result.model_dump(mode="json")


def extract_tool_text(result: dict) -> str:
    """
    Extract and concatenate the plain-text blocks from a tools/call result.
    
    Selects entries in `result["content"]` whose `type` is `"text"`, extracts their `text` fields, and joins them with newline characters.
    
    Parameters:
        result (dict): A `tools/call` result envelope; expected to contain a `content` sequence of block objects.
    
    Returns:
        str: The concatenated text of all text blocks separated by `\n`, or an empty string if none are present.
    """
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
