"""Remote MCP client — JSON-RPC 2.0 over Streamable HTTP.

`mcp_client.py` speaks stdio for local connector subprocesses. This module is
the small HTTP transport for remote MCP servers: JSON-RPC requests are POSTed to
one endpoint, and the server may reply as JSON or as an SSE stream.

The client exposes only `initialize`, `tools/list`, and `tools/call`. The
`token_provider` callable is invoked per request so credential refresh stays
outside the transport layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2025-06-18"
TokenProvider = Callable[[], Awaitable[str]]


class RemoteMCPError(RuntimeError):
    pass


def _extract_jsonrpc(resp: httpx.Response, request_id: int) -> dict:
    """Pull the JSON-RPC message matching `request_id` from a JSON or SSE response."""
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        for line in resp.text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                msg = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("id") == request_id:
                return msg
        raise RemoteMCPError("no matching JSON-RPC response found in SSE stream")
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise RemoteMCPError(f"non-JSON response: {resp.text[:200]!r}") from e


class RemoteMCPClient:
    def __init__(
        self,
        *,
        endpoint: str,
        token_provider: TokenProvider,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.endpoint = endpoint
        self._token_provider = token_provider
        self._client = http_client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = http_client is None
        self._session_id: str | None = None
        self._req_id = 0
        self._initialized = False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _headers(self) -> dict[str, str]:
        token = await self._token_provider()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        request_id = self._next_id()
        body: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            body["params"] = params
        try:
            resp = await self._client.post(
                self.endpoint, json=body, headers=await self._headers()
            )
        except httpx.HTTPError as e:
            raise RemoteMCPError(f"{method}: transport error: {e}") from e
        if resp.status_code >= 400:
            raise RemoteMCPError(
                f"{method}: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        payload = _extract_jsonrpc(resp, request_id)
        if "error" in payload:
            raise RemoteMCPError(f"{method}: JSON-RPC error: {payload['error']}")
        return payload.get("result", {})

    async def _notify(self, method: str) -> None:
        """Fire-and-forget JSON-RPC notification (no id, no response expected)."""
        try:
            await self._client.post(
                self.endpoint,
                json={"jsonrpc": "2.0", "method": method},
                headers=await self._headers(),
            )
        except httpx.HTTPError as e:  # best-effort — don't fail the whole call
            log.debug("remote-mcp notification %s failed: %s", method, e)

    async def initialize(self) -> dict:
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "nexus", "version": "0.1"},
            },
        )
        await self._notify("notifications/initialized")
        self._initialized = True
        return result

    async def list_tools(self) -> list[dict]:
        if not self._initialized:
            await self.initialize()
        result = await self._rpc("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke a tool. Returns the raw MCP `tools/call` result envelope."""
        if not self._initialized:
            await self.initialize()
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})


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
