"""RemoteMCPClient — JSON-RPC 2.0 over Streamable HTTP."""

from __future__ import annotations

import asyncio
import json

import httpx

from nexus.connectors.remote_mcp import (
    RemoteMCPClient,
    RemoteMCPError,
    parse_tool_result,
)


async def _token() -> str:
    return "fake-bearer-token"


def _client(handler) -> RemoteMCPClient:
    return RemoteMCPClient(
        endpoint="https://mcp.example/v1/mcp",
        token_provider=_token,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _ok(request: httpx.Request) -> httpx.Response:
    """A handler that initialises and answers one tools/call with JSON."""
    body = json.loads(request.content)
    method = body.get("method")
    if method == "initialize":
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
    if method == "tools/call":
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": '{"ok": true}'}]},
            },
        )
    return httpx.Response(202)  # notifications/initialized


def test_call_tool_json_rpc_framing() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok(request)

    client = _client(handler)
    result = asyncio.run(client.call_tool("getJiraIssue", {"issueIdOrKey": "P-1"}))

    assert parse_tool_result(result) == {"ok": True}
    methods = [m["method"] for m in seen]
    assert methods[0] == "initialize"  # handshake happens first
    assert "tools/call" in methods
    call = next(m for m in seen if m["method"] == "tools/call")
    assert call["jsonrpc"] == "2.0"
    assert call["params"]["name"] == "getJiraIssue"
    assert call["params"]["arguments"] == {"issueIdOrKey": "P-1"}


def test_json_rpc_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        if "id" in body:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -32000, "message": "boom"},
                },
            )
        return httpx.Response(202)

    client = _client(handler)
    try:
        asyncio.run(client.call_tool("x", {}))
        raise AssertionError("expected RemoteMCPError")
    except RemoteMCPError as e:
        assert "boom" in str(e)


def test_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream down")

    client = _client(handler)
    try:
        asyncio.run(client.call_tool("x", {}))
        raise AssertionError("expected RemoteMCPError")
    except RemoteMCPError as e:
        assert "500" in str(e)


def test_sse_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}})
        if body.get("method") == "tools/call":
            msg = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": "hello"}]},
            }
            return httpx.Response(
                200,
                content=f"event: message\ndata: {json.dumps(msg)}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(202)

    client = _client(handler)
    result = asyncio.run(client.call_tool("x", {}))
    assert parse_tool_result(result) == {"text": "hello"}


def test_session_id_header_is_echoed() -> None:
    seen_headers: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"Mcp-Session-Id": "sess-42"},
            )
        return _ok(request)

    client = _client(handler)
    asyncio.run(client.call_tool("x", {}))
    # The tools/call request (after initialize) must echo the session id.
    assert any(h.get("mcp-session-id") == "sess-42" for h in seen_headers)
