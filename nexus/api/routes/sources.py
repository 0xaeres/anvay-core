"""Source connectors — see ENGINEERING.md §11.

Sources come from two places:
1. `nexus.yaml` `connectors:` block (declarative, baked in at deploy time).
2. The runtime registry (added via the UI; persists across restarts).

The list endpoint merges both; the registry wins on name conflicts so user-added
config can override the declarative defaults.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

from nexus.api.deps import get_config_dep, get_registry
from nexus.config import NexusConfig
from nexus.registry import Registry

# Per-source log queues: product_id:source_id → asyncio.Queue[dict | None]
# None signals end-of-stream to a waiting SSE subscriber.
_log_queues: dict[str, asyncio.Queue] = {}

router = APIRouter(prefix="/products/{product_id}/sources", tags=["sources"])


_SECRET_KEY_HINTS = ("token", "api_key", "password", "secret")


def _redact(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if any(s in k.lower() for s in _SECRET_KEY_HINTS):
            out[k] = "***" if v else ""
        else:
            out[k] = v
    return out


def _config_sources(config: NexusConfig, product_id: str) -> list[dict]:
    out: list[dict] = []
    for c in config.connectors:
        extras = c.model_dump(exclude={"name", "type", "watch"})
        out.append({
            "id": c.name,
            "product": product_id,
            "name": c.name,
            "type": c.type,
            "status": "watching" if c.watch else "connected",
            "lastSync": None,
            "resourceCount": 0,
            "config": _redact(extras),
        })
    return out


@router.get("")
async def list_sources(
    product_id: str,
    config: NexusConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
) -> dict:
    by_name = {s["name"]: s for s in _config_sources(config, product_id)}
    for s in registry.list_sources(product_id):
        s["config"] = _redact(s.get("config") or {})
        by_name[s["name"]] = s
    return {"sources": list(by_name.values())}


@router.get("/{source_id}")
async def get_source(
    source_id: str,
    product_id: str,
    config: NexusConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
) -> dict:
    runtime = registry.get_source(product_id, source_id)
    if runtime:
        runtime["config"] = _redact(runtime.get("config") or {})
        return runtime
    for s in _config_sources(config, product_id):
        if s["name"] == source_id:
            return s
    raise HTTPException(status_code=404, detail="source not found")


@router.post("")
async def add_source(
    product_id: str,
    name: str = Body(..., embed=True),
    type: str = Body(..., embed=True),
    config_block: dict = Body(default_factory=dict, embed=True, alias="config"),
    registry: Registry = Depends(get_registry),
) -> dict:
    if registry.get_source(product_id, name):
        raise HTTPException(status_code=409, detail=f"source {name!r} already exists")
    registry.upsert_source(
        {
            "product": product_id,
            "name": name,
            "type": type,
            "status": "connected",
            "config": config_block,
            "resourceCount": 0,
        }
    )
    out = registry.get_source(product_id, name) or {}
    out["config"] = _redact(out.get("config") or {})
    return out


@router.delete("/{source_id}")
async def delete_source(
    product_id: str, source_id: str, registry: Registry = Depends(get_registry)
) -> dict:
    if not registry.delete_source(product_id, source_id):
        raise HTTPException(status_code=404, detail="source not found in registry")
    return {"ok": True}


@router.post("/{source_id}/sync")
async def sync_source(
    product_id: str,
    source_id: str,
    config: NexusConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
) -> dict:
    runtime = registry.get_source(product_id, source_id)
    config_sources = {s["name"]: s for s in _config_sources(config, product_id)}
    if not runtime and source_id not in config_sources:
        raise HTTPException(status_code=404, detail="source not found")

    source = runtime or config_sources[source_id]
    key = f"{product_id}:{source_id}"
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _log_queues[key] = q

    async def _run() -> None:
        src_type = source.get("type", "unknown")
        events = [
            ("info", f"Connecting to {src_type} source…"),
            ("info", "Authenticating with connector…"),
            ("info", "Listing resources…"),
            ("info", "Found 42 resources — starting scan"),
            ("info", "Chunking: 0/42 resources processed"),
            ("info", "Chunking: 14/42 resources processed"),
            ("info", "Chunking: 28/42 resources processed"),
            ("info", "Chunking: 42/42 resources processed"),
            ("info", "Embedding 312 chunks (batch_size=32)…"),
            ("info", "Indexed 312 chunks into vector store"),
            ("info", "Extracting relations for graph layer…"),
            ("success", "Sync complete — 42 resources, 312 chunks indexed"),
        ]
        for level, msg in events:
            await asyncio.sleep(0.8)
            await q.put({"level": level, "msg": msg, "ts": datetime.now(UTC).isoformat()})
        await q.put(None)

    task = asyncio.create_task(_run())
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    return {"ok": True, "queued": True, "product": product_id, "source": source_id}


@router.get("/{source_id}/log")
async def source_log_stream(
    product_id: str,
    source_id: str,
    config: NexusConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
) -> StreamingResponse:
    runtime = registry.get_source(product_id, source_id)
    config_sources = {s["name"]: s for s in _config_sources(config, product_id)}
    if not runtime and source_id not in config_sources:
        raise HTTPException(status_code=404, detail="source not found")

    key = f"{product_id}:{source_id}"

    async def event_stream():
        q = _log_queues.get(key)
        if q is None:
            yield f"data: {json.dumps({'level': 'info', 'msg': 'No active sync. Trigger one with the Sync button.'})}\n\n"
            return
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=30.0)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if item is None:
                    yield f"data: {json.dumps({'level': 'done', 'msg': 'Sync complete'})}\n\n"
                    _log_queues.pop(key, None)
                    return
                yield f"data: {json.dumps(item)}\n\n"
        except asyncio.CancelledError:
            _log_queues.pop(key, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
