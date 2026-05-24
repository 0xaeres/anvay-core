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
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from git import Repo

from nexus.api.deps import get_config_dep, get_registry
from nexus.config import NexusConfig
from nexus.connectors.local_fs import LocalFsConfig, LocalFsSource
from nexus.ingest.models import ResourceRef
from nexus.ingest.pipeline import run_ingest
from nexus.registry import Registry
from nexus.setup.bootstrap import _authenticated_clone_url

log = logging.getLogger(__name__)

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
    q: asyncio.Queue = asyncio.Queue(maxsize=2048)
    _log_queues[key] = q

    async def _run() -> None:
        await _real_ingest(
            product_id=product_id,
            source=source,
            runtime=runtime,
            config=config,
            registry=registry,
            q=q,
        )

    task = asyncio.create_task(_run())
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    return {"ok": True, "queued": True, "product": product_id, "source": source_id}


# ---------------------------------------------------------------- real ingest


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _emit(q: asyncio.Queue, level: str, msg: str, **extra) -> None:
    try:
        await q.put({"level": level, "msg": msg, "ts": _now(), **extra})
    except Exception:
        # Queue full or closed — degrade silently. The pipeline keeps going.
        log.debug("ingest log queue rejected: %s %s", level, msg)


class _ProgressAdapter:
    """Wraps a LocalFsSource to emit structured progress events per resource read.

    Implements the `_Source` protocol from nexus.ingest.pipeline.
    """

    def __init__(self, inner: LocalFsSource, put: Callable[..., None], total: int):
        self.inner = inner
        self.put = put
        self.source_id = inner.source_id
        self.total = total
        self.processed = 0

    async def list_resources(self) -> AsyncIterator[ResourceRef]:
        async for ref in self.inner.list_resources():
            yield ref

    async def read_resource(self, ref: ResourceRef) -> str:
        self.processed += 1
        pct = round(100 * self.processed / self.total) if self.total else 0
        short = ref.uri.rsplit("/", 1)[-1] or ref.uri
        self.put(
            "progress",
            f"Indexing {self.processed} of {self.total} — {short[:80]}",
            done=self.processed,
            total=self.total,
            pct=pct,
        )
        return await self.inner.read_resource(ref)


async def _real_ingest(
    *,
    product_id: str,
    source: dict,
    runtime: dict | None,
    config: NexusConfig,
    registry: Registry,
    q: asyncio.Queue,
) -> None:
    """Drive the actual run_ingest pipeline + stream progress."""
    src_type = source.get("type", "unknown")
    cleanup_dir: Path | None = None
    try:
        await _emit(q, "info", f"Starting sync for '{source.get('name')}' ({src_type})")

        if src_type == "github":
            local_root, cleanup_dir = await _clone_github(source, q)
        elif src_type in ("filesystem", "local_fs"):
            roots = source.get("config", {}).get("roots") or [
                source.get("config", {}).get("root")
            ]
            if not roots or not roots[0]:
                await _emit(q, "error", "filesystem source missing 'root' in config")
                return
            local_root = Path(str(roots[0]))
            if not local_root.is_dir():
                await _emit(q, "error", f"filesystem root not a directory: {local_root}")
                return
        else:
            await _emit(
                q,
                "error",
                f"Connector type {src_type!r} is not yet wired for sync. "
                "Currently supported: github, filesystem.",
            )
            return

        await _emit(q, "info", f"Walking {local_root}…")

        def _put(level: str, msg: str, **extra) -> None:
            try:
                q.put_nowait({"level": level, "msg": msg, "ts": _now(), **extra})
            except asyncio.QueueFull:
                log.debug("ingest log queue full; dropping: %s", msg)

        fs_source = LocalFsSource(LocalFsConfig(root=local_root))

        await _emit(q, "info", "Counting files…")
        total = 0
        async for _ in fs_source.list_resources():
            total += 1
        await _emit(q, "started", f"Starting ingestion — {total} files found", total=total)

        adapter = _ProgressAdapter(fs_source, _put, total=total)

        await _emit(q, "info", "Running ingest pipeline (chunk → enrich → embed → index)…")
        stats = await run_ingest(
            product_id=product_id, source=adapter, config=config, enrich=True
        )

        if stats.embed_errors:
            await _emit(
                q,
                "warn",
                f"{stats.embed_errors} batch(es) failed to embed — "
                "chunks too large for embedder token limit. "
                "Restart llama-server with --ubatch-size 2048 and re-sync.",
            )
        await _emit(
            q,
            "success" if not stats.embed_errors else "done",
            (
                f"Sync complete — {stats.resources_indexed} indexed, "
                f"{stats.resources_skipped} skipped, "
                f"{stats.chunks_indexed} chunks in vector store"
            ),
        )

        if runtime:
            registry.upsert_source({
                **runtime,
                "lastSync": _now(),
                "resourceCount": stats.resources_indexed,
                "status": "connected",
            })

        # ---- repo map: extract symbol outline while local files still exist ----
        try:
            from nexus.retrieval.repomap import (
                extract_repo_map,
                repomap_path_for,
                save_repo_map,
            )

            await _emit(q, "info", "Building repo map (tree-sitter symbol outline)…")
            rm = await asyncio.to_thread(extract_repo_map, local_root)
            state_dir = config.storage.proposal_queue.parent
            save_repo_map(rm, repomap_path_for(state_dir, product_id))
            await _emit(
                q,
                "info",
                f"Repo map: {len(rm.symbols)} symbols across {rm.file_count} files",
            )
        except Exception as e:
            log.warning("repomap build failed: %s", e)
            await _emit(q, "warn", f"Repo map build failed: {e} (council will still run)")

    except Exception as e:
        log.exception(
            "sync_source failed for product=%s source=%s", product_id, source.get("name")
        )
        await _emit(q, "error", f"Ingest failed: {type(e).__name__}: {e}")
        if runtime:
            try:
                registry.upsert_source({**runtime, "status": "error"})
            except Exception:
                log.debug("failed to mark source 'error' status")
    finally:
        if cleanup_dir is not None:
            await asyncio.to_thread(shutil.rmtree, str(cleanup_dir), ignore_errors=True)
        await q.put(None)


async def _clone_github(source: dict, q: asyncio.Queue) -> tuple[Path, Path]:
    """Shallow-clone the first configured repo into a temp dir. Returns (root, cleanup_dir)."""
    cfg = source.get("config") or {}
    repos = cfg.get("repos") or []
    token = cfg.get("token") or None
    if not repos:
        raise ValueError(
            "github source has no 'repos' configured (expected a GitHub URL)"
        )
    url = str(repos[0]).rstrip("/").removesuffix(".git")
    if not url.startswith("https://github.com/") and not url.startswith("git@"):
        raise ValueError(
            f"unsupported github URL: {url!r} (expected https://github.com/<org>/<repo>)"
        )

    await _emit(q, "info", f"Cloning {url} (shallow, depth=1)…")
    tmp = Path(tempfile.mkdtemp(prefix="nexus-ingest-"))
    clone_path = tmp / "repo"
    auth_url = _authenticated_clone_url(url + ".git", token)

    try:
        await asyncio.to_thread(
            Repo.clone_from, auth_url, str(clone_path), depth=1, multi_options=["--quiet"]
        )
    except Exception as e:
        shutil.rmtree(str(tmp), ignore_errors=True)
        raise RuntimeError(f"git clone failed: {e}") from e

    await _emit(q, "info", "Clone complete")
    return clone_path, tmp


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
