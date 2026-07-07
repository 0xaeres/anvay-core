"""Continuous index daemon - subscribes to connector update events and
re-indexes affected resources incrementally.

Two phases:

1. Bootstrap: one-shot full ingest across all configured connectors.
2. Watch: loop over `manager.updates()` forever, calling `reindex_resource`
   for each event. Stays up across MCP server crashes (manager reconnects).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from anvay.config import AnvayConfig
from anvay.connectors.manager import ConnectorManager
from anvay.ingest.embedder import EmbedderClient
from anvay.ingest.enricher import ContextualEnricher
from anvay.ingest.incremental import reindex_resource
from anvay.ingest.indexer_factory import create_indexer
from anvay.ingest.pipeline import embedding_version
from anvay.registry import Registry

log = logging.getLogger(__name__)


async def run_daemon(
    *,
    config: AnvayConfig,
    product_id: str,
    bootstrap: bool = True,
) -> None:
    """Block forever. Caller is responsible for signal handling."""
    async with AsyncExitStack() as stack:
        embedder = EmbedderClient.from_cfg(
            config.models.embedding,
            batch_size=config.ingestion.embed_batch_size,
        )
        enricher = ContextualEnricher(
            base_url=config.models.light.base_url or "https://api.deepinfra.com/v1/openai",
            model=config.models.light.model,
            api_key=config.models.light.api_key,
            enrich_code=config.ingestion.enrich_chunks.code,
            enrich_docs=config.ingestion.enrich_chunks.docs,
            concurrency=config.ingestion.enricher_concurrency,
        )
        indexer = create_indexer(config)
        manager = ConnectorManager(config)
        registry = Registry(config.storage.proposal_queue.parent / "registry.db")
        embed_version = embedding_version(config)

        stack.push_async_callback(embedder.aclose)
        stack.push_async_callback(enricher.aclose)
        stack.push_async_callback(indexer.aclose)
        stack.push_async_callback(manager.stop)

        await indexer.ensure_collections()
        await manager.start()

        if bootstrap:
            await _bootstrap_sync(
                manager=manager,
                product_id=product_id,
                embedder=embedder,
                enricher=enricher,
                indexer=indexer,
                enrich=True,
                registry=registry,
                embed_version=embed_version,
            )

        maintenance = asyncio.create_task(
            _periodic_maintenance(
                registry=registry,
                manager=manager,
                product_id=product_id,
                embedder=embedder,
                enricher=enricher,
                indexer=indexer,
                embed_version=embed_version,
                orphan_sweep=config.ingestion.orphan_sweep,
            )
        )
        stack.callback(maintenance.cancel)

        log.info("daemon: watching for updates (product=%s)", product_id)
        async for event in manager.updates():
            try:
                content = await _read_with_manager(manager, event)
            except Exception as e:
                log.warning("daemon: read failed for %s: %s", event.resource.uri, e)
                continue
            try:
                await reindex_resource(
                    product_id=product_id,
                    resource=event.resource,
                    content=content,
                    embedder=embedder,
                    enricher=enricher,
                    indexer=indexer,
                    registry=registry,
                    # Daemon manifests are namespaced by connector source_id
                    # (e.g. "local:docs"), distinct from the API's
                    # "<source_name>:<repo>" keys.
                    source_key=event.source_id,
                    embedding_version=embed_version,
                )
                log.info("daemon: reindexed %s", event.resource.uri)
            except Exception as e:
                log.exception("daemon: reindex failed for %s: %s", event.resource.uri, e)


async def _bootstrap_sync(
    *,
    manager: ConnectorManager,
    product_id: str,
    embedder: EmbedderClient,
    enricher: ContextualEnricher,
    indexer: Any,
    enrich: bool,
    registry: Registry | None = None,
    embed_version: str = "",
) -> None:
    log.info("daemon: bootstrap sync (product=%s)", product_id)
    started = datetime.now(UTC)
    count = 0
    async for resource, reader in manager.sync_all(product_id):
        try:
            content = await reader(resource)
        except Exception as e:
            log.debug("bootstrap skip %s: %s", resource.uri, e)
            continue
        try:
            await reindex_resource(
                product_id=product_id,
                resource=resource,
                content=content,
                embedder=embedder,
                enricher=enricher,
                indexer=indexer,
                enrich=enrich,
                registry=registry,
                source_key=resource.source_id,
                embedding_version=embed_version,
            )
            count += 1
        except Exception as e:
            log.warning("bootstrap reindex failed %s: %s", resource.uri, e)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    log.info("daemon: bootstrap done - %d resources in %.1fs", count, elapsed)


@dataclass
class _StuckEvent:
    source_id: str
    resource: Any


async def _periodic_maintenance(
    *,
    registry: Registry,
    manager: ConnectorManager,
    product_id: str,
    embedder: EmbedderClient,
    enricher: ContextualEnricher,
    indexer: Any,
    embed_version: str,
    orphan_sweep=None,
    interval_s: float = 300.0,
    stuck_after_s: float = 600.0,
) -> None:
    """Re-run indexing for manifest rows stuck in index_status=pending, and
    (when enabled) sweep orphaned Qdrant points that no manifest row claims.

    Catches resources whose index run died between the pending mark and the
    manifest commit (process kill, Qdrant outage mid-write)."""
    from anvay.ingest.models import ResourceRef
    from anvay.ingest.reconcile import sweep_orphan_points

    while True:
        await asyncio.sleep(interval_s)
        if orphan_sweep is not None and orphan_sweep.enabled:
            try:
                await sweep_orphan_points(
                    registry=registry,
                    indexer=indexer,
                    product_id=product_id,
                    grace_minutes=orphan_sweep.grace_minutes,
                    dry_run=orphan_sweep.dry_run,
                )
            except Exception:
                log.exception("daemon: orphan sweep failed")
        cutoff = (datetime.now(UTC) - timedelta(seconds=stuck_after_s)).isoformat()
        try:
            stuck = registry.list_stuck_index_pending(product_id, older_than_iso=cutoff)
        except Exception:
            log.exception("daemon: stuck-pending query failed")
            continue
        for row in stuck:
            uri = row["resourceUri"]
            ref = ResourceRef(
                source_id=row["sourceKey"],
                uri=uri,
                mime=row.get("mime") or "text/plain",
            )
            try:
                content = await _read_with_manager(
                    manager, _StuckEvent(source_id=row["sourceKey"], resource=ref)
                )
            except Exception as e:
                log.warning("daemon: sweep read failed for %s: %s", uri, e)
                continue
            try:
                await reindex_resource(
                    product_id=product_id,
                    resource=ref,
                    content=content,
                    embedder=embedder,
                    enricher=enricher,
                    indexer=indexer,
                    registry=registry,
                    source_key=row["sourceKey"],
                    embedding_version=embed_version,
                )
                log.info("daemon: sweep re-indexed stuck resource %s", uri)
            except Exception:
                log.exception("daemon: sweep reindex failed for %s", uri)


async def _read_with_manager(manager: ConnectorManager, event) -> str:
    """Open a transient client to read the updated resource."""
    from anvay.connectors.local_fs import LocalFsSource
    from anvay.connectors.mcp_client import McpClientHandle

    for state in manager._states.values():
        if event.source_id == f"mcp:{state.cfg.name}":
            async with McpClientHandle(state.cfg) as handle:
                return await handle.read_resource(event.resource.uri)
        if event.source_id == f"local:{state.cfg.name}" and state.cfg.type == "local_fs":
            extras = state.cfg.model_dump(exclude={"name", "type", "watch"})
            src = LocalFsSource(
                __import__("anvay.connectors.local_fs", fromlist=["LocalFsConfig"]).LocalFsConfig(
                    root=Path(extras.get("root", "."))
                )
            )
            return await src.read_resource(event.resource)
    raise RuntimeError(f"no connector for source_id={event.source_id}")
