"""Durable background enrichment worker.

Foreground source sync writes raw dense + BM25 vectors quickly, then queues
changed resources here. The worker enriches those resources and upserts the
same deterministic chunk IDs, upgrading retrieval quality without blocking sync.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from anvay.config import AnvayConfig
from anvay.ingest.chunker import chunk_resource
from anvay.ingest.embedder import EmbedderClient
from anvay.ingest.enricher import ContextualEnricher
from anvay.ingest.indexer import Indexer
from anvay.ingest.indexer_factory import create_indexer
from anvay.ingest.models import ResourceRef
from anvay.ingest.pipeline import embedding_version, enrichment_version
from anvay.retrieval.sparse import aencode_passages

log = logging.getLogger(__name__)


class _Registry(Protocol):
    def claim_enrichment_job(self, *, max_attempts: int) -> dict | None: ...
    def complete_enrichment_job(self, job_id: str) -> bool: ...
    def fail_enrichment_job(self, job_id: str, *, error: str, max_attempts: int) -> None: ...
    def reset_running_enrichment_jobs(self) -> int: ...
    def get_resource_manifest(
        self, product_id: str, source_key: str, resource_uri: str
    ) -> dict | None: ...
    def update_resource_enrichment(
        self,
        product_id: str,
        source_key: str,
        resource_uri: str,
        *,
        enrichment_version: str,
        enrichment_status: str,
    ) -> bool: ...


@dataclass
class EnrichmentWorker:
    registry: _Registry
    config: AnvayConfig
    embedder: EmbedderClient
    enricher: ContextualEnricher
    indexer: Indexer
    max_attempts: int = 3
    poll_interval_s: float = 5.0

    @classmethod
    def from_config(cls, *, registry: _Registry, config: AnvayConfig) -> EnrichmentWorker:
        return cls(
            registry=registry,
            config=config,
            embedder=EmbedderClient.from_cfg(
                config.models.embedding,
                batch_size=config.ingestion.embed_batch_size,
            ),
            enricher=ContextualEnricher(
                base_url=config.models.light.base_url or "https://api.deepinfra.com/v1/openai",
                model=config.models.light.model,
                api_key=config.models.light.api_key,
                enrich_code=config.ingestion.enrich_chunks.code,
                enrich_docs=config.ingestion.enrich_chunks.docs,
                concurrency=config.ingestion.enricher_concurrency,
            ),
            indexer=create_indexer(config),
            max_attempts=config.ingestion.enrichment_worker.max_attempts,
            poll_interval_s=config.ingestion.enrichment_worker.poll_interval_s,
        )

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await self.enricher.aclose()
        await self.indexer.aclose()

    async def run_forever(self, *, stop: asyncio.Event | None = None) -> None:
        reset = self.registry.reset_running_enrichment_jobs()
        if reset:
            log.info("enrichment worker: reset %d running job(s)", reset)
        await self.indexer.ensure_collections()
        while stop is None or not stop.is_set():
            processed = await self.process_one()
            if not processed:
                try:
                    if stop is None:
                        await asyncio.sleep(self.poll_interval_s)
                    else:
                        await asyncio.wait_for(stop.wait(), timeout=self.poll_interval_s)
                except TimeoutError:
                    pass

    async def process_one(self) -> bool:
        job = self.registry.claim_enrichment_job(max_attempts=self.max_attempts)
        if job is None:
            return False
        try:
            if not self._job_still_current(job):
                log.info(
                    "enrichment worker: dropping stale job product=%s uri=%s",
                    job["product"],
                    job["resourceUri"],
                )
                self.registry.complete_enrichment_job(job["id"])
                return True
            inserted = await self._process_job(job)
            log.info(
                "enrichment worker: enriched product=%s uri=%s chunks=%d",
                job["product"],
                job["resourceUri"],
                inserted,
            )
            self.registry.update_resource_enrichment(
                job["product"],
                job["sourceKey"],
                job["resourceUri"],
                enrichment_version=enrichment_version(self.config),
                enrichment_status="complete",
            )
            self.registry.complete_enrichment_job(job["id"])
        except Exception as e:
            log.exception("enrichment worker: job failed id=%s", job.get("id"))
            self.registry.fail_enrichment_job(
                job["id"], error=f"{type(e).__name__}: {e}", max_attempts=self.max_attempts
            )
        return True

    def _job_still_current(self, job: dict) -> bool:
        manifest = self.registry.get_resource_manifest(
            job["product"], job["sourceKey"], job["resourceUri"]
        )
        return bool(manifest and manifest.get("contentHash") == job["contentHash"])

    async def _process_job(self, job: dict) -> int:
        resource = ResourceRef(
            source_id=job["sourceId"],
            uri=job["resourceUri"],
            mime=job.get("mime") or "",
            size_bytes=job.get("sizeBytes"),
            last_modified=job.get("lastModified"),
        )
        chunks = chunk_resource(job["product"], resource, job["content"])
        if not chunks:
            return 0
        chunks = await self.enricher.enrich(chunks, doc_contents={resource.uri: job["content"]})
        if not any(chunk.context_summary for chunk in chunks):
            return 0
        embedded = await self.embedder.embed_chunks(chunks)
        sparse_by_id = {}
        if getattr(self.indexer, "requires_sparse_vectors", True):
            sparse_vecs = await aencode_passages([c.text_for_embedding() for c in chunks])
            sparse_by_id = {c.id: sv for c, sv in zip(chunks, sparse_vecs, strict=True)}
        content_hash_by_id = {c.id: job["contentHash"] for c in chunks}
        return await self.indexer.upsert(
            embedded,
            sparse_by_id=sparse_by_id,
            source_key=job["sourceKey"],
            content_hash_by_id=content_hash_by_id,
            embedding_version=embedding_version(self.config),
            indexed_at=datetime.now(UTC).isoformat(),
        )
