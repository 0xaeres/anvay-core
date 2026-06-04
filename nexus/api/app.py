"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from nexus.api.deps import get_config_dep, get_registry
from nexus.api.routes import (
    council,
    dashboard,
    products,
    proposals,
    setup,
    skills,
    sources,
)
from nexus.ingest.enrichment_worker import EnrichmentWorker
from nexus.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

_enrichment_stop: asyncio.Event | None = None
_enrichment_task: asyncio.Task | None = None


async def start_enrichment_worker() -> None:
    global _enrichment_stop, _enrichment_task
    config = get_config_dep()
    if not config.ingestion.enrichment_worker.enabled:
        return
    registry = get_registry()
    worker = EnrichmentWorker.from_config(registry=registry, config=config)
    stop = asyncio.Event()
    _enrichment_stop = stop

    async def _run() -> None:
        try:
            await worker.run_forever(stop=stop)
        finally:
            await worker.aclose()

    _enrichment_task = asyncio.create_task(_run())

    def _log_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                "background enrichment worker stopped unexpectedly",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    _enrichment_task.add_done_callback(_log_done)


async def stop_enrichment_worker() -> None:
    global _enrichment_stop, _enrichment_task
    if _enrichment_stop is not None:
        _enrichment_stop.set()
    if _enrichment_task is not None:
        await _enrichment_task
    _enrichment_stop = None
    _enrichment_task = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await start_enrichment_worker()
    try:
        yield
    finally:
        await stop_enrichment_worker()


app = FastAPI(
    title="Nexus API",
    description="Backend for the Nexus context engine. See ENGINEERING.md §8.",
    version="0.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "request method=%s path=%s status=%s elapsed_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(products.router)
app.include_router(dashboard.router)
app.include_router(sources.router)
app.include_router(council.router)
app.include_router(skills.router)
app.include_router(proposals.router)
app.include_router(setup.router)
