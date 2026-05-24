"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from nexus.api.routes import (
    council,
    dashboard,
    products,
    proposals,
    setup,
    skills,
    sources,
)
from nexus.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

app = FastAPI(
    title="Nexus API",
    description="Backend for the Nexus context engine. See ENGINEERING.md §8.",
    version="0.0.1",
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
