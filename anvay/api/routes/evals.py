"""Multi-product eval runs + live/replay SSE.

Mirrors the council async-job pattern (``anvay/api/routes/council.py`` +
``anvay/council/runner.py``): POST kicks off a background ``asyncio`` task,
progress streams over an in-process hub, and results are read back from the
filesystem artifacts the harness already writes under ``artifacts/evals/``.

Unlike council, routes are **multi-product** (not nested under one
``product_id``) — the whole point is cross-product comparison so a RAG change
is judged on the cross-product mean, not a single corpus. Each run is still
internally product-scoped: every requested product is guarded with
``assert_product_access`` and unknown products are rejected.

Job status lives in-process (ephemeral); the durable record is the per-run
``summary.json`` on disk, so history survives restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from anvay.api.authz import assert_product_access
from anvay.api.deps import get_config_dep, get_registry
from anvay.config import AnvayConfig
from anvay.registry import Registry
from evals.corpus import PRODUCTS, SuiteName, has_suite_data
from evals.harness import (
    ALL_SUITES,
    DEFAULT_OUT_DIR,
    EvalRunArtifact,
    SuiteArtifact,
    run_for_product,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["evals"])

_CONFIG_PATH = Path("anvay.yaml")

# suite -> dashboard section (retrieval | answer | code)
_SECTION: dict[str, str] = {"retrieval": "retrieval", "rag": "answer", "code": "code"}


# ---------------------------------------------------------------- shapes


class EvalJobRef(BaseModel):
    job_id: str
    status: str


class EvalJobStatus(BaseModel):
    job_id: str
    status: str  # running | completed | failed
    products: list[str]
    suites: list[str]
    started_at: str
    completed_at: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # "product/suite" pairs with no data
    error: str | None = None


class SuiteSummary(BaseModel):
    suite: str
    section: str  # retrieval | answer | code
    passed: bool
    product_id: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    notes: list[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    run_id: str
    product_id: str
    generated_at: str
    passed: bool
    suites: list[SuiteSummary]


class EvalRunDetail(EvalRunSummary):
    config_fingerprint: dict = Field(default_factory=dict)
    per_suite: dict[str, dict] = Field(default_factory=dict)  # suite -> raw output json


class ProductEvalInfo(BaseModel):
    product_id: str
    language: str
    repo_url: str
    has_retrieval: bool
    has_rag: bool


class StartRunBody(BaseModel):
    products: list[str]
    suites: list[str] = Field(default_factory=lambda: list(ALL_SUITES))


# ---------------------------------------------------------------- job hub


class _JobHub:
    """One asyncio.Queue per job_id; SSE readers fan out via queues."""

    _END = "__end__"

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict | str]]] = {}
        self._live: set[str] = set()
        self._completed: set[str] = set()
        self._lock = asyncio.Lock()

    async def start(self, job_id: str) -> None:
        async with self._lock:
            self._live.add(job_id)
            self._completed.discard(job_id)

    async def publish(self, job_id: str, event: dict) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(job_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("eval job %s: subscriber queue full; dropping event", job_id)

    async def finish(self, job_id: str) -> None:
        async with self._lock:
            self._live.discard(job_id)
            self._completed.add(job_id)
            queues = list(self._subscribers.get(job_id, []))
        for q in queues:
            await q.put(self._END)

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
            if job_id in self._completed:
                await q.put(self._END)
        return q

    async def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        async with self._lock:
            queues = self._subscribers.get(job_id)
            if queues and q in queues:
                queues.remove(q)
            if not queues:
                self._subscribers.pop(job_id, None)

    def is_live(self, job_id: str) -> bool:
        return job_id in self._live and job_id not in self._completed


HUB = _JobHub()
# In-process job status records, keyed by job_id (ephemeral; disk is durable).
_JOBS: dict[str, EvalJobStatus] = {}
# Anchor background tasks so the GC does not cancel them mid-flight.
_RUNNING: set[asyncio.Task] = set()


def _make_job_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"ej_{ts}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------- runner


async def _run_eval_job(
    *,
    job_id: str,
    products: list[str],
    suites: tuple[SuiteName, ...],
    config: AnvayConfig,
) -> None:
    status = _JOBS[job_id]
    try:
        for product_id in products:
            p = PRODUCTS[product_id]
            runnable = tuple(s for s in suites if has_suite_data(p, s))
            skipped = [f"{product_id}/{s}" for s in suites if not has_suite_data(p, s)]
            status.skipped.extend(skipped)
            await HUB.publish(
                job_id,
                {
                    "event": "product_start",
                    "data": {"product_id": product_id, "suites": list(runnable)},
                },
            )
            if not runnable:
                await HUB.publish(
                    job_id,
                    {
                        "event": "product_done",
                        "data": {"product_id": product_id, "skipped": True, "run_id": None},
                    },
                )
                continue
            artifact = await run_for_product(
                product_id=product_id,
                suites=runnable,
                config=config,
                config_path=_CONFIG_PATH,
                out_dir=DEFAULT_OUT_DIR,
            )
            status.run_ids.append(artifact.run_id)
            await HUB.publish(
                job_id,
                {
                    "event": "product_done",
                    "data": {
                        "product_id": product_id,
                        "skipped": False,
                        "run_id": artifact.run_id,
                        "passed": artifact.passed,
                    },
                },
            )
        status.status = "completed"
        status.completed_at = datetime.now(UTC).isoformat()
        await HUB.publish(job_id, {"event": "job_done", "data": {"run_ids": status.run_ids}})
    except Exception as e:  # pragma: no cover - defensive
        log.exception("eval job %s crashed", job_id)
        status.status = "failed"
        status.completed_at = datetime.now(UTC).isoformat()
        status.error = f"{type(e).__name__}: {e}"
        await HUB.publish(
            job_id,
            {"event": "error", "data": {"message": str(e), "type": type(e).__name__}},
        )
    finally:
        await HUB.finish(job_id)


def _validate_products(request: Request, registry: Registry, products: list[str]) -> None:
    if not products:
        raise HTTPException(status_code=400, detail="no products specified")
    unknown = [pid for pid in products if pid not in PRODUCTS]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown product(s): {', '.join(unknown)}")
    for pid in products:
        assert_product_access(request, registry, pid, action="council")


def _parse_suites(suites: list[str]) -> tuple[SuiteName, ...]:
    invalid = [s for s in suites if s not in ALL_SUITES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"unknown suite(s): {', '.join(invalid)}")
    return tuple(dict.fromkeys(suites)) or ALL_SUITES  # stable de-dupe


# ---------------------------------------------------------------- endpoints


@router.post("/evals/runs")
async def start_run(
    request: Request,
    body: StartRunBody = Body(...),
    config: AnvayConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
) -> EvalJobRef:
    """Kick off an eval run across the requested products/suites."""
    _validate_products(request, registry, body.products)
    suites = _parse_suites(body.suites)

    job_id = _make_job_id()
    _JOBS[job_id] = EvalJobStatus(
        job_id=job_id,
        status="running",
        products=list(body.products),
        suites=list(suites),
        started_at=datetime.now(UTC).isoformat(),
    )
    await HUB.start(job_id)
    task = asyncio.create_task(
        _run_eval_job(job_id=job_id, products=list(body.products), suites=suites, config=config),
        name=f"eval-{job_id}",
    )
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return EvalJobRef(job_id=job_id, status="running")


@router.get("/evals/jobs/{job_id}")
async def get_job(job_id: str) -> EvalJobStatus:
    status = _JOBS.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@router.get("/evals/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> EventSourceResponse:
    """Live stream while the job runs; replay terminal events if already done."""
    status = _JOBS.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="job not found")
    if HUB.is_live(job_id):
        return EventSourceResponse(_stream_events(job_id))

    async def replay() -> AsyncIterator[dict]:
        for run_id in status.run_ids:
            yield {"event": "product_done", "data": json.dumps({"run_id": run_id})}
        terminal = "error" if status.status == "failed" else "job_done"
        yield {
            "event": terminal,
            "data": json.dumps({"run_ids": status.run_ids, "error": status.error}),
        }

    return EventSourceResponse(replay())


@router.get("/evals/runs")
async def list_runs() -> dict:
    """Project every persisted run artifact into a summary (all products)."""
    summaries = [_summary_from_artifact(a) for a in _load_artifacts()]
    summaries.sort(key=lambda s: s.generated_at, reverse=True)
    return {"runs": summaries}


@router.get("/evals/runs/{run_id}")
async def get_run(run_id: str) -> EvalRunDetail:
    artifact = _load_artifact(run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="run not found")
    base = _summary_from_artifact(artifact)
    per_suite: dict[str, dict] = {}
    for suite in artifact.suites:
        raw = Path(suite.output_json)
        if raw.exists():
            try:
                per_suite[suite.suite] = json.loads(raw.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log.warning("could not read suite output %s", raw)
    return EvalRunDetail(
        **base.model_dump(),
        config_fingerprint=artifact.config_fingerprint,
        per_suite=per_suite,
    )


@router.get("/evals/corpus")
async def get_corpus() -> dict:
    """The product registry + which suites have authored data."""
    products = [
        ProductEvalInfo(
            product_id=p.product_id,
            language=p.language,
            repo_url=p.repo_url,
            has_retrieval=has_suite_data(p, "retrieval"),
            has_rag=has_suite_data(p, "rag"),
        )
        for p in PRODUCTS.values()
    ]
    return {"products": products}


# ---------------------------------------------------------------- helpers


async def _stream_events(job_id: str) -> AsyncIterator[dict]:
    q = await HUB.subscribe(job_id)
    try:
        while True:
            item = await q.get()
            if item == _JobHub._END:
                return
            assert isinstance(item, dict)
            yield {
                "event": item.get("event", "message"),
                "data": json.dumps(item.get("data", {})),
            }
    finally:
        await HUB.unsubscribe(job_id, q)


def _summary_from_artifact(artifact: EvalRunArtifact) -> EvalRunSummary:
    product_id = artifact.suites[0].product_id if artifact.suites else "unknown"
    return EvalRunSummary(
        run_id=artifact.run_id,
        product_id=product_id,
        generated_at=artifact.generated_at,
        passed=artifact.passed,
        suites=[_suite_summary(s) for s in artifact.suites],
    )


def _suite_summary(s: SuiteArtifact) -> SuiteSummary:
    return SuiteSummary(
        suite=s.suite,
        section=_SECTION.get(s.suite, s.suite),
        passed=s.passed,
        product_id=s.product_id,
        metrics=s.metrics,
        thresholds=s.thresholds,
        notes=s.notes,
    )


def _load_artifacts() -> list[EvalRunArtifact]:
    out: list[EvalRunArtifact] = []
    for path in sorted(DEFAULT_OUT_DIR.glob("*/summary.json")):
        artifact = _read_artifact(path)
        if artifact is not None:
            out.append(artifact)
    return out


def _load_artifact(run_id: str) -> EvalRunArtifact | None:
    path = DEFAULT_OUT_DIR / run_id / "summary.json"
    return _read_artifact(path) if path.exists() else None


def _read_artifact(path: Path) -> EvalRunArtifact | None:
    try:
        return EvalRunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("could not parse eval artifact %s", path)
        return None
