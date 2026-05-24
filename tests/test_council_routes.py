from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from nexus.api.app import app
from nexus.api.deps import get_proposal_queue
from nexus.council.queue import ProposalQueue
from nexus.council.runner import HUB


def test_council_stream_route_matches_before_session_detail(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "queue.db")
    queue.record_session(
        session_id="cs_done",
        product_id="p",
        topic="topic",
        proposal_id=None,
        deliberation=[],
        costs=[],
        started_at="2026-05-24T00:00:00Z",
        completed_at="2026-05-24T00:00:01Z",
    )
    app.dependency_overrides[get_proposal_queue] = lambda: queue
    try:
        client = TestClient(app)
        response = client.get("/council/sessions/cs_done/stream")
    finally:
        app.dependency_overrides.pop(get_proposal_queue, None)

    assert response.status_code == 200
    assert "event: session_start" in response.text


def test_council_stream_route_accepts_live_session_before_queue_record(
    tmp_path: Path,
) -> None:
    queue = ProposalQueue(tmp_path / "queue.db")
    asyncio.run(HUB.start("cs_live"))
    app.dependency_overrides[get_proposal_queue] = lambda: queue
    try:
        client = TestClient(app)
        with client.stream("GET", "/council/sessions/cs_live/stream") as response:
            assert response.status_code == 200
    finally:
        asyncio.run(HUB.finish("cs_live"))
        app.dependency_overrides.pop(get_proposal_queue, None)
