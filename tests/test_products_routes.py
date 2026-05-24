"""Route-level tests for product creation."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nexus.api.app import app
from nexus.api.deps import get_registry
from nexus.registry import Registry


def test_create_product_accepts_owner_team(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.db")
    app.dependency_overrides[get_registry] = lambda: registry
    try:
        client = TestClient(app)
        r = client.post(
            "/products",
            json={
                "id": "payments-api",
                "name": "Payments API",
                "owner": {"team": "Payments Platform"},
            },
        )
    finally:
        app.dependency_overrides.pop(get_registry, None)

    assert r.status_code == 200
    body = r.json()
    assert body["owner"] == {"team": "Payments Platform"}

