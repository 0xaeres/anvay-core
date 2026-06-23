from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import anvay.api.app as api_app
from anvay.api.app import app
from anvay.api.deps import get_auth_store
from anvay.api.routes.metrics import WebVitalsMetric
from anvay.auth.store import AuthStore


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("ANVAY_SECRET_KEY", "test-secret")
    store = AuthStore(tmp_path / "auth.db", secret_key="test-secret")
    store.create_user(
        email="admin@example.com",
        password="correct horse battery staple",
        role="admin",
    )
    monkeypatch.setattr(api_app, "get_auth_store", lambda: store)
    app.dependency_overrides[get_auth_store] = lambda: store
    client = TestClient(app, base_url="https://testserver")
    res = client.post(
        "/auth/login",
        json={
            "email": "admin@example.com",
            "password": "correct horse battery staple",
        },
    )
    assert res.status_code == 200
    return client


def test_web_vitals_accepts_authed_metric_without_csrf(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    try:
        res = client.post(
            "/metrics/web-vitals",
            json={
                "name": "LCP",
                "value": 1234.5,
                "rating": "good",
                "id": "v1-123",
                "route": "/p/anvay/dashboard?secret=drop",
                "product_id": "anvay",
                "navigation_type": "navigate",
            },
        )
    finally:
        app.dependency_overrides.pop(get_auth_store, None)

    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_web_vitals_metric_strips_query_strings() -> None:
    metric = WebVitalsMetric.model_validate(
        {
            "name": "LCP",
            "value": 1234.5,
            "id": "v1-123",
            "route": "https://example.com/p/anvay/dashboard?token=secret#frag",
        }
    )

    assert metric.route == "/p/anvay/dashboard"


def test_web_vitals_rejects_invalid_metric(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    try:
        res = client.post(
            "/metrics/web-vitals",
            json={
                "name": "UNKNOWN",
                "value": 1,
                "id": "v1-123",
                "route": "/login",
            },
        )
    finally:
        app.dependency_overrides.pop(get_auth_store, None)

    assert res.status_code == 422


def test_web_vitals_rejects_oversized_payload(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    try:
        res = client.post(
            "/metrics/web-vitals",
            json={
                "name": "CLS",
                "value": 0,
                "id": "x" * 129,
                "route": "/login",
            },
        )
    finally:
        app.dependency_overrides.pop(get_auth_store, None)

    assert res.status_code == 422
