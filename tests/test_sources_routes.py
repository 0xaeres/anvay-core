"""Tests for product-scoped source routes and GitHub multi-repo sync."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from nexus.api.app import app
from nexus.api.deps import get_config_dep, get_registry
from nexus.api.routes import sources
from nexus.auth.token_cipher import TokenCipher
from nexus.config import NexusConfig
from nexus.ingest.pipeline import IngestStats
from nexus.registry import Registry


def _config(tmp_path: Path) -> NexusConfig:
    return NexusConfig(
        models={
            "council": {"provider": "test", "model": "test"},
            "light": {"provider": "test", "model": "test"},
            "embedding": {"provider": "test", "model": "test", "url": "http://embed"},
            "reranker": {"provider": "test", "model": "test", "url": "http://rerank"},
        },
        storage={
            "proposal_queue": tmp_path / "proposals.db",
            "council_checkpoint": tmp_path / "council.sqlite",
        },
    )


def test_add_source_refuses_plaintext_secret_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_TOKEN_KEY", raising=False)
    registry = Registry(tmp_path / "registry.db")
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_config_dep] = lambda: _config(tmp_path)
    try:
        client = TestClient(app)
        r = client.post(
            "/products/demo/sources",
            json={
                "name": "github",
                "type": "github",
                "config": {
                    "token": "ghp_secret",
                    "repos": ["https://github.com/acme/api"],
                },
            },
        )
    finally:
        app.dependency_overrides.pop(get_registry, None)
        app.dependency_overrides.pop(get_config_dep, None)

    assert r.status_code == 400
    assert "NEXUS_TOKEN_KEY is required" in r.json()["detail"]


def test_github_repo_urls_validate_all_before_clone() -> None:
    source = {
        "config": {
            "repos": [
                "https://github.com/acme/api",
                "not-a-github-url",
            ]
        }
    }

    try:
        sources._github_repo_urls(source)
    except ValueError as e:
        assert "not-a-github-url" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected invalid repo URL to fail")


def test_github_sync_clones_all_repos_and_aggregates_count(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NEXUS_TOKEN_KEY", TokenCipher.generate_key())
    registry = Registry(tmp_path / "registry.db")
    cfg = _config(tmp_path)
    runtime = {
        "product": "demo",
        "name": "github",
        "type": "github",
        "status": "connected",
        "config": {
            "token": "ghp_secret",
            "repos": [
                "https://github.com/acme/api",
                "git@github.com:acme/web",
            ],
        },
        "resourceCount": 0,
    }
    registry.upsert_source(runtime)
    runtime = registry.get_source("demo", "github")
    assert runtime is not None

    cloned: list[str] = []

    async def fake_clone(url, token, q, index, total):
        cloned.append(url)
        root = tmp_path / f"repo-{index}"
        root.mkdir()
        return root, root

    async def fake_ingest_root(*, root_label, **kwargs):
        count = 2 if root_label.endswith("/api") else 3
        return IngestStats(resources_seen=count, resources_indexed=count), None

    monkeypatch.setattr(sources, "_clone_github_repo", fake_clone)
    monkeypatch.setattr(sources, "_ingest_root", fake_ingest_root)

    asyncio.run(
        sources._sync_source_contents(
            product_id="demo",
            source=runtime,
            runtime=runtime,
            config=cfg,
            registry=registry,
            q=asyncio.Queue(),
        )
    )

    assert cloned == ["https://github.com/acme/api", "git@github.com:acme/web"]
    updated = registry.get_source("demo", "github")
    assert updated is not None
    assert updated["resourceCount"] == 5
