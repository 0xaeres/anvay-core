"""Tests for the SQLite Registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus.registry import Registry


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry.db")


def test_registry_seed_defaults(registry: Registry) -> None:
    user = registry.get_user("admin")
    assert user is not None
    assert user["name"] == "Admin"
    assert user["role"] == "org_admin"


def test_registry_products(registry: Registry) -> None:
    product = {
        "id": "test-prod",
        "name": "Test Product",
        "tagline": "A product for testing",
        "owner": {"team": "QA", "lead": "Bob"},
        "onboardedAt": "2026-05-23T12:00:00Z",
    }
    registry.upsert_product(product)

    loaded = registry.get_product("test-prod")
    assert loaded is not None
    assert loaded["name"] == "Test Product"
    assert loaded["owner"] == {"team": "QA", "lead": "Bob"}

    prods = registry.list_products()
    assert len(prods) == 1
    assert prods[0]["id"] == "test-prod"


def test_registry_sources_id_and_name_lookup(registry: Registry) -> None:
    source = {
        "product": "test-prod",
        "name": "my-github-source",
        "type": "github",
        "config": {"token": "secret-token", "repos": ["a/b"]},
    }
    registry.upsert_source(source)

    # List sources
    sources = registry.list_sources("test-prod")
    assert len(sources) == 1

    stored = sources[0]
    sid = stored["id"]
    assert sid.startswith("src_")
    assert stored["name"] == "my-github-source"
    assert stored["type"] == "github"

    # Query by NAME
    by_name = registry.get_source("test-prod", "my-github-source")
    assert by_name is not None
    assert by_name["id"] == sid

    # Query by ID (this was the bug, now resolved!)
    by_id = registry.get_source("test-prod", sid)
    assert by_id is not None
    assert by_id["name"] == "my-github-source"

    # Delete by ID
    assert registry.delete_source("test-prod", sid) is True
    assert registry.get_source("test-prod", sid) is None
    assert registry.get_source("test-prod", "my-github-source") is None


def test_registry_source_encryption(registry: Registry, monkeypatch) -> None:
    from nexus.auth.token_cipher import TokenCipher

    key = TokenCipher.generate_key()
    monkeypatch.setenv("NEXUS_TOKEN_KEY", key)

    source = {
        "product": "test-prod",
        "name": "secure-source",
        "type": "github",
        "config": {"token": "my-ultra-secret-token", "repos": ["a/b"]},
    }
    registry.upsert_source(source)

    # Reading should automatically decrypt
    loaded = registry.get_source("test-prod", "secure-source")
    assert loaded is not None
    assert loaded["config"]["token"] == "my-ultra-secret-token"

    # Direct database verification to ensure it's encrypted at rest
    with registry._conn() as conn:
        row = conn.execute(
            "SELECT config_js FROM sources WHERE name = ?", ("secure-source",)
        ).fetchone()
        assert row is not None
        config_data = row["config_js"]
        assert "enc:" in config_data
        assert "my-ultra-secret-token" not in config_data
