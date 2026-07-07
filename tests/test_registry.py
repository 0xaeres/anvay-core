"""Tests for the SQLite Registry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from anvay.registry import Registry


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry.db")


def test_registry_seed_defaults(registry: Registry) -> None:
    user = registry.get_user("admin")
    assert user is not None
    assert user["name"] == "Admin"
    assert user["role"] == "admin"


def test_registry_migrates_legacy_user_roles(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    registry = Registry(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO users (id, name, role, products_js) VALUES (?,?,?,?)",
            ("u1", "Legacy", "org_admin", "[]"),
        )

    registry = Registry(db)

    assert registry.get_user("u1")["role"] == "admin"
    with sqlite3.connect(db) as conn:
        role = conn.execute("SELECT role FROM users WHERE id = ?", ("u1",)).fetchone()[0]
    assert role == "admin"


def test_registry_backfills_legacy_user_products(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    registry = Registry(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO users (id, name, role, products_js) VALUES (?,?,?,?)",
            ("u1", "User", "viewer", '["prod-a"]'),
        )

    registry = Registry(db)

    assert registry.list_product_ids_for_user("u1") == ["prod-a"]
    assert registry.list_product_memberships("u1") == {"prod-a": "owner"}


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
        "name": "my-filesystem-source",
        "type": "filesystem",
        "config": {"roots": ["/tmp/code"]},
    }
    registry.upsert_source(source)

    sources = registry.list_sources("test-prod")
    assert len(sources) == 1

    stored = sources[0]
    sid = stored["id"]
    assert sid.startswith("src_")
    assert stored["name"] == "my-filesystem-source"
    assert stored["type"] == "filesystem"

    by_name = registry.get_source("test-prod", "my-filesystem-source")
    assert by_name is not None
    assert by_name["id"] == sid

    by_id = registry.get_source("test-prod", sid)
    assert by_id is not None
    assert by_id["name"] == "my-filesystem-source"

    assert registry.delete_source("test-prod", sid) is True
    assert registry.get_source("test-prod", sid) is None
    assert registry.get_source("test-prod", "my-filesystem-source") is None


def test_registry_resource_manifest_roundtrip(registry: Registry) -> None:
    registry.upsert_resource_manifest(
        {
            "product": "p",
            "sourceKey": "src",
            "resourceUri": "file.py",
            "contentHash": "abc",
            "mime": "text/x-python",
            "sizeBytes": 123,
            "lastSeenSync": "sync-1",
            "chunkIds": ["c1", "c2"],
            "indexedAt": "now",
            "embeddingVersion": "v1",
        }
    )

    row = registry.get_resource_manifest("p", "src", "file.py")
    assert row is not None
    assert row["contentHash"] == "abc"
    assert row["chunkIds"] == ["c1", "c2"]
    assert registry.list_resource_manifests("p", "src")[0]["resourceUri"] == "file.py"
    assert registry.delete_resource_manifest("p", "src", "file.py") is True
    assert registry.get_resource_manifest("p", "src", "file.py") is None


def test_registry_refuses_plaintext_source_secrets(registry: Registry, monkeypatch) -> None:
    from anvay.auth.token_cipher import TokenCipherError

    monkeypatch.delenv("ANVAY_TOKEN_KEY", raising=False)

    with pytest.raises(TokenCipherError, match="ANVAY_TOKEN_KEY is required"):
        registry.upsert_source(
            {
                "product": "test-prod",
                "name": "insecure-source",
                "type": "github",
                "config": {"token": "secret-token", "repos": ["a/b"]},
            }
        )


def test_registry_source_encryption(registry: Registry, monkeypatch) -> None:
    from anvay.auth.token_cipher import TokenCipher

    key = TokenCipher.generate_key()
    monkeypatch.setenv("ANVAY_TOKEN_KEY", key)

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


def test_index_status_columns_added_to_legacy_db(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE source_resources (
            product_id TEXT NOT NULL, source_key TEXT NOT NULL,
            resource_uri TEXT NOT NULL, content_hash TEXT NOT NULL,
            mime TEXT NOT NULL DEFAULT '', size_bytes INTEGER,
            last_seen_sync TEXT NOT NULL, chunk_ids_js TEXT NOT NULL DEFAULT '[]',
            indexed_at TEXT NOT NULL, embedding_version TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (product_id, source_key, resource_uri)
        )"""
    )
    conn.commit()
    conn.close()

    reg = Registry(db)
    with reg._conn() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(source_resources)")}
    assert "index_status" in cols
    assert "index_status_at" in cols


def test_index_status_roundtrip_and_stuck_query(registry: Registry) -> None:
    row = {
        "product": "demo",
        "sourceKey": "local:test",
        "resourceUri": "a.py",
        "contentHash": "h1",
        "lastSeenSync": "s1",
        "indexedAt": "2026-07-06T00:00:00+00:00",
        "indexStatus": "pending",
        "indexStatusAt": "2026-07-06T00:00:00+00:00",
    }
    registry.upsert_resource_manifest(row)

    got = registry.get_resource_manifest("demo", "local:test", "a.py")
    assert got["indexStatus"] == "pending"

    stuck = registry.list_stuck_index_pending(
        "demo", older_than_iso="2026-07-06T01:00:00+00:00"
    )
    assert [r["resourceUri"] for r in stuck] == ["a.py"]

    # Not stuck when cutoff is before the pending mark.
    assert (
        registry.list_stuck_index_pending(
            "demo", older_than_iso="2026-07-05T00:00:00+00:00"
        )
        == []
    )

    registry.update_resource_index_status(
        "demo", "local:test", "a.py", status="indexed", at="2026-07-06T02:00:00+00:00"
    )
    assert (
        registry.list_stuck_index_pending(
            "demo", older_than_iso="2026-07-07T00:00:00+00:00"
        )
        == []
    )


def test_mark_index_pending_inserts_stub_for_new_resource(registry: Registry) -> None:
    registry.mark_resource_index_pending(
        "demo", "local:test", "new.py", at="2026-07-06T00:00:00+00:00"
    )
    got = registry.get_resource_manifest("demo", "local:test", "new.py")
    assert got is not None
    assert got["indexStatus"] == "pending"

    # Pre-migration rows (index_status = '') are never treated as stuck.
    registry.upsert_resource_manifest(
        {
            "product": "demo",
            "sourceKey": "local:test",
            "resourceUri": "old.py",
            "contentHash": "h",
            "lastSeenSync": "s",
            "indexedAt": "2020-01-01T00:00:00+00:00",
        }
    )
    stuck = registry.list_stuck_index_pending(
        "demo", older_than_iso="2027-01-01T00:00:00+00:00"
    )
    assert [r["resourceUri"] for r in stuck] == ["new.py"]
