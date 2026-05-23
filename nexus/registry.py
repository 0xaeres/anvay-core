"""Tiny SQLite registry for products + users.

Phase-4 minimum viable surface for the API. RBAC + multi-tenant onboarding
arrives in later slices; for now we seed a default product + user on first boot
so the UI has something to render.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from nexus.auth.token_cipher import TokenCipher, TokenCipherError

log = logging.getLogger(__name__)

# Keys whose values are encrypted at rest in source config blobs.
_SECRET_KEY_HINTS = ("token", "api_key", "password", "secret")


def _get_cipher() -> TokenCipher | None:
    key = os.environ.get("NEXUS_TOKEN_KEY")
    if not key:
        log.warning(
            "NEXUS_TOKEN_KEY is not set — connector secrets will be stored in "
            "plaintext. Set this variable to enable encryption at rest."
        )
        return None
    try:
        return TokenCipher(key)
    except TokenCipherError:
        log.exception("NEXUS_TOKEN_KEY is set but invalid; connector secrets unencrypted")
        return None


def _encrypt_config(config: dict, cipher: TokenCipher | None) -> dict:
    if not cipher:
        return config
    out: dict = {}
    for k, v in config.items():
        if isinstance(v, str) and v and any(s in k.lower() for s in _SECRET_KEY_HINTS):
            out[k] = "enc:" + cipher.encrypt(v)
        else:
            out[k] = v
    return out


def _decrypt_config(config: dict, cipher: TokenCipher | None) -> dict:
    out: dict = {}
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("enc:"):
            if cipher:
                try:
                    out[k] = cipher.decrypt(v[4:])
                except TokenCipherError:
                    log.error("Failed to decrypt config key %r — returning redacted value", k)
                    out[k] = ""
            else:
                # Key was encrypted but NEXUS_TOKEN_KEY is now missing.
                log.warning(
                    "Encrypted value for %r but NEXUS_TOKEN_KEY not set; returning empty", k
                )
                out[k] = ""
        else:
            out[k] = v
    return out


_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    tagline         TEXT NOT NULL DEFAULT '',
    owner_js        TEXT NOT NULL DEFAULT '{}',
    onboarded_at    TEXT NOT NULL,
    master_skill_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,
    products_js TEXT NOT NULL DEFAULT '[]'
);

-- Runtime-added sources. Merged with nexus.yaml connectors at read time.
CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    product_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'connected',
    config_js   TEXT NOT NULL DEFAULT '{}',
    last_sync   TEXT,
    resource_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_product
    ON sources(product_id);
"""


class Registry:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        self._seed_defaults()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _seed_defaults(self) -> None:
        with self._conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if n == 0:
                conn.execute(
                    """INSERT INTO users (id, name, role, products_js)
                       VALUES (?,?,?,?)""",
                    ("admin", "Admin", "org_admin", json.dumps([])),
                )

    # ------------------------------------------------------------ products

    def list_products(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM products ORDER BY name").fetchall()
        return [_row_to_product(r) for r in rows]

    def get_product(self, product_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        return _row_to_product(row) if row else None

    def upsert_product(self, product: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO products
                   (id, name, tagline, owner_js, onboarded_at, master_skill_id)
                   VALUES (?,?,?,?,?,?)""",
                (
                    product["id"],
                    product["name"],
                    product.get("tagline", ""),
                    json.dumps(product.get("owner", {})),
                    product["onboardedAt"],
                    product.get("masterSkillId", ""),
                ),
            )

    # ------------------------------------------------------------ users

    def get_user(self, user_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["products"] = json.loads(d.pop("products_js"))
        return d

    def list_users(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["products"] = json.loads(d.pop("products_js"))
            out.append(d)
        return out


def _row_to_product(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["owner"] = json.loads(d.pop("owner_js"))
    d["onboardedAt"] = d.pop("onboarded_at")
    d["masterSkillId"] = d.pop("master_skill_id")
    return d


# ---------------------------------------------------------------- sources


def _row_to_source(row: sqlite3.Row) -> dict:
    d = dict(row)
    raw_config = json.loads(d.pop("config_js") or "{}")
    d["config"] = _decrypt_config(raw_config, _get_cipher())
    d["lastSync"] = d.pop("last_sync", None)
    d["resourceCount"] = d.pop("resource_count", 0)
    d["product"] = d.pop("product_id")
    return d


def add_source_methods(cls):
    """Mix-in style: add source helpers to Registry without re-declaring it."""

    def list_sources(self, product_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE product_id = ? ORDER BY created_at DESC",
                (product_id,),
            ).fetchall()
        return [_row_to_source(r) for r in rows]

    def get_source(self, product_id: str, source_id_or_name: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE product_id = ? AND (id = ? OR name = ?)",
                (product_id, source_id_or_name, source_id_or_name),
            ).fetchone()
        return _row_to_source(row) if row else None

    def upsert_source(self, source: dict) -> None:
        import uuid as _uuid
        from datetime import UTC as _UTC
        from datetime import datetime as _dt

        sid = source.get("id") or f"src_{_uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sources
                   (id, product_id, name, type, status, config_js, last_sync,
                    resource_count, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    source["product"],
                    source["name"],
                    source["type"],
                    source.get("status", "connected"),
                    json.dumps(_encrypt_config(source.get("config", {}), _get_cipher())),
                    source.get("lastSync"),
                    int(source.get("resourceCount", 0)),
                    source.get("createdAt") or _dt.now(_UTC).isoformat(),
                ),
            )

    def delete_source(self, product_id: str, source_id_or_name: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM sources WHERE product_id = ? AND (id = ? OR name = ?)",
                (product_id, source_id_or_name, source_id_or_name),
            )
            return cur.rowcount > 0

    cls.list_sources = list_sources
    cls.get_source = get_source
    cls.upsert_source = upsert_source
    cls.delete_source = delete_source
    return cls


Registry = add_source_methods(Registry)
