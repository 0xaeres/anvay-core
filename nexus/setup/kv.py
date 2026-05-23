"""Setup key-value store — runtime config overrides written by the setup wizard.

Lives in the same SQLite file as the rest of the registry so backups stay
coherent. Resolution at boot is `SetupKV.get("skills_repo")` → `nexus.yaml`
fallback → empty (setup required).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS setup_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SetupKV:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def get(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM setup_kv WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO setup_kv (key, value) VALUES (?, ?)",
                (key, value),
            )

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM setup_kv WHERE key = ?", (key,))
