from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from anvay.ingest.reconcile import sweep_orphan_points
from anvay.registry import Registry

NOW = datetime(2026, 7, 6, 12, 0, 0, tzinfo=UTC)
OLD_TS = "2026-07-06T09:00:00+00:00"   # 3h before NOW — past any grace window
FRESH_TS = "2026-07-06T11:59:00+00:00"  # 1 min before NOW — inside grace


class FakeIndexer:
    _code = "anvay_code"
    _text = "anvay_text"

    def __init__(self, points: dict[str, list[tuple[str, dict]]]):
        # points keyed by vector_kind ("code"/"text")
        self.points = points
        self.deleted: list[str] = []

    async def iter_chunk_payloads(self, *, product_id: str, vector_kind: str, **_kw):
        for pid, payload in self.points.get(vector_kind, []):
            yield pid, payload

    async def delete_points_by_ids(self, point_ids):
        ids = [p for v in point_ids.values() for p in v]
        self.deleted.extend(ids)
        return len(ids)


def _registry(tmp_path: Path, chunk_ids: list[str]) -> Registry:
    reg = Registry(tmp_path / "registry.db")
    reg.upsert_resource_manifest(
        {
            "product": "demo",
            "sourceKey": "local:test",
            "resourceUri": "a.py",
            "contentHash": "h",
            "lastSeenSync": "s",
            "chunkIds": chunk_ids,
            "indexedAt": OLD_TS,
        }
    )
    return reg


def _payload(*, artifact_type="code", source_id="local:test", indexed_at=OLD_TS) -> dict:
    return {
        "artifact_type": artifact_type,
        "source_id": source_id,
        "indexed_at": indexed_at,
    }


@pytest.mark.asyncio
async def test_orphan_deleted_manifest_covered_kept(tmp_path: Path) -> None:
    registry = _registry(tmp_path, ["known-1"])
    indexer = FakeIndexer(
        {
            "code": [
                ("known-1", _payload()),
                ("orphan-1", _payload()),
            ]
        }
    )
    result = await sweep_orphan_points(
        registry=registry,
        indexer=indexer,
        product_id="demo",
        grace_minutes=60,
        dry_run=False,
        now=NOW,
    )
    assert result.orphan_ids == ["orphan-1"]
    assert indexer.deleted == ["orphan-1"]


@pytest.mark.asyncio
async def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    registry = _registry(tmp_path, [])
    indexer = FakeIndexer({"code": [("orphan-1", _payload())]})
    result = await sweep_orphan_points(
        registry=registry,
        indexer=indexer,
        product_id="demo",
        dry_run=True,
        now=NOW,
    )
    assert result.orphans == 1
    assert result.deleted == 0
    assert indexer.deleted == []


@pytest.mark.asyncio
async def test_fresh_orphans_and_exempt_artifacts_kept(tmp_path: Path) -> None:
    registry = _registry(tmp_path, [])
    indexer = FakeIndexer(
        {
            "code": [
                ("fresh", _payload(indexed_at=FRESH_TS)),
                ("no-ts", _payload(indexed_at="")),
                ("skill", _payload(source_id="skill:demo")),
                ("summary", _payload(artifact_type="summary")),
                ("community", _payload(artifact_type="graph_community_summary")),
                ("real-orphan", _payload()),
            ]
        }
    )
    result = await sweep_orphan_points(
        registry=registry,
        indexer=indexer,
        product_id="demo",
        grace_minutes=60,
        dry_run=False,
        now=NOW,
    )
    assert result.orphan_ids == ["real-orphan"]
    assert indexer.deleted == ["real-orphan"]
