"""Reverse reconciliation: Qdrant → manifest.

`run_ingest` already repairs the forward direction (manifest rows whose points
vanished from Qdrant get re-indexed). This module handles the reverse: points
left in Qdrant with no manifest row claiming them — the residue of a crash
between an upsert and the manifest commit.

Safety posture (this sweep deletes data, so it is deliberately conservative):

- only raw resource chunks are candidates (`artifact_type` in {code, doc});
  skill chunks (`source_id` = "skill:*") and synthetic summary artifacts are
  always exempt,
- points younger than the grace window are kept (they may belong to an
  in-flight batch whose manifest write hasn't landed yet),
- dry-run by default: candidates are logged, nothing is deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

log = logging.getLogger(__name__)

_SWEEPABLE_ARTIFACT_TYPES = {"code", "doc"}


@dataclass
class SweepResult:
    scanned: int = 0
    orphans: int = 0
    deleted: int = 0
    dry_run: bool = True
    orphan_ids: list[str] = field(default_factory=list)


async def sweep_orphan_points(
    *,
    registry,
    indexer,
    product_id: str,
    grace_minutes: int = 60,
    dry_run: bool = True,
    now: datetime | None = None,
) -> SweepResult:
    known_ids = registry.list_manifest_chunk_ids(product_id)
    cutoff = (now or datetime.now(UTC)) - timedelta(minutes=grace_minutes)
    result = SweepResult(dry_run=dry_run)

    orphans_by_kind: dict[str, list[str]] = {}
    for vector_kind in ("code", "text"):
        async for point_id, payload in indexer.iter_chunk_payloads(
            product_id=product_id, vector_kind=vector_kind
        ):
            result.scanned += 1
            if point_id in known_ids:
                continue
            artifact_type = payload.get("artifact_type") or payload.get("kind") or ""
            if artifact_type not in _SWEEPABLE_ARTIFACT_TYPES:
                continue
            source_id = payload.get("source_id") or ""
            if source_id.startswith("skill:"):
                continue
            indexed_at = _parse_ts(payload.get("indexed_at"))
            if indexed_at is None or indexed_at >= cutoff:
                # Unknown age or too fresh — might be an in-flight batch.
                continue
            orphans_by_kind.setdefault(vector_kind, []).append(point_id)
            result.orphan_ids.append(point_id)

    result.orphans = len(result.orphan_ids)
    if not result.orphans:
        return result

    if dry_run:
        log.info(
            "orphan sweep (dry-run) product=%s: %d orphan point(s) would be deleted: %s",
            product_id,
            result.orphans,
            result.orphan_ids[:20],
        )
        return result

    buckets = {
        (indexer._code if kind == "code" else indexer._text): ids
        for kind, ids in orphans_by_kind.items()
    }
    result.deleted = await indexer.delete_points_by_ids(buckets)
    log.info(
        "orphan sweep product=%s: deleted %d orphan point(s)",
        product_id,
        result.deleted,
    )
    return result


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts
