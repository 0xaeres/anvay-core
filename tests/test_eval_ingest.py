"""Eval ingest wiring (offline).

Guards plan 1a: the eval ingest must run ``run_ingest`` on the **delta path**
(``registry`` + ``source_key`` set), because that is the only path that builds
the graph and writes chunk↔graph linkage into Qdrant payloads. Dropping either
arg silently kills the graph channel, so we assert they are passed.
"""

from __future__ import annotations

import pytest

import evals.ingest as eval_ingest
from anvay.ingest.pipeline import IngestStats


@pytest.mark.asyncio
async def test_ensure_ingested_runs_delta_path(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    async def _fake_run_ingest(**kwargs):
        captured.update(kwargs)
        return IngestStats()

    # Stub the heavy collaborators so the test stays offline.
    async def _noop_reset(product, *, config, registry):
        return None

    monkeypatch.setattr(eval_ingest, "run_ingest", _fake_run_ingest)
    monkeypatch.setattr(eval_ingest.Registry, "__init__", lambda self, path: None)
    monkeypatch.setattr(eval_ingest, "_already_indexed", lambda product, config: False)
    monkeypatch.setattr(eval_ingest, "_prepare_checkout", lambda product: None)
    monkeypatch.setattr(eval_ingest, "_reset_product_index", _noop_reset)

    root = tmp_path / "src"
    root.mkdir()

    class _Product:
        product_id = "anvay"
        source_path = root

        def ingest_root(self):
            return root

    class _Config:
        class storage:
            class proposal_queue:
                parent = tmp_path

    await eval_ingest.ensure_ingested(_Product(), config=_Config(), force=True)

    assert captured["enrich"] is False
    assert captured["registry"] is not None
    assert captured["source_key"] == "anvay:eval-local"
