"""Self-contained corpus ingest for evals.

Ensures each product's index exists before it is evaluated. ``anvay`` ingests
this repo's package dir; ``zod``/``guava`` are shallow-cloned into
``INGEST_CACHE_DIR`` and a bounded subdir is ingested. Ingest runs with
``enrich=False``: a 3-product eval (run 20260629, n=5) showed HQE consistently
**degrades** quality here — ``context_precision`` dropped on every product and
``context_recall`` was flat-to-down — so it is off. The deterministic **FalkorDB
graph** still builds (gated by ``config.ingestion.graph``, not ``enrich``).

Idempotent: a product whose Qdrant code collection already has points is left
alone unless ``force=True``.
"""

from __future__ import annotations

import logging

from anvay.config import AnvayConfig
from anvay.connectors.local_fs import LocalFsConfig, LocalFsSource
from anvay.ingest.pipeline import IngestStats, run_ingest
from anvay.registry import Registry
from evals.corpus import ProductEval

log = logging.getLogger(__name__)


async def ensure_ingested(
    product: ProductEval,
    *,
    config: AnvayConfig,
    force: bool = False,
) -> IngestStats | None:
    """Ingest ``product`` if its index is empty (or ``force``). Returns stats
    when an ingest ran, else ``None``. Runs the delta path (registry + source_key)
    so the graph builds and chunk↔graph linkage lands in Qdrant payloads."""
    if not force and _already_indexed(product, config):
        log.info("product %s already indexed; skipping ingest", product.product_id)
        return None

    _prepare_checkout(product)
    root = product.ingest_root()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"ingest root not found for {product.product_id}: {root}")

    log.info("ingesting %s from %s (enrich+graph)", product.product_id, root)
    source = LocalFsSource(LocalFsConfig(root=root))
    # A registry + source_key are what flip ``run_ingest`` onto the delta path,
    # which is the **only** path that extracts the graph and writes chunk↔graph
    # linkage (``graph_node_ids``/``entity_ids``) into Qdrant payloads. Without
    # them ``delta_enabled`` is False and the graph channel is dead (plan 1a). We
    # reuse the production registry DB under an eval-scoped source_key so the eval
    # exercises exactly the production graph path.
    registry = Registry(config.storage.proposal_queue.parent / "registry.db")
    if force:
        # Delta tracks source-file content hashes, so a plain re-ingest skips
        # unchanged files even under ``force`` — leaving stale chunks from a prior
        # config (e.g. HQE off). For a *clean* forced re-ingest, wipe this
        # product's points + this source_key's manifest so every resource is
        # re-embedded fresh under the current settings.
        await _reset_product_index(product, config=config, registry=registry)
    return await run_ingest(
        product_id=product.product_id,
        source=source,
        config=config,
        enrich=False,
        registry=registry,
        source_key=_eval_source_key(product),
    )


def _eval_source_key(product: ProductEval) -> str:
    """Stable eval-scoped source key; isolates eval manifests from real sources."""
    return f"{product.product_id}:eval-local"


async def _reset_product_index(
    product: ProductEval, *, config: AnvayConfig, registry: Registry
) -> None:
    """Wipe a product's vector points + its eval-source manifest so a forced
    re-ingest re-embeds everything fresh (no stale chunks, no duplicates)."""
    from anvay.ingest.indexer_factory import create_indexer

    source_key = _eval_source_key(product)
    for row in registry.list_resource_manifests(product.product_id, source_key):
        registry.delete_resource_manifest(product.product_id, source_key, row["resourceUri"])

    indexer = create_indexer(config)
    try:
        deleted = await indexer.delete_by_product(product_id=product.product_id)
        log.info("force reset %s: cleared points %s", product.product_id, deleted)
    finally:
        await indexer.aclose()


def _already_indexed(product: ProductEval, config: AnvayConfig) -> bool:
    """Collections are shared across products; isolation is by ``product_id``
    payload filter (AGENTS.md invariant). Count this product's points."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    collection = config.vector_store.collections.code
    client = QdrantClient(url=config.vector_store.url)
    flt = Filter(
        must=[FieldCondition(key="product_id", match=MatchValue(value=product.product_id))]
    )
    try:
        count = client.count(collection_name=collection, count_filter=flt, exact=True)
        return (count.count or 0) > 0
    except Exception:
        return False
    finally:
        client.close()


def _prepare_checkout(product: ProductEval) -> None:
    """Clone the upstream repo for products that need it. Local-source products
    (``source_path`` set) are used in place."""
    if product.source_path is not None:
        return
    if not product.repo_url:
        raise ValueError(f"product {product.product_id} has neither source_path nor repo_url")

    dest = product.checkout_dir()
    if dest.exists():
        return
    import git  # gitpython is already a runtime dep

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s -> %s (shallow)", product.repo_url, dest)
    git.Repo.clone_from(product.repo_url, dest, depth=1)
