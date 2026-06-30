"""Multi-product eval corpus registry.

Single source of truth mapping each evaluated product to its unified golden
dataset (``evals/products/<product_id>/golden.jsonl``) and, for products that
must be ingested for a self-contained run, a Git ``repo_url`` to clone. Plain
config — a frozen dataclass plus an in-memory dict, no new runtime deps.
Everything is keyed by ``product_id`` so the eval boundary is always
product-scoped (AGENTS.md invariant: product = root entity).

``anvay`` is evaluated against the live ingested index (no clone). ``zod`` and
``guava`` are cloned + ingested on demand by the harness ingest helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PRODUCTS_DIR = Path("evals/products")
# Where the ingest helper clones source for products without a local checkout.
# Kept out of the repo; gitignored alongside other eval artifacts.
INGEST_CACHE_DIR = Path("artifacts/eval-corpus")


@dataclass(frozen=True)
class ProductEval:
    product_id: str
    language: str
    golden_path: Path  # unified golden dataset (jsonl)
    # Exactly one ingest source: a local directory (``source_path``) or a Git
    # ``repo_url`` cloned into INGEST_CACHE_DIR. Ingest always runs with the
    # graph layer enabled so the graph channel is exercised.
    source_path: Path | None = None
    repo_url: str = ""
    # Subdirectory of the checkout to ingest (keeps large upstream repos bounded
    # and the index focused on the area the golden set covers). "" = whole tree.
    ingest_subdir: str = ""

    def checkout_dir(self) -> Path:
        """Top-level directory cloned/used for this product."""
        if self.source_path is not None:
            return self.source_path
        return INGEST_CACHE_DIR / self.product_id

    def ingest_root(self) -> Path:
        """Directory whose contents are actually ingested."""
        root = self.checkout_dir()
        return root / self.ingest_subdir if self.ingest_subdir else root


PRODUCTS: dict[str, ProductEval] = {
    "anvay": ProductEval(
        product_id="anvay",
        language="python",
        golden_path=_PRODUCTS_DIR / "anvay" / "golden.jsonl",
        source_path=Path("anvay"),  # this codebase's package dir
    ),
    "zod": ProductEval(
        product_id="zod",
        language="typescript",
        golden_path=_PRODUCTS_DIR / "zod" / "golden.jsonl",
        repo_url="https://github.com/colinhacks/zod",
        ingest_subdir="packages/zod/src/v3",
    ),
    "guava": ProductEval(
        product_id="guava",
        language="java",
        golden_path=_PRODUCTS_DIR / "guava" / "golden.jsonl",
        repo_url="https://github.com/google/guava",
        ingest_subdir="guava/src/com/google/common/collect",
    ),
}
