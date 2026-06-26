"""Multi-product eval corpus registry.

Single source of truth mapping each evaluated product to its eval assets
(retrieval queries golden, answer-eval golden, ingest fixture). Plain config —
a frozen dataclass plus an in-memory dict, no new runtime deps. Everything is
keyed by ``product_id`` so the eval boundary is always product-scoped
(AGENTS.md invariant: product = root entity).

Datasets land under ``evals/products/<product_id>/`` as they are authored
(Phase 4). ``anvay`` reuses its existing ``tests/eval/queries.json``. A product
with ``None`` for a suite path simply has no data for that suite yet; the
dashboard renders a "needs dataset" state rather than faking scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SuiteName = Literal["retrieval", "rag", "code"]


@dataclass(frozen=True)
class ProductEval:
    product_id: str
    language: str
    repo_url: str  # onboarding/ingest reference
    queries_path: Path | None  # retrieval suite golden (queries.json); None until authored
    golden_path: Path | None  # answer-eval (rag) golden (jsonl); None until authored
    fixture_path: Path | None  # source to ingest for self-contained runs (None = live index)


PRODUCTS: dict[str, ProductEval] = {
    "anvay": ProductEval(
        product_id="anvay",
        language="python",
        repo_url="https://github.com/anvay/anvay",
        queries_path=Path("tests/eval/queries.json"),
        golden_path=None,
        fixture_path=Path("."),
    ),
    "guava": ProductEval(
        product_id="guava",
        language="java",
        repo_url="https://github.com/google/guava",
        queries_path=None,
        golden_path=None,
        fixture_path=None,
    ),
    "zod": ProductEval(
        product_id="zod",
        language="typescript",
        repo_url="https://github.com/colinhacks/zod",
        queries_path=None,
        golden_path=None,
        fixture_path=None,
    ),
    "cobra": ProductEval(
        product_id="cobra",
        language="go",
        repo_url="https://github.com/spf13/cobra",
        queries_path=None,
        golden_path=None,
        fixture_path=None,
    ),
}


def has_suite_data(p: ProductEval, suite: SuiteName) -> bool:
    """Whether ``p`` carries an authored dataset for ``suite``."""
    return {
        "retrieval": p.queries_path,
        "rag": p.golden_path,
        "code": p.golden_path,
    }.get(suite) is not None
