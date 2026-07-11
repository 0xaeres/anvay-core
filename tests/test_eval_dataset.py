"""Unit tests for the unified eval dataset loader + registry."""

from __future__ import annotations

from pathlib import Path

from evals.corpus import PRODUCTS
from evals.dataset import GoldenItem, load_golden


def test_golden_item_from_dict_defaults() -> None:
    item = GoldenItem.from_dict({"id": "x", "query": "q"})
    assert item.id == "x"
    assert item.query == "q"
    assert item.expected_files == []
    assert item.expected_answer == ""
    assert item.category == "conceptual"
    assert item.complexity == "medium"


def test_golden_item_from_dict_full() -> None:
    item = GoldenItem.from_dict(
        {
            "id": "y",
            "query": "how?",
            "expected_files": ["a.py", "b.py"],
            "expected_answer": "because",
            "category": "how-to",
            "complexity": "hard",
        }
    )
    assert item.expected_files == ["a.py", "b.py"]
    assert item.category == "how-to"
    assert item.complexity == "hard"


def test_registry_golden_files_exist_and_load() -> None:
    for product in PRODUCTS.values():
        path = Path(product.golden_path)
        assert path.exists(), f"missing golden for {product.product_id}: {path}"
        items = load_golden(path)
        assert items, f"empty golden for {product.product_id}"
        # ids are unique and every item carries a query + reference answer
        ids = [it.id for it in items]
        assert len(ids) == len(set(ids)), f"duplicate ids in {product.product_id}"
        for it in items:
            assert it.query and it.expected_answer and it.expected_files


def test_ingest_root_respects_subdir() -> None:
    zod = PRODUCTS["zod"]
    assert zod.ingest_subdir
    assert str(zod.ingest_root()).endswith(zod.ingest_subdir)
