"""Unified golden-dataset schema + loader for Anvay evals.

One format, one loader. Replaces the old split between
``tests/eval/queries.json`` (file-match) and ``evals/golden.jsonl``
(answer-eval). Each product carries a single ``golden.jsonl`` whose lines are
:class:`GoldenItem` records:

```json
{"id": "...", "query": "natural-language dev question",
 "expected_files": ["anvay/retrieval/hybrid.py"],
 "expected_answer": "reference answer grounded in those files",
 "category": "architecture|how-to|debugging|conceptual|api",
 "complexity": "simple|medium|hard"}
```

``expected_files`` drives the deterministic IR metrics (recall/mrr/ndcg);
``expected_answer`` is the RAGAS reference for faithfulness/correctness/context
metrics. Both are required for a row to contribute to every metric — rows
missing one still score the metrics they can.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_CATEGORIES = ("architecture", "how-to", "debugging", "conceptual", "api")
VALID_COMPLEXITY = ("simple", "medium", "hard")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    query: str
    expected_files: list[str] = field(default_factory=list)
    expected_answer: str = ""
    category: str = "conceptual"
    complexity: str = "medium"

    @classmethod
    def from_dict(cls, d: dict) -> GoldenItem:
        return cls(
            id=str(d["id"]),
            query=str(d["query"]),
            expected_files=list(d.get("expected_files") or []),
            expected_answer=str(d.get("expected_answer") or ""),
            category=str(d.get("category") or "conceptual"),
            complexity=str(d.get("complexity") or "medium"),
        )


def load_golden(path: Path) -> list[GoldenItem]:
    """Load a product's golden jsonl. Blank lines and ``#`` comments are skipped."""
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(GoldenItem.from_dict(json.loads(line)))
    return items
