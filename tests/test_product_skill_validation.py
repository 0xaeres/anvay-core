from __future__ import annotations

import pytest

from anvay.council.skill_catalog import OPTIONAL_PRODUCT_SKILL_SECTIONS
from anvay.council.skill_evals import evaluate_skill_draft
from anvay.council.skill_parser import required_sections_for_tier, validate_skill_markdown
from anvay.council.state import EvidenceChunk, SkillDraft

_ENUMERABLE = {
    "capabilities and workflows",
    "system map",
    "data model",
    "interfaces and contracts",
    "invariants and constraints",
    "security and secrets",
    "known traps",
    "freshness and evidence",
    "product language",
}


def _body(*, omit: str | None = None, extra: str = "", file: str = "a.py", line: int = 1) -> str:
    lines = ["# product-skill", ""]
    for title in required_sections_for_tier("product_master"):
        if title == omit:
            continue
        lines.append(f"## {title}")
        if title == "Use This Skill When":
            lines.append("Use for product orientation and grounded development.")
        elif title == "How To Use The Knowledge Base":
            lines.append("Treat KB/RAG as source of truth for fresh lookup.")
        elif title == "How To Work In This Product":
            lines.append("Check local conventions before editing.")
        elif title.lower() in _ENUMERABLE:
            lines.append(f"- Evidence for {title.lower()} item one [file: {file}:{line}].")
            lines.append(f"- Evidence for {title.lower()} item two [file: {file}:{line}].")
        else:
            lines.append(f"Evidence supports {title.lower()} [file: {file}:{line}].")
        lines.append("")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def test_product_skill_required_headings_pass() -> None:
    report = validate_skill_markdown(_body(), tier="product_master")
    assert report.is_complete


def test_product_skill_missing_or_empty_required_headings_fail() -> None:
    missing = validate_skill_markdown(
        _body(omit="Product Snapshot"),
        tier="product_master",
    )
    assert "Product Snapshot" in missing.missing_sections

    empty = validate_skill_markdown(
        _body().replace(
            "## Product Snapshot\nEvidence supports product snapshot [file: a.py:1].",
            "## Product Snapshot\n",
        ),
        tier="product_master",
    )
    assert any("Product Snapshot" in item for item in empty.short_sections)


def test_product_skill_optional_headings_allowed_with_evidence() -> None:
    optional = "\n".join(
        f"## {title}\nEvidence supports {title.lower()} [file: a.py:1].\n"
        for title in OPTIONAL_PRODUCT_SKILL_SECTIONS
    )
    report = validate_skill_markdown(_body(extra=optional), tier="product_master")
    assert report.is_complete


def test_product_skill_unknown_heading_fails() -> None:
    report = validate_skill_markdown(
        _body(extra="## Extra Notes\nNo."),
        tier="product_master",
    )
    assert any("unexpected sections" in item for item in report.short_sections)


@pytest.mark.asyncio
async def test_product_skill_fabricated_citation_fails() -> None:
    evidence = [
        EvidenceChunk(chunk_id="c1", file="a.py", line=1, score=0.9, excerpt="facts")
    ]
    draft = SkillDraft(
        name="product-skill",
        description="Use for product orientation and grounded development.",
        tier="product_master",
        body=_body(file="missing.py", line=99),
    )

    result = await evaluate_skill_draft(draft=draft, evidence=evidence, plan=[], chat=object())

    assert result.status == "failed"
    assert any("not in retrieved evidence" in failure for failure in result.failures)
