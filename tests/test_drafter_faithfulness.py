"""Markdown skill parser + completeness validator tests (no LLM calls)."""

from nexus.council.skill_parser import (
    _normalise_name,  # private but stable for tests
    parse_skill_markdown,
    strip_uncited_rules,
    validate_completeness,
)
from nexus.council.state import EvidenceChunk


def _evi() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(chunk_id="c1", file="a.rs", line=10, score=0.9, excerpt="x"),
        EvidenceChunk(chunk_id="c2", file="b.rs", line=20, score=0.8, excerpt="y"),
    ]


# ---- strip_uncited_rules ---------------------------------------------------


def test_strip_keeps_cited_rule_items() -> None:
    body = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Rules\n\n"
        "1. Cited rule [file: a.rs:10].\n"
        "2. Another cited rule [file: b.rs:20].\n"
    )
    out, dropped = strip_uncited_rules(body)
    assert dropped == 0
    assert out == body


def test_strip_removes_uncited_rule_items() -> None:
    body = (
        "# Title\n\n"
        "## Rules\n\n"
        "1. Cited rule [file: a.rs:10].\n"
        "2. Uncited claim without anchor.\n"
        "3. Another cited rule [file: b.rs:20].\n"
    )
    out, dropped = strip_uncited_rules(body)
    assert dropped == 1
    assert "Uncited claim" not in out
    assert "Cited rule" in out


def test_strip_leaves_prose_outside_rules_alone() -> None:
    body = (
        "# Title\n\nFree-form intro with no citation. Still fine.\n\n"
        "## Rules\n\n"
        "1. Cited [file: a.rs:10].\n\n"
        "## Anti-patterns\n\n"
        "- An uncited anti-pattern, allowed in this section.\n"
    )
    out, dropped = strip_uncited_rules(body)
    assert dropped == 0
    assert "Free-form intro" in out
    assert "uncited anti-pattern" in out


# ---- _normalise_name --------------------------------------------------------


def test_normalise_name_kebab_cases_and_caps_length() -> None:
    assert _normalise_name("PDA Seed Validation") == "pda-seed-validation"
    assert _normalise_name("snake_case_input") == "snake-case-input"
    assert _normalise_name("--leading--and--trailing--") == "leading-and-trailing"
    long_name = "x" * 200
    assert len(_normalise_name(long_name)) <= 60


# ---- parse_skill_markdown ---------------------------------------------------


def test_parse_extracts_name_body_and_citations() -> None:
    md = (
        "# auth-token-rotation\n\n"
        "Intro.\n\n"
        "## Rules\n"
        "1. Do X [file: a.rs:10].\n"
        "2. Do Y [file: b.rs:20].\n"
    )
    parsed = parse_skill_markdown(md, evidence=_evi())
    assert parsed.name == "auth-token-rotation"
    assert "## Rules" in parsed.body
    assert {c.file for c in parsed.citations} == {"a.rs", "b.rs"}
    # excerpt populated from evidence pool when anchor matches
    assert any(c.excerpt for c in parsed.citations)


def test_parse_falls_back_when_no_h1() -> None:
    parsed = parse_skill_markdown("body only", fallback_name="My Topic")
    assert parsed.name == "my-topic"


def test_parse_dedupes_citations() -> None:
    md = "# t\n\n[file: a.rs:10] mentioned twice [file: a.rs:10]."
    parsed = parse_skill_markdown(md, evidence=_evi())
    assert len(parsed.citations) == 1


# ---- validate_completeness --------------------------------------------------


def test_complete_skill_reports_complete() -> None:
    md = (
        "# t\n\nIntro.\n\n"
        "## Rules\n"
        "1. r1 [file: a.rs:10]\n"
        "2. r2 [file: a.rs:11]\n"
        "3. r3 [file: a.rs:12]\n\n"
        "## Anti-patterns\n"
        "- avoid X\n"
    )
    report = validate_completeness(md)
    assert report.is_complete


def test_validate_flags_missing_anti_patterns() -> None:
    md = (
        "# t\n\nIntro.\n\n"
        "## Rules\n"
        "1. r1 [file: a.rs:10]\n"
        "2. r2 [file: a.rs:11]\n"
        "3. r3 [file: a.rs:12]\n"
    )
    report = validate_completeness(md)
    assert not report.is_complete
    assert "anti-patterns" in report.missing_sections


def test_validate_flags_too_few_rules() -> None:
    md = (
        "# t\n\n"
        "## Rules\n"
        "1. r1 [file: a.rs:10]\n\n"
        "## Anti-patterns\n"
        "- x\n"
    )
    report = validate_completeness(md)
    assert not report.is_complete
    assert any("rules" in s for s in report.short_sections)


def test_validate_flags_missing_title() -> None:
    md = (
        "no heading\n\n"
        "## Rules\n"
        "1. r1 [file: a.rs:10]\n"
        "2. r2 [file: b.rs:20]\n"
        "3. r3 [file: a.rs:11]\n\n"
        "## Anti-patterns\n- x\n"
    )
    report = validate_completeness(md)
    assert "title" in report.missing_sections
