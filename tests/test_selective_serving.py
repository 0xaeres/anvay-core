"""Tests for MCP selective-serving in nexus.mcp_server.tools.

Verifies that find_skills honors applies_to.files, applies_to.contexts, and
walks composes_with so the LLM context stays tight.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from nexus.mcp_server.tools import (
    ToolState,
    _expand_composes_with,
    _matches_context,
    _matches_file_globs,
    find_skills,
)
from nexus.skills.models import (
    AppliesTo,
    OrgSkill,
    OrgSkillKind,
)
from nexus.skills.store import SkillStore

STARTER_ROOT = (
    Path(__file__).resolve().parent.parent / "nexus" / "skills" / "starter" / "shared"
)


# ---------- helpers ----------------------------------------------------------


def _state_with_skills(skills: list[OrgSkill]) -> ToolState:
    config = MagicMock()
    state = ToolState(product="test", config=config)
    fake_store = MagicMock(spec=SkillStore)
    fake_store.iter_skills.return_value = skills
    state._store = fake_store
    return state


def _skill(
    name: str,
    *,
    kind: OrgSkillKind = OrgSkillKind.LANGUAGE,
    files: list[str] | None = None,
    contexts: list[str] | None = None,
    composes_with: list[str] | None = None,
    body: str = "",
) -> OrgSkill:
    return OrgSkill(
        name=name,
        kind=kind,
        confidence=0.8,
        quality_score=0.8,
        ratified_by="t",
        ratified_at="2026-05-23T00:00:00Z",
        applies_to=AppliesTo(files=files or [], contexts=contexts or []),
        composes_with=composes_with or [],
        body=body or f"# {name}\n\nbody about {name}.",
    )


# ---------- _matches_file_globs ---------------------------------------------


def test_matches_file_globs_empty_means_universal() -> None:
    assert _matches_file_globs("any/path.py", []) is True
    assert _matches_file_globs(None, []) is True


def test_matches_file_globs_no_current_file_passes() -> None:
    # No current_file means "no file filter requested"; every skill passes.
    assert _matches_file_globs(None, ["**/*.py"]) is True


def test_matches_file_globs_matches_recursive_glob() -> None:
    assert _matches_file_globs("src/app/main.py", ["**/*.py"]) is True
    assert _matches_file_globs("src/app/main.py", ["**/*.ts"]) is False


def test_matches_file_globs_any_of_many() -> None:
    globs = ["**/*.ts", "**/*.tsx"]
    assert _matches_file_globs("ui/Foo.tsx", globs) is True
    assert _matches_file_globs("server/main.py", globs) is False


# ---------- _matches_context -------------------------------------------------


def test_matches_context_empty_skill_contexts_passes() -> None:
    assert _matches_context("code-review", []) is True


def test_matches_context_general_is_no_filter() -> None:
    assert _matches_context("general", ["code-review"]) is True
    assert _matches_context("", ["code-review"]) is True


def test_matches_context_requires_exact_membership() -> None:
    assert _matches_context("security-audit", ["code-review"]) is False
    assert _matches_context("security-audit", ["security-audit", "code-review"]) is True


# ---------- _expand_composes_with -------------------------------------------


def test_expand_composes_with_includes_prerequisites() -> None:
    a = _skill("a")
    b = _skill("b", composes_with=["org/a"])
    out = _expand_composes_with([b], [a, b])
    ids = [s.id for s in out]
    assert "org/b" in ids
    assert "org/a" in ids


def test_expand_composes_with_transitive() -> None:
    a = _skill("a")
    b = _skill("b", composes_with=["org/a"])
    c = _skill("c", composes_with=["org/b"])
    out = _expand_composes_with([c], [a, b, c])
    ids = {s.id for s in out}
    assert ids == {"org/a", "org/b", "org/c"}


def test_expand_composes_with_tolerates_cycles() -> None:
    a = _skill("a", composes_with=["org/b"])
    b = _skill("b", composes_with=["org/a"])
    out = _expand_composes_with([a], [a, b])
    assert len(out) == 2
    assert {s.id for s in out} == {"org/a", "org/b"}


def test_expand_composes_with_missing_dep_is_silent() -> None:
    a = _skill("a", composes_with=["org/does-not-exist"])
    out = _expand_composes_with([a], [a])
    assert [s.id for s in out] == ["org/a"]


# ---------- find_skills filtering -------------------------------------------


def test_find_skills_current_file_filters_by_globs() -> None:
    py = _skill("python-conventions", files=["**/*.py"])
    ts = _skill("typescript-conventions", files=["**/*.ts", "**/*.tsx"])
    universal = _skill("code-review", files=[], contexts=["code-review"])
    state = _state_with_skills([py, ts, universal])

    result = asyncio.run(
        find_skills(state, query="review this", current_file="src/foo.py")
    )
    ids = [s["id"] for s in result["skills"]]
    assert "org/python-conventions" in ids
    assert "org/typescript-conventions" not in ids
    assert "org/code-review" in ids  # empty files == universal


def test_find_skills_context_filter_drops_irrelevant() -> None:
    review = _skill("code-review", contexts=["code-review"])
    sec = _skill("security-baseline", contexts=["security-audit"])
    universal = _skill("git-workflow", contexts=[])
    state = _state_with_skills([review, sec, universal])

    result = asyncio.run(find_skills(state, query="audit", context="security-audit"))
    ids = [s["id"] for s in result["skills"]]
    assert "org/security-baseline" in ids
    assert "org/code-review" not in ids
    assert "org/git-workflow" in ids


def test_find_skills_general_context_disables_filter() -> None:
    review = _skill("code-review", contexts=["code-review"])
    sec = _skill("security-baseline", contexts=["security-audit"])
    state = _state_with_skills([review, sec])

    result = asyncio.run(find_skills(state, query="anything", context="general"))
    ids = {s["id"] for s in result["skills"]}
    assert ids == {"org/code-review", "org/security-baseline"}


def test_find_skills_pulls_in_composes_with() -> None:
    js = _skill("javascript-conventions", files=["**/*.js"])
    ts = _skill(
        "typescript-conventions",
        files=["**/*.ts"],
        composes_with=["org/javascript-conventions"],
    )
    state = _state_with_skills([js, ts])

    result = asyncio.run(
        find_skills(state, query="lint", current_file="src/foo.ts")
    )
    ids = {s["id"] for s in result["skills"]}
    assert ids == {"org/typescript-conventions", "org/javascript-conventions"}

    by_id = {s["id"]: s for s in result["skills"]}
    assert by_id["org/typescript-conventions"]["included_as"] == "match"
    assert by_id["org/javascript-conventions"]["included_as"] == "prerequisite"


def test_find_skills_reports_filter_metadata() -> None:
    py = _skill("python-conventions", files=["**/*.py"])
    ts = _skill("typescript-conventions", files=["**/*.ts"])
    state = _state_with_skills([py, ts])
    result = asyncio.run(
        find_skills(state, query="review", current_file="src/foo.py")
    )
    assert result["filtered_from"] == 2
    assert result["current_file"] == "src/foo.py"
    assert len(result["skills"]) == 1


# ---------- end-to-end on the real starter pack -----------------------------


def test_starter_pack_python_review_keeps_relevant_only() -> None:
    """Reviewing a .py file should drop Java/HTML/CSS/Angular/Springboot/TS."""
    store = SkillStore(STARTER_ROOT)
    skills = store.iter_skills()
    assert len(skills) >= 13

    config = MagicMock()
    state = ToolState(product="test", config=config)
    fake_store = MagicMock(spec=SkillStore)
    fake_store.iter_skills.return_value = skills
    state._store = fake_store

    result = asyncio.run(
        find_skills(
            state,
            query="review this code",
            context="code-review",
            current_file="src/app/main.py",
        )
    )
    ids = {s["id"] for s in result["skills"]}

    # Should keep: python, plus universal practices tagged code-review
    assert "org/python-conventions" in ids
    assert "org/code-review-baseline" in ids
    assert "org/security-hardening" in ids

    # Should drop language skills that don't match .py:
    assert "org/java-conventions" not in ids
    assert "org/typescript-conventions" not in ids
    assert "org/javascript-conventions" not in ids
    assert "org/html-conventions" not in ids
    assert "org/css-conventions" not in ids
    assert "org/angular-conventions" not in ids
    assert "org/springboot-conventions" not in ids


def test_starter_pack_tsx_review_brings_in_js_via_composes_with() -> None:
    """TypeScript declares composes_with JavaScript; the chain should pull it in."""
    store = SkillStore(STARTER_ROOT)
    skills = store.iter_skills()
    config = MagicMock()
    state = ToolState(product="test", config=config)
    fake_store = MagicMock(spec=SkillStore)
    fake_store.iter_skills.return_value = skills
    state._store = fake_store

    result = asyncio.run(
        find_skills(
            state,
            query="review",
            context="code-review",
            current_file="ui/components/Foo.tsx",
        )
    )
    ids = {s["id"] for s in result["skills"]}
    assert "org/typescript-conventions" in ids
    # JS is a prerequisite via composes_with even though .tsx doesn't match its files glob
    assert "org/javascript-conventions" in ids
    by_id = {s["id"]: s for s in result["skills"]}
    assert by_id["org/javascript-conventions"]["included_as"] == "prerequisite"
