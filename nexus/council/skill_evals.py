"""Deterministic skill evals for generated Nexus Agent Skills.

The eval layer is deliberately small: deterministic gates catch structure,
identity, trigger, and citation faithfulness failures. Human approval remains
the final quality gate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from nexus.council.skill_parser import parse_skill_markdown, validate_skill_markdown
from nexus.council.state import EvidenceChunk, SkillDraft, SkillEvalResult, SkillPlanItem

SUITE_VERSION = "skill-quality-v1"
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


async def evaluate_skill_draft(
    *,
    draft: SkillDraft,
    evidence: Sequence[EvidenceChunk],
    plan: Sequence[SkillPlanItem],
    chat: object,
    attempt: int = 0,
    signals_used: Sequence[str] = (),
) -> SkillEvalResult:
    _ = chat
    failures: list[str] = []
    passed_checks = 0
    total_checks = 5

    if _valid_agent_skill_identity(draft):
        passed_checks += 1
    else:
        failures.append(
            "Agent Skills identity failed: name must be lowercase kebab-case, <=64 chars, "
            "and description must be 1-1024 chars."
        )

    structure = validate_skill_markdown(draft.body, tier=draft.tier)
    if structure.is_complete:
        passed_checks += 1
    else:
        failures.append(
            "Nexus structure failed: "
            + ", ".join([*structure.missing_sections, *structure.short_sections])
        )

    parsed = parse_skill_markdown(draft.body, fallback_name=draft.name, evidence=list(evidence))
    if parsed.name == draft.name:
        passed_checks += 1
    else:
        failures.append(
            f"Agent Skills name mismatch: body title normalizes to `{parsed.name}`, expected `{draft.name}`."
        )

    citation_failures = _citation_failures(parsed, evidence)
    if not citation_failures:
        passed_checks += 1
    else:
        failures.extend(citation_failures)

    trigger_failures = _trigger_failures(draft, plan)
    if not trigger_failures:
        passed_checks += 1
    else:
        failures.extend(trigger_failures)

    deterministic_score = passed_checks / total_checks
    quality_score = max(0.0, min(1.0, deterministic_score))
    status = "passed" if not failures else "failed"
    if status == "passed" and attempt > 0:
        status = "repaired"
    return SkillEvalResult(
        skill_name=draft.name,
        status=status,
        summary=_summary_for(status=status, failures=failures),
        failures=_dedupe(failures),
        quality_score=round(quality_score, 4),
        attempts=attempt,
        signals_used=list(signals_used),
    )


def failure_brief(result: SkillEvalResult) -> str:
    if not result.failures:
        return "No eval failures."
    return "\n".join(f"- {failure}" for failure in result.failures)


def _valid_agent_skill_identity(draft: SkillDraft) -> bool:
    return (
        bool(_NAME_RE.match(draft.name))
        and len(draft.description.strip()) > 0
        and len(draft.description) <= 1024
    )


def _citation_failures(parsed, evidence: Sequence[EvidenceChunk]) -> list[str]:
    if not parsed.citations:
        return ["Citation faithfulness failed: no real `[file: path:line]` citations found."]
    evidence_anchors = {(e.file, int(e.line)) for e in evidence}
    failures = [
        f"Citation faithfulness failed: `[file: {c.file}:{c.line}]` was not in retrieved evidence."
        for c in parsed.citations
        if (c.file, int(c.line)) not in evidence_anchors
    ]
    return failures[:5]


def _trigger_failures(draft: SkillDraft, plan: Sequence[SkillPlanItem]) -> list[str]:
    if not plan:
        return []
    positives = _positive_trigger_queries(draft)
    failures: list[str] = []
    for query in positives:
        ranked = _rank_plans(query, plan)
        if not ranked or ranked[0][1].name != draft.name or ranked[0][0] <= 0:
            failures.append(
                f"Description trigger failed: `{query}` did not rank `{draft.name}` first."
            )
    for query in ("what time is it", "summarize this unrelated PDF"):
        own_score = next(
            (score for score, item in _rank_plans(query, plan) if item.name == draft.name),
            0.0,
        )
        if own_score > 0:
            failures.append(
                f"Description over-triggered: unrelated prompt `{query}` matched `{draft.name}`."
            )
    return failures[:4]


def _positive_trigger_queries(draft: SkillDraft) -> list[str]:
    words = [t for t in _tokens(draft.description) if len(t) >= 4]
    phrase = " ".join(words[:4]) or draft.name
    return [f"I need help with {phrase}"]


def _rank_plans(query: str, plan: Sequence[SkillPlanItem]) -> list[tuple[float, SkillPlanItem]]:
    q_tokens = set(_tokens(query))
    scored: list[tuple[float, SkillPlanItem]] = []
    for item in plan:
        haystack = " ".join(
            [
                item.name,
                item.description,
                item.purpose,
                " ".join(str(t) for t in item.coverage.get("topics", [])),
            ]
        )
        h_tokens = set(_tokens(haystack))
        score = len(q_tokens & h_tokens) / max(len(q_tokens), 1)
        scored.append((score, item))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def _evidence_summary(evidence: Sequence[EvidenceChunk]) -> str:
    if not evidence:
        return "(none)"
    return "\n".join(f"- {e.chunk_id}: {e.file}:{e.line}" for e in evidence[:20])


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _summary_for(*, status: str, failures: Sequence[str]) -> str:
    if status in {"passed", "repaired"}:
        return "Skill passed Nexus quality eval."
    first = failures[0] if failures else "Unknown eval failure."
    return f"Skill failed Nexus quality eval: {first}"
