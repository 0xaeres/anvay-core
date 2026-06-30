"""Deterministic skill evals for generated Anvay Agent Skills.

The eval layer is deliberately small: deterministic gates catch structure,
identity, trigger, and citation faithfulness failures. Human approval remains
the final quality gate.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from anvay.council.skill_parser import parse_skill_markdown, validate_skill_markdown
from anvay.council.state import EvidenceChunk, SkillDraft, SkillEvalResult, SkillPlanItem

log = logging.getLogger(__name__)

SUITE_VERSION = "skill-quality-v1"
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_CITATION_RE = re.compile(r"\[file:\s*(.+?):(\d+)\]", re.IGNORECASE)

# Faithfulness gate bounds: judge at most this many cited claims per draft, and
# only judge claims whose cited excerpt is long enough to entail anything. Short
# or empty excerpts (e.g. test stubs) are skipped so the gate stays fail-soft.
_MAX_FAITHFULNESS_CLAIMS = 6
_MIN_EXCERPT_LEN = 24


async def evaluate_skill_draft(
    *,
    draft: SkillDraft,
    evidence: Sequence[EvidenceChunk],
    plan: Sequence[SkillPlanItem],
    chat: object,
    attempt: int = 0,
    signals_used: Sequence[str] = (),
) -> SkillEvalResult:
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
            "Anvay structure failed: "
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

    # Deterministic checks set the quality score. The LLM faithfulness gate is
    # an additional, fail-soft pass: it can only add failures (never raise the
    # score), and it no-ops unless a real chat client and citable excerpts are
    # present. Human approval remains the final quality gate.
    deterministic_score = passed_checks / total_checks
    quality_score = max(0.0, min(1.0, deterministic_score))
    signals = list(signals_used)
    faithfulness_failures = await _faithfulness_failures(draft=draft, evidence=evidence, chat=chat)
    if faithfulness_failures:
        failures.extend(faithfulness_failures)
        signals.append("llm_faithfulness")

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
        signals_used=signals,
    )


def failure_brief(result: SkillEvalResult) -> str:
    if not result.failures:
        return "No eval failures."
    return "\n".join(f"- {failure}" for failure in result.failures)


def _cited_claims(body: str, evidence: Sequence[EvidenceChunk]) -> list[dict]:
    """Pair each `[file: path:line]` citation with its claim line and excerpt.

    Only claims whose cited (file, line) maps to a non-trivial retrieved excerpt
    are returned — the judge needs source text to check entailment against.
    """
    excerpt_by_anchor = {(e.file, int(e.line)): e.excerpt for e in evidence}
    claims: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for line in body.splitlines():
        match = _CITATION_RE.search(line)
        if not match:
            continue
        file_ = match.group(1).strip()
        try:
            anchor_line = int(match.group(2))
        except ValueError:
            continue
        claim = _CITATION_RE.sub("", line).strip(" -*0123456789.\t")
        excerpt = (excerpt_by_anchor.get((file_, anchor_line)) or "").strip()
        if len(claim) < 8 or len(excerpt) < _MIN_EXCERPT_LEN:
            continue
        key = (file_, claim, anchor_line)
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {"id": len(claims), "claim": claim, "anchor": f"{file_}:{anchor_line}", "excerpt": excerpt}
        )
        if len(claims) >= _MAX_FAITHFULNESS_CLAIMS:
            break
    return claims


async def _faithfulness_failures(
    *,
    draft: SkillDraft,
    evidence: Sequence[EvidenceChunk],
    chat: object,
) -> list[str]:
    """Bounded, fail-soft LLM entailment check for cited claims.

    Returns failure strings only for claims the judge marks as NOT supported by
    their cited excerpt. Any error, missing chat client, unparseable judge
    output, or absence of citable excerpts yields no failures (fail-soft).
    """
    judge = getattr(chat, "chat", None)
    if not callable(judge):
        return []
    claims = _cited_claims(draft.body, evidence)
    if not claims:
        return []
    payload = {
        "claims": [{"id": c["id"], "claim": c["claim"], "excerpt": c["excerpt"]} for c in claims]
    }
    try:
        resp = await judge(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict citation-faithfulness judge for an engineering "
                        "knowledge base. For each claim, decide whether the claim is "
                        "supported (entailed) by its cited excerpt alone. Mark a claim "
                        "unsupported only when the excerpt clearly does not back it. When "
                        "unsure, treat it as supported. Return JSON "
                        '{"unsupported": [<id>, ...]} listing only unsupported claim ids.'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            json_mode=True,
            max_tokens=400,
            stream=False,
        )
    except Exception as e:  # fail-soft: never block the gate on judge errors
        log.warning("faithfulness judge failed for skill %s: %s", draft.name, e)
        return []

    unsupported = _parse_unsupported_ids(getattr(resp, "content", "") or "")
    by_id = {c["id"]: c for c in claims}
    failures = [
        f"Citation faithfulness failed: claim `{by_id[i]['claim'][:80]}` is not supported "
        f"by cited excerpt `[{by_id[i]['anchor']}]`."
        for i in unsupported
        if i in by_id
    ]
    return failures[:_MAX_FAITHFULNESS_CLAIMS]


def _parse_unsupported_ids(content: str) -> list[int]:
    text = content.strip()
    if not text:
        return []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    raw = data.get("unsupported") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


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
    """Return 3 semantically varied positive queries to catch narrow routing.

    Using a single query trivially passes for almost any coherent description.
    Three varied phrasings (imperative, explanatory, identity) catch edge cases
    where the routing logic only fires on exact lexical overlap.
    """
    words = [t for t in _tokens(draft.description) if len(t) >= 4]
    name_phrase = draft.name.replace("-", " ")
    queries: list[str] = []

    # 1. Imperative phrasing from the first half of the description.
    if words:
        queries.append(f"I need help with {' '.join(words[:4])}")

    # 2. How-to phrasing from the second half (different token window).
    if len(words) >= 5:
        queries.append(f"how do I handle {' '.join(words[2:6])}")
    elif words:
        queries.append(f"how do I use {name_phrase}")

    # 3. Explain phrasing using the skill name — always distinct from the above.
    queries.append(f"explain {name_phrase}")

    # De-duplicate while preserving order, cap at 3.
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result[:3] or [f"I need help with {name_phrase}"]


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
        return "Skill passed Anvay quality eval."
    first = failures[0] if failures else "Unknown eval failure."
    return f"Skill failed Anvay quality eval: {first}"
