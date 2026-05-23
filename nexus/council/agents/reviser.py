"""Reviser — produces v2 of the proposal given a blocking critique.

Same long-form mechanics as Drafter: markdown output (not JSON), auto-
continuation on truncation, single section-fill pass if validation finds gaps.
Fires at most once per council session (revision_count caps at 1). Re-uses the
proposal id so the queue row updates in place. Sees the merged evidence pool
(Drafter's + Critic's combined via the state reducer).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from nexus.config import NexusConfig
from nexus.council.agents._common import evidence_for_prompt
from nexus.council.skill_parser import (
    parse_skill_markdown,
    strip_uncited_rules,
    validate_completeness,
)
from nexus.council.state import (
    AgentCost,
    CouncilState,
    DeliberationMessage,
)
from nexus.llm.client import ChatClient, TokenUsage
from nexus.skills.models import SkillProposal, compute_confidence

log = logging.getLogger(__name__)


_SYSTEM = (
    "You are the Reviser, an agent of the Nexus LLM Council. You are given a "
    "draft skill, the Critic's defect list, and the full evidence pool. Produce "
    "v2 of the skill that addresses every blocking defect. Keep what works; "
    "replace what fails. Every non-trivial claim MUST carry a "
    "`[file: path:line]` citation from the evidence. Uncited assertions in the "
    "Rules section will be stripped from your output."
)


_USER_TEMPLATE = """Topic: {topic}

# Draft v1 (to revise)

{draft_body}

# Critic's defects (you must address every blocking item)

{defects}

Critic's recommendation: {recommendation}

# Full evidence pool (drafter + critic combined)

{evidence}

# Task — output plain Markdown (no JSON wrapper, no surrounding code fences)

Required structure (in this order):

# {{kebab-case-name}}
A 2-3 sentence opening paragraph.
## Rules
1. … (3-7 numbered rules, each with `[file: path:line]`)
## Anti-patterns
- … (1-5 items)

Output the markdown directly. Do NOT wrap in JSON. Do NOT add commentary.
"""


_SECTION_FILL_TEMPLATE = """The following sections are missing or too short in
the revision you just produced:

{missing}

Here is the revision as it stands:

{current_body}

Here is the available evidence (cite by file:line):

{evidence}

Produce ONLY the missing section(s), in markdown, in the order listed above.
Each section starts with its `##` heading. Maintain the same style + voice.
"""


async def run(
    state: CouncilState,
    *,
    config: NexusConfig,
    chat: ChatClient,
) -> dict:
    proposal = state.get("proposal")
    critique = state.get("critique")
    evidence = state.get("evidence") or []

    if proposal is None or critique is None:
        return {}

    defects = (
        "\n".join(f"- {i.get('description', '')}" for i in critique.issues)
        or "(none listed)"
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                topic=state["topic"],
                draft_body=proposal.body,
                defects=defects,
                recommendation=critique.recommendation or "(none)",
                evidence=evidence_for_prompt(evidence),
            ),
        },
    ]
    resp = await chat.chat_markdown(messages, max_tokens=3000, max_continuations=2)
    usage = resp.usage
    body = resp.content.strip()

    # ---- completeness gate ----
    report = validate_completeness(body)
    fill_attempts = 0
    while not report.is_complete and fill_attempts < 1:
        fill_attempts += 1
        missing_summary = _format_missing(report)
        log.info("reviser: section-fill pass — %s", missing_summary)
        fill_messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _SECTION_FILL_TEMPLATE.format(
                    missing=missing_summary,
                    current_body=body,
                    evidence=evidence_for_prompt(evidence),
                ),
            },
        ]
        fill_resp = await chat.chat_markdown(
            fill_messages, max_tokens=1500, max_continuations=1
        )
        body = _merge_section_fill(body, fill_resp.content.strip())
        usage = TokenUsage(
            prompt=usage.prompt + fill_resp.usage.prompt,
            completion=usage.completion + fill_resp.usage.completion,
        )
        report = validate_completeness(body)

    body, dropped = strip_uncited_rules(body)
    parsed = parse_skill_markdown(body, fallback_name=proposal.name, evidence=evidence)

    paragraphs = max(1, parsed.body.count("\n\n") + 1)
    confidence = compute_confidence(
        citations=parsed.citations, paragraphs=paragraphs, revision_count=1
    )

    revised = SkillProposal(
        id=proposal.id,  # re-use id so queue row updates in place
        name=parsed.name,
        body=parsed.body,
        citations=parsed.citations,
        confidence=confidence,
        status="pending",
        created_at=proposal.created_at,
    )

    note_parts: list[str] = []
    if dropped:
        note_parts.append(f"{dropped} uncited line(s) stripped")
    if resp.truncated:
        note_parts.append("continued after truncation")
    if fill_attempts:
        note_parts.append(f"{fill_attempts} section-fill pass(es)")
    note = f" ({'; '.join(note_parts)})" if note_parts else ""

    summary = (
        f"Revised **{parsed.name}** — confidence {confidence:.2f}, "
        f"{len(parsed.citations)} citations, addressing "
        f"{len(critique.issues)} defect(s){note}."
    )

    return {
        "proposal": revised,
        "revision_count": 1,
        "critique": None,
        "deliberation": [
            DeliberationMessage(
                agent="reviser",
                timestamp=datetime.now(UTC).isoformat(),
                body=summary,
                cite_ids=[c.id for c in parsed.citations if c.id],
            )
        ],
        "costs": [
            AgentCost(
                agent="reviser",
                prompt_tokens=usage.prompt,
                completion_tokens=usage.completion,
                model=chat.model,
            )
        ],
    }


# ---------------------------------------------------------------- helpers


def _format_missing(report) -> str:
    parts = list(report.missing_sections) + list(report.short_sections)
    return ", ".join(parts) if parts else "(none)"


def _merge_section_fill(current: str, fill: str) -> str:
    fill = fill.strip()
    if not fill:
        return current
    return current.rstrip() + "\n\n" + fill + "\n"
