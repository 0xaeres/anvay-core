"""Critic — verifies the Drafter's proposal against the source corpus.

The critic does its OWN fresh retrieval (not just the drafter's evidence) and
scores the draft against a fixed faithfulness rubric. Re-retrieval is the proven
lever per Reflexion (2023) and Anthropic's Constitutional AI — without it the
critic devolves into sycophantic agreement.

Severity gate: only `blocking` triggers a single revision pass.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from nexus.config import NexusConfig
from nexus.council.agents._common import (
    evidence_for_prompt,
    hits_to_evidence,
)
from nexus.council.state import (
    AgentCost,
    CouncilState,
    DeliberationMessage,
)
from nexus.llm.client import ChatClient
from nexus.retrieval.pipeline import RetrievalContext, retrieve
from nexus.skills.models import Critique, SkillProposal

log = logging.getLogger(__name__)


_SYSTEM = (
    "You are the Critic, an agent of the Nexus LLM Council. Your job is to "
    "verify a draft skill against the actual codebase. You have FRESH evidence "
    "retrieved for this verification pass — different chunks than the drafter "
    "saw. Compare the draft's claims against this evidence. Be skeptical, "
    "specific, and bounded: never invent code that isn't in the evidence; cite "
    "the file:line of every flaw."
)


_USER_TEMPLATE = """Topic: {topic}

# Draft under review

{draft_body}

# Fresh evidence (re-retrieved for verification — may overlap with the
# drafter's evidence but is queried independently)

{fresh_evidence}

# Rubric — score the draft on each axis

1. **Faithfulness** — does every cited `[file: path:line]` actually appear in
   the evidence? Are there claims unsupported by ANY evidence (hallucinations)?
2. **Completeness** — does the draft miss obvious patterns visible in the
   evidence? Cite the missing thing.
3. **Specificity** — are rules concrete (cite a function/file) or vague
   ("use best practices")? Vague rules are defects.
4. **Anti-patterns** — does it warn against real footguns visible in the
   evidence?

# Output

For each defect, emit one issue. Classify the WORST defect's severity:

- **blocking**: hallucinated citation, fabricated symbol, or a rule that
  contradicts the evidence. Reviser MUST address before human sees this.
- **major**: missing important pattern, vague rule, or unsupported claim
  that isn't an outright fabrication. Net-useful draft, but should be fixed.
- **minor**: wording, polish, nits.

Output ONLY JSON in this schema:

{{
  "severity": "blocking" | "major" | "minor",
  "issues": [
    {{"description": "specific defect with file:line if applicable",
      "counter_example": "optional — what the evidence actually shows"}}
  ],
  "recommendation": "1-3 sentence directive to the Reviser (only acted on if blocking)"
}}

If you find no real defects, return severity `minor` with an empty issues list.
"""


async def run(
    state: CouncilState,
    *,
    config: NexusConfig,
    retrieval: RetrievalContext,
    chat: ChatClient,
) -> dict:
    proposal = state.get("proposal")
    if proposal is None:
        return {}

    fresh = await _retrieve_for_critique(
        retrieval=retrieval,
        product_id=state["product_id"],
        topic=state["topic"],
        proposal=proposal,
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                topic=state["topic"],
                draft_body=proposal.body,
                fresh_evidence=evidence_for_prompt(fresh) or "(no fresh evidence)",
            ),
        },
    ]
    payload, usage = await chat.chat_json(messages, max_tokens=1500)

    severity = str(payload.get("severity", "minor")).lower()
    if severity not in ("blocking", "major", "minor"):
        severity = "minor"
    issues = [
        {
            "description": str(i.get("description", "")).strip(),
            "counter_example": str(i.get("counter_example", "")).strip() or None,
        }
        for i in (payload.get("issues") or [])
        if isinstance(i, dict) and i.get("description")
    ]
    crit = Critique(
        severity=severity,  # type: ignore[arg-type]
        issues=issues,
        recommendation=str(payload.get("recommendation", "")).strip(),
    )

    # Stamp the critique onto the proposal so the queue row carries it.
    updated_proposal = proposal.model_copy(update={"adversary_critique": crit})

    msg = DeliberationMessage(
        agent="critic",
        timestamp=datetime.now(UTC).isoformat(),
        body=_render_summary(crit),
    )
    cost = AgentCost(
        agent="critic",
        prompt_tokens=usage.prompt,
        completion_tokens=usage.completion,
        model=chat.model,
    )
    return {
        "critique": crit,
        "proposal": updated_proposal,
        "evidence": fresh,  # merged into shared evidence pool via reducer
        "deliberation": [msg],
        "costs": [cost],
    }


# ---------------------------------------------------------------- helpers


async def _retrieve_for_critique(
    *,
    retrieval: RetrievalContext,
    product_id: str,
    topic: str,
    proposal: SkillProposal,
):
    """Re-retrieve using the proposal's own claims as the query — surfaces
    counter-evidence the drafter may have missed."""
    cited_files = " ".join({c.file for c in proposal.citations[:6]})
    query = f"{topic} {cited_files} {proposal.name}".strip()
    result = await retrieve(
        ctx=retrieval, product_id=product_id, query=query, top_k=15, mode="auto"
    )
    return hits_to_evidence(result.hits, limit=15)


def _render_summary(crit: Critique) -> str:
    lead = {
        "blocking": "BLOCKING critique — Reviser will run.",
        "major": "Major critique attached (no revision).",
        "minor": "Minor critique attached.",
    }[crit.severity]
    parts = [lead]
    if crit.issues:
        parts.append("\nIssues:")
        for i in crit.issues:
            parts.append(f"  - {i['description']}")
    if crit.recommendation:
        parts.append(f"\nRecommendation: {crit.recommendation}")
    return "\n".join(parts)
