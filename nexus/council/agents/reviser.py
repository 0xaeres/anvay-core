"""Reviser — produces v2 of the proposal given a blocking critique.

Fires at most once per council session (revision_count caps at 1). Re-uses the
proposal id so the queue row updates in place. Sees the original evidence pool
(Drafter's + Critic's combined) so it can address defects without re-retrieving.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from nexus.config import NexusConfig
from nexus.council.agents._common import evidence_for_prompt
from nexus.council.agents.drafter import (
    _build_citations,
    _normalise_name,
    _strip_uncited_assertions,
)
from nexus.council.state import (
    AgentCost,
    CouncilState,
    DeliberationMessage,
)
from nexus.llm.client import ChatClient
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

# Task

Produce v2. Output ONLY JSON in this schema (no markdown fences):

{{
  "name": "kebab-case-skill-name",
  "body": "the revised markdown body as a single string",
  "citations": [
    {{"file": "path", "line": 42, "excerpt": "..."}}
  ]
}}

`citations` must contain every distinct file:line you used in the body.
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

    defects = "\n".join(f"- {i.get('description', '')}" for i in critique.issues) or "(none listed)"

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
    payload, usage = await chat.chat_json(messages, max_tokens=3000)

    name = _normalise_name(payload.get("name") or proposal.name)
    raw_body = str(payload.get("body", "")).strip()
    body, dropped = _strip_uncited_assertions(raw_body, evidence)
    citations = _build_citations(payload.get("citations") or [], evidence)
    paragraphs = max(1, body.count("\n\n") + 1)
    confidence = compute_confidence(
        citations=citations, paragraphs=paragraphs, revision_count=1
    )

    revised = SkillProposal(
        id=proposal.id,  # re-use id so queue row updates in place
        name=name,
        body=body,
        citations=citations,
        confidence=confidence,
        status="pending",
        created_at=proposal.created_at,
    )

    note = f" ({dropped} uncited line(s) stripped)" if dropped else ""
    summary = (
        f"Revised **{name}** — confidence {confidence:.2f}, "
        f"{len(citations)} citations, addressing {len(critique.issues)} "
        f"defect(s){note}."
    )

    return {
        "proposal": revised,
        "revision_count": 1,
        "critique": None,  # cleared; Critic does not run again after Reviser
        "deliberation": [
            DeliberationMessage(
                agent="reviser",
                timestamp=datetime.now(UTC).isoformat(),
                body=summary,
                cite_ids=[c.id for c in citations if c.id],
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


# silence unused-import
_ = re
