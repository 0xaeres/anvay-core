"""Drafter — retrieves evidence and writes the initial skill proposal.

One LLM call. Receives top-k retrieved chunks; produces a Markdown skill body
with `[file: path:line]` citations on every non-trivial claim. Uncited
assertions inside the Rules section are stripped before storage.
"""

from __future__ import annotations

import logging
import re
import uuid
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
    EvidenceChunk,
)
from nexus.llm.client import ChatClient
from nexus.retrieval.pipeline import RetrievalContext, retrieve
from nexus.skills.models import Citation, SkillProposal, compute_confidence

log = logging.getLogger(__name__)


_SYSTEM = (
    "You are the Drafter, an agent of the Nexus LLM Council. Your job: read "
    "retrieved code and documentation evidence for a software product, and "
    "produce a SKILL — a short, opinionated Markdown playbook that guides "
    "future AI agents working in this codebase. Every non-trivial claim must "
    "carry a `[file: path:line]` citation drawn from the evidence below. "
    "Uncited claims will be stripped from your output."
)


_USER_TEMPLATE = """Topic: {topic}
Product: {product_id}

# Retrieved evidence

Each excerpt is labelled [E1], [E2], etc. with its file:line anchor.

{evidence}

# Task

Draft the skill as Markdown.

Structure:
1. `# Title` — kebab-cased noun phrase.
2. A 2-3 sentence opening that frames why this skill matters.
3. `## Rules` — 3-7 numbered rules. Each rule MUST cite at least one
   `[file: path:line]` from the evidence above.
4. `## Anti-patterns` — concrete things to avoid. Cite where possible;
   uncited general-best-practice items are allowed here.

Output ONLY JSON in this schema (no markdown fences):

{{
  "name": "kebab-case-skill-name",
  "body": "the markdown body as a single string",
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
    retrieval: RetrievalContext,
    chat: ChatClient,
) -> dict:
    topic = state["topic"]
    product_id = state["product_id"]

    result = await retrieve(
        ctx=retrieval, product_id=product_id, query=topic, top_k=20, mode="auto"
    )
    evidence = hits_to_evidence(result.hits, limit=20)

    if not evidence:
        return _empty_update(topic, chat.model)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                topic=topic,
                product_id=product_id,
                evidence=evidence_for_prompt(evidence),
            ),
        },
    ]
    payload, usage = await chat.chat_json(messages, max_tokens=3000)

    name = _normalise_name(payload.get("name") or topic)
    raw_body = str(payload.get("body", "")).strip()
    body, dropped = _strip_uncited_assertions(raw_body, evidence)
    citations = _build_citations(payload.get("citations") or [], evidence)
    paragraphs = max(1, body.count("\n\n") + 1)
    confidence = compute_confidence(
        citations=citations, paragraphs=paragraphs, revision_count=0
    )

    proposal = SkillProposal(
        id=str(uuid.uuid4()),
        name=name,
        body=body,
        citations=citations,
        confidence=confidence,
        status="pending",
        created_at=datetime.now(UTC).isoformat(),
    )

    note = f" ({dropped} uncited line(s) stripped)" if dropped else ""
    summary = (
        f"Drafted **{name}** — confidence {confidence:.2f}, "
        f"{len(citations)} citations, {paragraphs} paragraphs{note}."
    )

    return {
        "evidence": evidence,
        "proposal": proposal,
        "proposal_id": proposal.id,
        "revision_count": 0,
        "critique": None,
        "deliberation": [
            DeliberationMessage(
                agent="drafter",
                timestamp=datetime.now(UTC).isoformat(),
                body=summary,
                cite_ids=[c.id for c in citations if c.id],
            )
        ],
        "costs": [
            AgentCost(
                agent="drafter",
                prompt_tokens=usage.prompt,
                completion_tokens=usage.completion,
                model=chat.model,
            )
        ],
    }


# ---------------------------------------------------------------- helpers

_NAME_RE = re.compile(r"[^a-z0-9-]+")
_DASH_RUN = re.compile(r"-{2,}")
_CITATION_RE = re.compile(r"\[(?:file|cve)[^\]]+\]", re.IGNORECASE)


def _normalise_name(raw: str) -> str:
    s = raw.strip().lower().replace("_", "-").replace(" ", "-")
    s = _NAME_RE.sub("-", s)
    s = _DASH_RUN.sub("-", s).strip("-")
    return s[:60] or "untitled-skill"


def _strip_uncited_assertions(body: str, evidence: list[EvidenceChunk]) -> tuple[str, int]:
    """Drop list items in `## Rules` that lack any citation marker."""
    rules_block = re.search(
        r"##\s+Rules(.*?)(?=\n##\s+|\Z)", body, flags=re.DOTALL | re.IGNORECASE
    )
    if not rules_block:
        return body, 0

    block_text = rules_block.group(1)
    new_lines: list[str] = []
    dropped = 0
    for line in block_text.splitlines():
        is_list_item = bool(re.match(r"^\s*(?:\d+\.|[-*])\s", line))
        if is_list_item and not _CITATION_RE.search(line):
            dropped += 1
            continue
        new_lines.append(line)
    if dropped == 0:
        return body, 0
    new_block = "\n".join(new_lines)
    return body[: rules_block.start(1)] + new_block + body[rules_block.end(1) :], dropped


def _build_citations(raw: list, evidence: list[EvidenceChunk]) -> list[Citation]:
    by_anchor: dict[tuple[str, int], EvidenceChunk] = {(e.file, e.line): e for e in evidence}
    out: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for c in raw:
        try:
            file_ = str(c.get("file"))
            line = int(c.get("line"))
        except Exception:
            continue
        key = (file_, line)
        if key in seen:
            continue
        evi = by_anchor.get(key)
        out.append(
            Citation(
                id=evi.chunk_id if evi else None,
                file=file_,
                line=line,
                excerpt=(evi.excerpt if evi else str(c.get("excerpt", ""))),
            )
        )
        seen.add(key)
    return out


def _empty_update(topic: str, model: str) -> dict:
    proposal = SkillProposal(
        id=str(uuid.uuid4()),
        name=_normalise_name(topic),
        body=(
            "# (no proposal)\n\n"
            "The council could not gather enough evidence to draft a skill. "
            "Run an ingest against the relevant sources and try again."
        ),
        citations=[],
        confidence=0.0,
        status="pending",
        created_at=datetime.now(UTC).isoformat(),
    )
    return {
        "evidence": [],
        "proposal": proposal,
        "proposal_id": proposal.id,
        "revision_count": 0,
        "critique": None,
        "deliberation": [
            DeliberationMessage(
                agent="drafter",
                timestamp=datetime.now(UTC).isoformat(),
                body="No evidence retrieved — empty placeholder proposal.",
            )
        ],
        "costs": [AgentCost(agent="drafter", model=model)],
    }
