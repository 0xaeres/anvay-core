"""Single product-named skill council nodes.

The graph drafts one generic product context Markdown skill. Generated output
stays a proposal until human approval writes `<product-name>-skill.md`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from anvay.config import AnvayConfig
from anvay.council.agents._common import (
    evidence_for_prompt,
    evidence_set_to_evidence,
    hits_to_evidence,
)
from anvay.council.errors import CouncilIncompleteSkill, CouncilNoEvidence
from anvay.council.kb_guide import kb_usage_fill
from anvay.council.skill_catalog import (
    PRODUCT_SKILL_RETRIEVAL_QUERY,
    SKILL_CATALOG,
    catalog_plan,
)
from anvay.council.skill_evals import evaluate_skill_draft, failure_brief
from anvay.council.skill_parser import (
    parse_skill_markdown,
    required_sections_for_tier,
    strip_uncited_rules,
    validate_skill_markdown,
)
from anvay.council.state import (
    AgentCost,
    CouncilState,
    DeliberationMessage,
    EvidenceChunk,
    SkillDraft,
    SkillEvalResult,
    SkillPlanItem,
)
from anvay.llm.client import ChatClient, TokenUsage
from anvay.retrieval.chunk_grep import grep_indexed_chunks, sample_indexed_chunks
from anvay.retrieval.evidence import retrieve_evidence
from anvay.retrieval.pipeline import RetrievalContext, retrieve
from anvay.retrieval.repomap import load_repo_map_for_product, topic_bias_terms
from anvay.skills.models import Skill, SkillCoverage, SkillProposal, compute_confidence
from anvay.skills.store import SkillStore

log = logging.getLogger(__name__)

REPAIR_ATTEMPT_CAP = 3
EVIDENCE_CHUNKS_PER_SESSION_CAP = 20


async def planner(
    state: CouncilState,
    *,
    config: AnvayConfig,
    retrieval: RetrievalContext,
    chat: ChatClient,
    graph_store: object | None = None,
) -> dict:
    topic = state["topic"]
    product_id = state["product_id"]
    evidence: list[EvidenceChunk] = []
    result = await _retrieve_skill_evidence(
        retrieval=retrieval,
        product_id=product_id,
        query=_retrieval_query(topic, suffix=PRODUCT_SKILL_RETRIEVAL_QUERY),
        top_k=20,
        graph_store=graph_store,
        skills=await _approved_skills(config, product_id),
    )
    evidence.extend(_skill_result_to_evidence(result, limit=20))

    evidence = _select_evidence(evidence, limit=EVIDENCE_CHUNKS_PER_SESSION_CAP)
    if not evidence:
        raise _no_evidence_error(result, config)

    # Deterministic context pack: the structural/graph summaries the ingest
    # pipeline already distilled (see anvay/ingest/summaries.py). The synthesizer
    # grounds the system-map / data-model / interfaces sections on these instead
    # of an LLM expert re-deriving them from raw chunks.
    context_pack = _summaries_block(result)

    plan = catalog_plan(product_id, topic)
    msg = DeliberationMessage(
        agent="planner",
        timestamp=datetime.now(UTC).isoformat(),
        body="Planned one generic product context skill from KB artifacts.",
        cite_ids=[e.chunk_id for e in evidence[:8]],
    )
    return {
        "evidence": evidence,
        "skill_plan": plan,
        "context_pack": context_pack,
        "deliberation": [msg],
        "costs": [],
    }


async def synthesizer(
    state: CouncilState,
    *,
    config: AnvayConfig,
    chat: ChatClient,
) -> dict:
    evidence = _select_evidence(
        state.get("evidence") or [], limit=EVIDENCE_CHUNKS_PER_SESSION_CAP
    )
    context_pack = state.get("context_pack") or "(none)"
    plan = state.get("skill_plan") or catalog_plan(state["product_id"], state["topic"])
    repo_map = load_repo_map_for_product(config, state["product_id"])
    repo_map_block = repo_map.render(
        bias_terms=topic_bias_terms(state["topic"]), token_budget=700
    )

    drafts: list[SkillDraft] = []
    usage_total = TokenUsage()
    for item in plan[:1]:
        required = required_sections_for_tier(item.tier)
        item_evidence = _evidence_for_plan_item(evidence, item)
        # Prompt-caching layout: everything static-per-tier (role instructions,
        # citation rule, mandatory template, format invariants) lives in the
        # system message so it forms one stable cacheable prefix across the
        # continuation calls and across re-syncs of the same product. The user
        # message carries only the per-session variable payload (repo map,
        # summaries, signals, evidence), largest content last.
        resp = await chat.chat_markdown(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Synthesizer for Anvay. Generate one complete "
                        "generic product context Agent Skill body. Anvay adds frontmatter later. "
                        "Cite factual product claims with `[file: path:line]`. "
                        "Do not force citations onto procedural advice unless it names "
                        "a concrete product fact. Output Markdown body only. Use exact "
                        "required heading names; do not rename, skip, nest, or leave "
                        "any required section empty. Unknown headings are invalid. "
                        "Optional headings are allowed only when evidence supports them. "
                        "The `How To Use The Knowledge Base` section is supplied by Anvay "
                        "and spliced in deterministically — write a one-line placeholder "
                        "for it; do not author its routing content.\n\n"
                        "Citation rule: cite factual product claims only. Procedural "
                        "workflow, debugging, review, and anti-pattern guidance can be "
                        "uncited unless it names a concrete repo, file, command, API, "
                        "schema, model, entity, invariant, auth rule, or service.\n\n"
                        f"# Mandatory template\n{_template_for_tier(item.tier)}\n\n"
                        "Every required heading from the template must appear exactly "
                        "once. Use optional `## Testing Strategy` or `## Common Change "
                        "Patterns` only when you can cite evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Product: {state['product_id']}\nTopic: {state['topic']}\n"
                        f"Skill name: {item.name}\nTier: {item.tier}\nPurpose: {item.purpose}\n"
                        f"Activation description: {item.description}\n"
                        f"Required sections in order: {', '.join(required)}\n\n"
                        f"# Repo map\n{repo_map_block or '(none)'}\n\n"
                        f"# Graph and structural summaries (deterministic, from ingest)\n"
                        f"{context_pack}\n\n"
                        f"# Internal review and outcome signals\n"
                        f"{_signals_for_prompt(state.get('skill_signals') or [], skill_name=item.name)}\n\n"
                        f"# Evidence\n{evidence_for_prompt(item_evidence)}\n\n"
                        "Output the completed skill directly. The first line must be "
                        f"`# {item.name}`."
                    ),
                },
            ],
            max_tokens=4200,
            max_continuations=2,
        )
        usage_total = _add_usage(usage_total, resp.usage)
        # Splice the canonical KB-usage section in deterministically so the
        # routing table never drifts and costs no tokens (config.council).
        body = _merge_section_fill(
            resp.content.strip(), kb_usage_fill(state["product_id"])
        )
        drafts.append(
            SkillDraft(
                name=item.name,
                description=item.description,
                tier=item.tier,
                parent=item.parent,
                related=item.related,
                coverage=item.coverage,
                body=body.strip(),
            )
        )

    msg = DeliberationMessage(
        agent="synthesizer",
        timestamp=datetime.now(UTC).isoformat(),
        body="Synthesized product skill draft.",
    )
    return {
        "skill_drafts": drafts,
        "deliberation": [msg],
        "costs": [_cost("synthesizer", usage_total, chat)],
    }


def _blocking_skill_names(state: CouncilState) -> set[str]:
    """Skills whose latest eval verdict is `failed` — the repair targets."""
    latest: dict[str, str] = {}
    for result in state.get("eval_results") or []:
        latest[result.skill_name] = result.status
    return {name for name, status in latest.items() if status == "failed"}


async def repair_loop(
    state: CouncilState,
    *,
    chat: ChatClient,
    retrieval: RetrievalContext | None = None,
) -> dict:
    """One targeted repair pass per graph iteration, driven by eval verdicts.

    Only drafts whose latest eval status is `failed` are repaired; the graph's
    conditional edge (bounded by MAX_EVAL_REPAIR_ATTEMPTS) decides whether to
    loop back through eval or proceed. Never raises — an unfixable draft is
    passed through with its warnings and filtered at finalization."""
    evidence = state.get("evidence") or []
    supplemental_evidence: list[EvidenceChunk] = []
    repaired: list[SkillDraft] = []
    total = TokenUsage()
    messages: list[DeliberationMessage] = []
    blocking = _blocking_skill_names(state)
    # First iteration (no eval results yet): fall back to repairing every draft.
    repair_all = not state.get("eval_results")
    attempt_no = state.get("eval_repair_attempts", 0) + 1

    for draft in state.get("skill_drafts") or []:
        if not repair_all and draft.name not in blocking:
            repaired.append(draft)  # eval passed it — leave untouched
            continue
        body = draft.body
        if not body.strip():
            repaired.append(
                draft.model_copy(
                    update={
                        "repair_warnings": [*draft.repair_warnings, "empty draft body"],
                    }
                )
            )
            continue
        attempts = 0
        while attempts <= REPAIR_ATTEMPT_CAP:
            body = _ensure_fixed_h1(body, draft.name)
            body, _dropped = _strip_uncited_rules_if_safe(body)
            report = validate_skill_markdown(body, tier=draft.tier)
            citation_requirement_issues = _citation_requirement_issues(report, tier=draft.tier)
            if citation_requirement_issues:
                section_evidence = await _evidence_for_repair_issues(
                    state,
                    draft=draft,
                    body=body,
                    issues=citation_requirement_issues,
                    base_evidence=[*evidence, *supplemental_evidence],
                    retrieval=retrieval,
                )
                supplemental_evidence.extend(section_evidence)
                anchored = _anchor_missing_section_citations(
                    body,
                    issues=citation_requirement_issues,
                    evidence=[*evidence, *supplemental_evidence],
                )
                if anchored != body:
                    body = anchored
                    report = validate_skill_markdown(body, tier=draft.tier)
            citation_issues = _fabricated_citation_issues(
                body,
                tier=draft.tier,
                evidence=[*evidence, *supplemental_evidence],
            )
            if citation_issues:
                section_evidence = await _evidence_for_repair_issues(
                    state,
                    draft=draft,
                    body=body,
                    issues=citation_issues,
                    base_evidence=[*evidence, *supplemental_evidence],
                    retrieval=retrieval,
                )
                supplemental_evidence.extend(section_evidence)
                replaced = _replace_section_citations(
                    body,
                    issues=citation_issues,
                    evidence=[*evidence, *supplemental_evidence],
                )
                if replaced != body:
                    body = replaced
                    report = validate_skill_markdown(body, tier=draft.tier)
                    citation_issues = _fabricated_citation_issues(
                        body,
                        tier=draft.tier,
                        evidence=[*evidence, *supplemental_evidence],
                    )
            visible_evidence = _visible_repair_evidence(evidence, supplemental_evidence)
            body, citation_issues = _align_citations_to_evidence(
                body,
                tier=draft.tier,
                evidence=visible_evidence,
            )
            report = validate_skill_markdown(body, tier=draft.tier)
            if report.is_complete and not citation_issues:
                repaired.append(
                    draft.model_copy(
                        update={
                            "body": body,
                            "repair_attempts": draft.repair_attempts + attempts,
                        }
                    )
                )
                break
            if attempts == REPAIR_ATTEMPT_CAP:
                # Don't raise — pass the best-effort draft through with a warning
                # note. The graph-level eval→repair loop (bounded) re-evaluates;
                # finalizer filters drafts still failing after the loop budget.
                warning = (
                    f"incomplete after {REPAIR_ATTEMPT_CAP} inner attempts: "
                    f"{_format_missing(report)}{_format_citation_issues(citation_issues)}"
                )
                repaired.append(
                    draft.model_copy(
                        update={
                            "body": body,
                            "repair_attempts": draft.repair_attempts + attempts,
                            "repair_warnings": [*draft.repair_warnings, warning],
                        }
                    )
                )
                break
            attempts += 1
            issues = citation_issues if report.is_complete else _repair_issues(report, tier=draft.tier)
            section_evidence = await _evidence_for_repair_issues(
                state,
                draft=draft,
                body=body,
                issues=issues,
                base_evidence=[*evidence, *supplemental_evidence],
                retrieval=retrieval,
            )
            supplemental_evidence.extend(section_evidence)
            repair_evidence = _visible_repair_evidence(evidence, supplemental_evidence)
            resp = await chat.chat_markdown(
                [
                    {
                        "role": "system",
                        "content": (
                            "Repair incomplete Anvay skill sections. Output only "
                            "the requested `##` sections. Do not repeat any other "
                            "section. Use only citations from supplied evidence. "
                            "Use exact heading text.\n\n"
                            # Static template kept in the system prefix so it caches
                            # across the up-to-3 repair attempts (the user message —
                            # draft + evidence — changes each attempt).
                            f"# Required template\n{_template_for_tier(draft.tier)}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Skill tier: {draft.tier}\n"
                            f"Specific repairs:\n{_repair_issue_prompt(issues)}\n\n"
                            f"# Current draft\n{body}\n\n"
                            f"# Evidence\n{evidence_for_prompt(repair_evidence)}\n\n"
                            "Return only these sections, in this order: "
                            f"{', '.join(issue.output_name for issue in issues)}."
                        ),
                    },
                ],
                max_tokens=1800,
                max_continuations=1,
            )
            total = _add_usage(total, resp.usage)
            body = _merge_section_fill(body, resp.content)

        if attempts:
            messages.append(
                DeliberationMessage(
                    agent="repair",
                    timestamp=datetime.now(UTC).isoformat(),
                    body=f"Repaired `{draft.name}` in {attempts} attempt(s).",
                )
            )

    if not messages:
        messages.append(
            DeliberationMessage(
                agent="repair",
                timestamp=datetime.now(UTC).isoformat(),
                body="All skill drafts passed completeness validation without repair.",
            )
        )
    return {
        "skill_drafts": repaired,
        "eval_repair_attempts": attempt_no,
        "deliberation": messages,
        "costs": [_cost("repair", total, chat)] if total.total else [],
        **(
            {"evidence": _visible_repair_evidence(evidence, supplemental_evidence)}
            if supplemental_evidence
            else {}
        ),
    }


async def evaluator(
    state: CouncilState,
    *,
    config: AnvayConfig,
    chat: ChatClient | None = None,
    retrieval: RetrievalContext | None = None,
) -> dict:
    evidence = list(state.get("evidence") or [])
    plan = state.get("skill_plan") or catalog_plan(state["product_id"], state["topic"])
    signals = state.get("skill_signals") or []
    all_drafts: list[SkillDraft] = []
    results: list[SkillEvalResult] = []
    messages: list[DeliberationMessage] = []
    total = TokenUsage()

    # Deterministic-only by default: the checks run without an LLM. The bounded
    # faithfulness judge is opt-in via config (check-only — it annotates the
    # verdict, it does not repair). Repair is the repair node's job, reached via
    # the conditional edge on a `failed` verdict. All drafts (pass and fail) flow
    # to the finalizer, which filters by eval status; failing drafts route back
    # to repair while the graph-level attempt budget allows.
    gate_on = bool(getattr(getattr(config, "council", None), "faithfulness_gate", False))
    judge: ChatClient | None = chat if gate_on else None
    attempt = min(state.get("eval_repair_attempts", 0), 1)

    for draft in state.get("skill_drafts") or []:
        signal_ids = _signal_ids_for_skill(signals, skill_name=draft.name)
        result = await evaluate_skill_draft(
            draft=draft,
            evidence=evidence,
            plan=plan,
            chat=judge,
            attempt=attempt,
            signals_used=signal_ids,
            indexer=getattr(retrieval, "indexer", None),
            product_id=state["product_id"],
        )
        results.append(result)
        all_drafts.append(draft)
        if result.status not in {"passed", "repaired"}:
            messages.append(
                DeliberationMessage(
                    agent="skill-eval",
                    timestamp=datetime.now(UTC).isoformat(),
                    body=(
                        f"`{draft.name}` failed deterministic skill eval. "
                        f"Fix instructions:\n{failure_brief(result)}"
                    ),
                )
            )

    ok = sum(1 for result in results if result.status in {"passed", "repaired"})
    messages.insert(
        0,
        DeliberationMessage(
            agent="skill-eval",
            timestamp=datetime.now(UTC).isoformat(),
            body=f"Skill quality eval passed {ok}/{len(results)} draft(s).",
        ),
    )
    return {
        "skill_drafts": all_drafts,
        "eval_results": results,
        "deliberation": messages,
        "costs": [_cost("skill-eval", total, chat)] if total.total else [],
    }


async def finalizer(state: CouncilState) -> dict:
    evidence = list(state.get("evidence") or [])
    proposals: list[SkillProposal] = []
    now = datetime.now(UTC).isoformat()
    # Latest verdict per skill (eval_results is append-only across repair loops).
    latest_eval: dict[str, SkillEvalResult] = {}
    for result in state.get("eval_results", []):
        latest_eval[result.skill_name] = result
    passed_names = {
        name for name, r in latest_eval.items() if r.status in {"passed", "repaired"}
    }
    for draft in state.get("skill_drafts") or []:
        # Only finalize drafts that passed eval. Drafts still failing after the
        # bounded eval→repair loop are dropped (not raised) so a single bad
        # draft can't sink a batch of otherwise-good proposals.
        if latest_eval and draft.name not in passed_names:
            continue
        body = _ensure_fixed_h1(draft.body, draft.name)
        report = validate_skill_markdown(body, tier=draft.tier)
        if not report.is_complete:
            continue
        parsed = parse_skill_markdown(body, fallback_name=draft.name, evidence=evidence)
        eval_result = latest_eval.get(draft.name)
        paragraphs = max(1, parsed.body.count("\n\n") + 1)
        confidence = compute_confidence(
            citations=parsed.citations,
            paragraphs=paragraphs,
            revision_count=min(draft.repair_attempts, 2),
        )
        proposals.append(
            SkillProposal(
                id=str(uuid.uuid4()),
                name=draft.name,
                description=draft.description,
                tier=draft.tier,
                parent=draft.parent,
                related=draft.related,
                coverage=SkillCoverage(**(draft.coverage or {})),
                body=parsed.body,
                citations=parsed.citations,
                confidence=confidence,
                eval_status=(eval_result.status if eval_result else "not_run"),
                eval_summary=(eval_result.summary if eval_result else ""),
                eval_failures=(eval_result.failures if eval_result else []),
                quality_score=(eval_result.quality_score if eval_result else 0.0),
                signals_used=(eval_result.signals_used if eval_result else []),
                status="pending",
                created_at=now,
            )
        )

    if not proposals:
        raise CouncilIncompleteSkill(
            user_message="Council stopped because no complete skill proposals were produced.",
            detail="finalizer received no skill drafts",
        )
    primary = next((p for p in proposals if p.tier == "product_master"), proposals[0])
    msg = DeliberationMessage(
        agent="finalizer",
        timestamp=datetime.now(UTC).isoformat(),
        body=f"Finalized {len(proposals)} proposal(s) for human review.",
        cite_ids=[c.id for p in proposals for c in p.citations if c.id][:20],
    )
    return {
        "proposals": proposals,
        "proposal": primary,
        "proposal_id": primary.id,
        "deliberation": [msg],
    }


def _summaries_block(result: object, *, limit: int = 12) -> str:
    """Render the structural/graph summary candidates from an EvidenceSet.

    These are the deterministic summaries `anvay/ingest/summaries.py` builds at
    ingest (node labels, entities, flows, runtime/config relations, doc/test
    links). They ground the synthesizer's structural sections without an LLM
    expert re-deriving them. Falls back to `(none)` when the summary channel
    surfaced nothing.
    """
    candidates = getattr(result, "candidates", None)
    if not candidates:
        return "(none)"
    lines: list[str] = []
    for candidate in candidates:
        if getattr(candidate, "channel", "") != "summary":
            continue
        anchor = getattr(candidate, "anchor", "") or getattr(candidate, "chunk_id", "")
        excerpt = (getattr(candidate, "excerpt", "") or "").strip()
        if not excerpt:
            continue
        lines.append(f"[{anchor}] {excerpt}")
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "(none)"


def _drafts_for_prompt(drafts: list[SkillDraft]) -> str:
    return "\n\n".join(
        f"## {draft.name} ({draft.tier})\n{draft.body[:2200]}" for draft in drafts
    )


async def _repair_eval_failure(
    *,
    draft: SkillDraft,
    result: SkillEvalResult,
    evidence: list[EvidenceChunk],
    chat: ChatClient,
) -> tuple[str, TokenUsage]:
    resp = await chat.chat_markdown(
        [
            {
                "role": "system",
                "content": (
                    "Repair one Anvay Agent Skill after quality eval failure. "
                    "Output the complete Markdown body only. Keep the exact title, "
                    "exact required headings, and real `[file: path:line]` citations.\n\n"
                    # Static template in the system prefix for cache reuse.
                    f"# Required template\n{_template_for_tier(draft.tier)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Skill name: {draft.name}\nTier: {draft.tier}\n"
                    f"Description: {draft.description}\n\n"
                    f"# Eval failures\n{failure_brief(result)}\n\n"
                    f"# Current draft\n{draft.body}\n\n"
                    f"# Evidence\n{evidence_for_prompt(evidence)}\n\n"
                    "Return a complete repaired body. The first line must be "
                    f"`# {draft.name}`."
                ),
            },
        ],
        max_tokens=3400 if draft.tier == "product_master" else 2600,
        max_continuations=1,
    )
    return resp.content.strip(), resp.usage


def _signals_for_prompt(signals: list[dict], *, skill_name: str) -> str:
    relevant = [
        s
        for s in signals
        if not s.get("skill_name") or s.get("skill_name") == skill_name
    ][:6]
    if not relevant:
        return "(none)"
    lines = []
    for signal in relevant:
        source = signal.get("source_type", "signal")
        text = str(signal.get("text", "")).strip().replace("\n", " ")
        lines.append(f"- {source}: {text[:240]}")
    return "\n".join(lines)


def _signal_ids_for_skill(signals: list[dict], *, skill_name: str) -> list[str]:
    return [
        str(s.get("id"))
        for s in signals
        if s.get("id") and (not s.get("skill_name") or s.get("skill_name") == skill_name)
    ][:8]


def _select_evidence(chunks: list[EvidenceChunk], *, limit: int) -> list[EvidenceChunk]:
    if limit <= 0:
        return []
    by_id: dict[str, EvidenceChunk] = {}
    for chunk in chunks:
        current = by_id.get(chunk.chunk_id)
        if current is None or chunk.score > current.score:
            by_id[chunk.chunk_id] = chunk

    ranked = sorted(by_id.values(), key=lambda e: e.score, reverse=True)
    selected: list[EvidenceChunk] = []
    used_files: set[str] = set()
    for chunk in ranked:
        if len(selected) >= limit:
            break
        if chunk.file in used_files:
            continue
        selected.append(chunk)
        used_files.add(chunk.file)
    for chunk in ranked:
        if len(selected) >= limit:
            break
        if chunk not in selected:
            selected.append(chunk)
    return selected


def _evidence_for_plan_item(
    evidence: list[EvidenceChunk], item: SkillPlanItem, *, limit: int = 20
) -> list[EvidenceChunk]:
    terms = {
        token
        for token in " ".join(
            [
                item.name,
                item.description,
                item.purpose,
                " ".join(str(t) for t in item.coverage.get("topics", [])),
            ]
        ).lower().replace("-", " ").split()
        if len(token) >= 4
    }

    def score(chunk: EvidenceChunk) -> tuple[int, float]:
        haystack = f"{chunk.file} {chunk.excerpt}".lower()
        return (sum(1 for term in terms if term in haystack), chunk.score)

    ranked = sorted(evidence, key=score, reverse=True)
    return _select_evidence(ranked, limit=limit)


def _ensure_fixed_h1(body: str, name: str) -> str:
    lines = body.strip().splitlines()
    if not lines:
        return f"# {name}\n"
    if lines[0].startswith("# "):
        lines[0] = f"# {name}"
        return "\n".join(lines).strip() + "\n"
    return f"# {name}\n\n{body.strip()}\n"


def _template_for_tier(tier: str) -> str:
    sections = required_sections_for_tier(tier)
    lines = ["# {skill-name}", ""]
    for title in sections:
        lines.append(f"## {title}")
        if title == "Use This Skill When":
            lines.append("One concise activation paragraph.")
        elif title == "Product Snapshot":
            lines.append(
                "2-3 sentences covering: what the product does, who uses it, and "
                "the primary runtime (language, framework, deployment target). "
                "[file: path:line]"
            )
        elif title == "Product Language":
            lines.append(
                "Bullet list of domain-specific terms and their precise meanings in this product. "
                "Each term on its own line: `- **Term**: definition. [file: path:line]` "
                "(3-8 terms minimum)."
            )
        elif title == "Capabilities And Workflows":
            lines.append(
                "Numbered list of the product's primary capabilities. "
                "Each item: `1. Capability name - one-line description. [file: path:line]` "
                "(at least 3 capabilities)."
            )
        elif title == "System Map":
            lines.append(
                "Bullet list of the named system components (services, repos, processes). "
                "Each item: `- **ComponentName**: role, runtime boundary, and key file or entrypoint. "
                "[file: path:line]`"
            )
        elif title == "Data Model":
            lines.append(
                "Bullet list of core entities/schemas. "
                "Each item: `- **EntityName**: fields that matter, storage backend, and any "
                "product invariants on this entity. [file: path:line]`"
            )
        elif title == "Interfaces And Contracts":
            lines.append(
                "Bullet list of external-facing interfaces (REST routes, MCP tools, gRPC, queues, etc.). "
                "Each item: `- `METHOD /path` or `tool-name`: purpose and key contract. [file: path:line]`"
            )
        elif title == "Invariants And Constraints":
            lines.append(
                "Numbered list of hard product invariants that must never be violated. "
                "Each item: `1. Invariant statement. [file: path:line]` "
                "(at least 3 invariants)."
            )
        elif title == "How To Use The Knowledge Base":
            lines.append(
                "Explain when and how to query the product knowledge base versus trusting this skill. "
                "State which MCP retrieval tools are available (e.g. find_skills, query_code_context, "
                "hybrid_search_corpus) and when each is appropriate: find_skills for curated guidance, "
                "query_code_context for symbol/definition lookup, hybrid_search_corpus for open-ended "
                "corpus search. Note when the skill alone is sufficient and when fresh retrieval is required."
            )
        elif title == "How To Work In This Product":
            lines.append(
                "Numbered steps or short bullet list covering: local setup, key commands to run, "
                "PR/review conventions, and any product-specific contribution rules. "
                "Cite concrete commands or config files where known."
            )
        elif title == "Security And Secrets":
            lines.append(
                "Bullet list of security patterns and secret-handling rules. "
                "Each item: `- Rule or pattern. [file: path:line]` "
                "Do not cite general best-practices — only product-specific rules."
            )
        elif title == "Known Traps":
            lines.append(
                "Bullet list of concrete gotchas, footguns, and common mistakes in this product. "
                "Each item: `- **Trap**: what goes wrong and why. [file: path:line]` "
                "(at least 2 traps)."
            )
        elif title == "Freshness And Evidence":
            lines.append(
                "Bullet list of freshness signals for this skill: last-known stable state, "
                "which files change most often, and what to re-verify before relying on this skill. "
                "[file: path:line]"
            )
        elif title in {"Anti-patterns", "Gotchas", "Review Checklist"}:
            lines.append("- Concrete product-aware guidance; cite only if naming a specific product fact.")
        elif title in {
            "Before Editing",
            "Debugging Playbook",
            "Grounding Workflow",
            "Skill Map",
            "When Evidence Is Missing",
        }:
            lines.append("One concise operational paragraph.")
        else:
            lines.append("One concise product-aware paragraph; cite factual product claims only.")
    return "\n".join(lines)


def _format_missing(report) -> str:
    parts = list(report.missing_sections) + list(report.short_sections)
    return ", ".join(parts) if parts else "(none)"


@dataclass(frozen=True)
class _RepairIssue:
    output_name: str
    instruction: str
    section_title: str


def _next_repair_issue(report, *, tier: str) -> _RepairIssue:
    if report.missing_sections:
        section = report.missing_sections[0]
        if section == "title":
            return _RepairIssue(
                output_name="the `# ...` title line",
                instruction="Add the missing top-level `# ...` title line only.",
                section_title="title",
            )
        title = _canonical_section_title(section, tier=tier)
        return _RepairIssue(
            output_name=f"`## {title}`",
            instruction=(
                f"Add only the missing `## {title}` section. "
                "Make it concise, concrete, and cite evidence where factual."
            ),
            section_title=title,
        )
    if report.short_sections:
        raw = report.short_sections[0]
        title = _canonical_section_title(raw.split("(", 1)[0].strip(), tier=tier)
        if "needs citation" in raw.lower():
            instruction = (
                f"Rewrite only `## {title}` so it includes at least one "
                "`[file: path:line]` citation from the evidence."
            )
        else:
            instruction = (
                f"Rewrite only `## {title}` to satisfy this exact issue: {raw}. "
                "Keep it short and evidence-backed."
            )
        return _RepairIssue(
            output_name=f"`## {title}`",
            instruction=instruction,
            section_title=title,
        )
    return _RepairIssue(output_name="nothing", instruction="No repair is needed.", section_title="")


def _repair_issues(report, *, tier: str) -> list[_RepairIssue]:
    issues: list[_RepairIssue] = []
    for section in report.missing_sections:
        if section == "title":
            issues.append(
                _RepairIssue(
                    output_name="the `# ...` title line",
                    instruction="Add the missing top-level `# ...` title line only.",
                    section_title="title",
                )
            )
            continue
        title = _canonical_section_title(section, tier=tier)
        issues.append(
            _RepairIssue(
                output_name=f"`## {title}`",
                instruction=(
                    f"Add only the missing `## {title}` section. Make it concise, "
                    "concrete, and cite evidence where factual."
                ),
                section_title=title,
            )
        )
    for raw in report.short_sections:
        title = _canonical_section_title(raw.split("(", 1)[0].strip(), tier=tier)
        if "needs citation" in raw.lower():
            instruction = (
                f"Rewrite `## {title}` so it includes at least one "
                "`[file: path:line]` citation from the evidence."
            )
        else:
            instruction = (
                f"Rewrite `## {title}` to satisfy this exact issue: {raw}. "
                "Keep it short and evidence-backed."
            )
        issue = _RepairIssue(
            output_name=f"`## {title}`",
            instruction=instruction,
            section_title=title,
        )
        if issue not in issues:
            issues.append(issue)
    return issues or [_next_repair_issue(report, tier=tier)]


def _repair_issue_prompt(issues: list[_RepairIssue]) -> str:
    return "\n".join(f"- {issue.instruction}" for issue in issues)


def _citation_requirement_issues(report, *, tier: str) -> list[_RepairIssue]:
    issues: list[_RepairIssue] = []
    for raw in report.short_sections:
        if "needs citation" not in raw.lower():
            continue
        title = _canonical_section_title(raw.split("(", 1)[0].strip(), tier=tier)
        issues.append(
            _RepairIssue(
                output_name=f"`## {title}`",
                instruction=(
                    f"Find indexed evidence for `## {title}` and add a real "
                    "`[file: path:line]` citation."
                ),
                section_title=title,
            )
        )
    return issues


def _anchor_missing_section_citations(
    body: str,
    *,
    issues: list[_RepairIssue],
    evidence: list[EvidenceChunk],
) -> str:
    if not issues or not evidence:
        return body
    lines = body.rstrip().splitlines()
    issue_titles = {issue.section_title.lower() for issue in issues}
    changed = False
    for heading, (start, end) in reversed(_h2_block_ranges(lines)):
        if heading.lower() not in issue_titles:
            continue
        block = "\n".join(lines[start:end])
        if "[file:" in block.lower():
            continue
        chosen = _best_section_evidence(heading, block, evidence)
        if chosen is None:
            continue
        citation = f"[file: {chosen.file}:{chosen.line}]"
        insert_at = _citation_insert_line(lines, start, end)
        if insert_at is None:
            lines.insert(end, f"- See indexed evidence {citation}.")
        else:
            lines[insert_at] = f"{lines[insert_at].rstrip()} {citation}"
        changed = True
    return "\n".join(lines).strip() + "\n" if changed else body


def _replace_section_citations(
    body: str,
    *,
    issues: list[_RepairIssue],
    evidence: list[EvidenceChunk],
) -> str:
    if not issues or not evidence:
        return body
    lines = body.rstrip().splitlines()
    issue_titles = {issue.section_title.lower() for issue in issues}
    changed = False
    citation_re = re.compile(r"\[file:\s*.+?:\d+\]", re.IGNORECASE)
    for heading, (start, end) in reversed(_h2_block_ranges(lines)):
        if heading.lower() not in issue_titles:
            continue
        block = "\n".join(lines[start:end])
        chosen = _best_section_evidence(heading, block, evidence)
        if chosen is None:
            continue
        for idx in range(start + 1, end):
            cleaned = citation_re.sub("", lines[idx]).rstrip()
            if cleaned != lines[idx]:
                lines[idx] = cleaned
                changed = True
        citation = f"[file: {chosen.file}:{chosen.line}]"
        insert_at = _citation_insert_line(lines, start, end)
        if insert_at is None:
            lines.insert(end, f"- See indexed evidence {citation}.")
        else:
            lines[insert_at] = f"{lines[insert_at].rstrip()} {citation}"
        changed = True
    return "\n".join(lines).strip() + "\n" if changed else body


def _best_section_evidence(
    heading: str, block: str, evidence: list[EvidenceChunk]
) -> EvidenceChunk | None:
    terms = {
        token
        for token in re.findall(r"[a-z0-9_./:-]+", f"{heading} {block}".lower())
        if len(token) >= 4
    }
    if not terms:
        return evidence[0] if evidence else None

    def score(chunk: EvidenceChunk) -> tuple[int, float]:
        haystack = f"{chunk.file} {chunk.excerpt}".lower()
        return (sum(1 for term in terms if term in haystack), chunk.score)

    return max(evidence, key=score, default=None)


def _citation_insert_line(lines: list[str], start: int, end: int) -> int | None:
    fallback: int | None = None
    for idx in range(start + 1, end):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if stripped.startswith(("-", "*")) or _starts_numbered(stripped):
            return idx
        if fallback is None:
            fallback = idx
    return fallback


def _fabricated_citation_issues(
    body: str, *, tier: str, evidence: list[EvidenceChunk]
) -> list[_RepairIssue]:
    valid = {(chunk.file, int(chunk.line)) for chunk in evidence}
    if not valid:
        return []
    lines = body.splitlines()
    issues: list[_RepairIssue] = []
    for heading, (start, end) in _h2_block_ranges(lines):
        block = "\n".join(lines[start:end])
        bad = []
        for match in re.finditer(r"\[file:\s*(.+?):(\d+)\]", block, re.IGNORECASE):
            try:
                anchor = (match.group(1).strip(), int(match.group(2)))
            except ValueError:
                continue
            if anchor not in valid:
                bad.append(f"{anchor[0]}:{anchor[1]}")
        if not bad:
            continue
        title = _canonical_section_title(heading, tier=tier)
        issues.append(
            _RepairIssue(
                output_name=f"`## {title}`",
                instruction=(
                    f"Rewrite `## {title}` because it cites anchors not in evidence: "
                    f"{', '.join(bad[:4])}. Use only supplied evidence citations."
                ),
                section_title=title,
            )
        )
    return issues


def _format_citation_issues(issues: list[_RepairIssue]) -> str:
    if not issues:
        return ""
    return "; fabricated citations: " + ", ".join(issue.section_title for issue in issues)


async def _evidence_for_repair_issues(
    state: CouncilState,
    *,
    draft: SkillDraft,
    body: str,
    issues: list[_RepairIssue],
    base_evidence: list[EvidenceChunk],
    retrieval: RetrievalContext | None,
) -> list[EvidenceChunk]:
    if retrieval is None:
        return []

    seen = {(e.chunk_id, e.file, e.line) for e in base_evidence}
    additions: list[EvidenceChunk] = []
    for issue in issues:
        if issue.section_title == "title":
            continue
        query = _repair_evidence_query(state, draft=draft, body=body, issue=issue)
        found = await grep_indexed_chunks(
            indexer=retrieval.indexer,
            product_id=state["product_id"],
            query=query,
            limit=6,
        )
        if len(found) < 2:
            try:
                result = await retrieve(
                    ctx=retrieval,
                    product_id=state["product_id"],
                    query=query,
                    top_k=6,
                    mode="auto",
                )
                found.extend(hits_to_evidence(result.hits, limit=6))
            except Exception as e:
                log.warning("repair evidence retrieval failed for %s: %s", draft.name, e)
        if not found:
            found.extend(
                await sample_indexed_chunks(
                    indexer=retrieval.indexer,
                    product_id=state["product_id"],
                    limit=4,
                )
            )
        for chunk in found:
            key = (chunk.chunk_id, chunk.file, chunk.line)
            if key in seen:
                continue
            seen.add(key)
            additions.append(chunk)
    return _select_evidence(additions, limit=24)


def _repair_evidence_query(
    state: CouncilState,
    *,
    draft: SkillDraft,
    body: str,
    issue: _RepairIssue,
) -> str:
    suffix = next(
        (item.retrieval_suffix for item in SKILL_CATALOG if item.tier == draft.tier),
        "",
    )
    section_text = _section_text(body, issue.section_title)
    return " ".join(
        part
        for part in (
            state.get("topic", ""),
            draft.name,
            draft.description,
            draft.tier,
            issue.section_title,
            issue.instruction,
            suffix,
            section_text[:700],
        )
        if part
    )


def _section_text(markdown: str, title: str) -> str:
    if not title or title == "title":
        return ""
    lines = markdown.splitlines()
    for heading, (start, end) in _h2_block_ranges(lines):
        if heading.lower() == title.lower():
            return "\n".join(lines[start + 1 : end]).strip()
    return ""


def _canonical_section_title(raw: str, *, tier: str) -> str:
    required = required_sections_for_tier(tier)
    key = raw.strip().lower()
    for title in required:
        if title.lower() == key:
            return title
    if key.startswith("rules"):
        return "Rules"
    if key.startswith("anti-patterns"):
        return "Anti-patterns"
    return raw.strip().title()


def _merge_section_fill(current: str, fill: str) -> str:
    fill = fill.strip()
    if not fill:
        return current
    prepended_title = False
    if not current.lstrip().startswith("# "):
        h1 = next((line.strip() for line in fill.splitlines() if line.startswith("# ")), "")
        if h1:
            current = f"{h1}\n\n{current.strip()}"
            prepended_title = True
    fill_blocks = _h2_blocks(fill)
    if not fill_blocks:
        if prepended_title:
            return current.rstrip() + "\n"
        return current.rstrip() + "\n\n" + fill + "\n"

    lines = current.rstrip().splitlines()
    current_blocks = _h2_block_ranges(lines)
    replaced: set[str] = set()
    for heading, block in fill_blocks:
        key = heading.lower()
        ranges = [rng for title, rng in current_blocks if title.lower() == key]
        if not ranges:
            continue
        start, end = ranges[-1]
        lines[start:end] = block
        replaced.add(key)
        current_blocks = _h2_block_ranges(lines)

    additions = [block for heading, block in fill_blocks if heading.lower() not in replaced]
    for block in additions:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    return "\n".join(lines).strip() + "\n"


def _visible_repair_evidence(
    evidence: list[EvidenceChunk],
    supplemental_evidence: list[EvidenceChunk],
) -> list[EvidenceChunk]:
    limit = 40 if supplemental_evidence else EVIDENCE_CHUNKS_PER_SESSION_CAP
    return _select_evidence([*evidence, *supplemental_evidence], limit=limit)


def _align_citations_to_evidence(
    body: str,
    *,
    tier: str,
    evidence: list[EvidenceChunk],
) -> tuple[str, list[_RepairIssue]]:
    issues = _fabricated_citation_issues(body, tier=tier, evidence=evidence)
    if not issues:
        return body, []
    replaced = _replace_section_citations(body, issues=issues, evidence=evidence)
    return replaced, _fabricated_citation_issues(replaced, tier=tier, evidence=evidence)


def _strip_uncited_rules_if_safe(md: str) -> tuple[str, int]:
    stripped, dropped = strip_uncited_rules(md)
    if dropped == 0:
        return stripped, dropped
    if _count_cited_rule_items(stripped) >= 3:
        return stripped, dropped
    return md, 0


def _count_cited_rule_items(md: str) -> int:
    lines = md.splitlines()
    rule_range = next(
        (
            (start, end)
            for heading, (start, end) in _h2_block_ranges(lines)
            if heading.lower() == "rules"
        ),
        None,
    )
    if not rule_range:
        return 0
    start, end = rule_range
    return sum(
        1
        for line in lines[start + 1 : end]
        if (
            line.lstrip().startswith(("-", "*"))
            or _starts_numbered(line)
        )
        and "[file:" in line.lower()
    )


def _h2_blocks(markdown: str) -> list[tuple[str, list[str]]]:
    lines = markdown.splitlines()
    ranges = _h2_block_ranges(lines)
    return [(heading, lines[start:end]) for heading, (start, end) in ranges]


def _h2_block_ranges(lines: list[str]) -> list[tuple[str, tuple[int, int]]]:
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            starts.append((line[3:].strip(), idx))
    ranges: list[tuple[str, tuple[int, int]]] = []
    for idx, (heading, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        ranges.append((heading, (start, end)))
    return ranges


def _anchor_uncited_sections(
    body: str, *, tier: str, evidence: list[EvidenceChunk]
) -> str:
    """Legacy no-op: missing citations must be repaired by the model or fail."""
    _ = tier, evidence
    return body


def _starts_numbered(line: str) -> bool:
    stripped = line.lstrip()
    dot = stripped.find(".")
    return dot > 0 and stripped[:dot].isdigit()


def _cost(agent: str, usage: TokenUsage, chat: ChatClient) -> AgentCost:
    return AgentCost(
        agent=agent,
        prompt_tokens=usage.prompt,
        completion_tokens=usage.completion,
        model=chat.model,
    )


def _retrieval_query(topic: str, *, suffix: str = "", limit: int = 900) -> str:
    """Keep retrieval queries small enough for local embedding servers.

    Revision topics may include a complete prior skill draft. The LLM still
    receives that context later, but retrieval only needs the SME request and
    high-signal terms.
    """
    text = topic.strip()
    if "\nPrevious draft:" in text:
        text = text.split("\nPrevious draft:", 1)[0].strip()
    if suffix:
        text = f"{text}\n{suffix.strip()}"
    return text[:limit].strip() or suffix.strip() or topic[:limit].strip()


async def _retrieve_skill_evidence(
    *,
    retrieval: RetrievalContext,
    product_id: str,
    query: str,
    top_k: int,
    graph_store: object | None = None,
    skills: list[Skill] | None = None,
):
    return await retrieve_evidence(
        ctx=retrieval,
        product_id=product_id,
        query=query,
        top_k=top_k,
        mode="auto",
        graph_store=graph_store,
        skills=skills or [],
    )


def _skill_result_to_evidence(result, *, limit: int) -> list[EvidenceChunk]:
    if hasattr(result, "candidates"):
        return evidence_set_to_evidence(result, limit=limit)
    return hits_to_evidence(result.hits, limit=limit)


async def _approved_skills(config: AnvayConfig | None, product_id: str) -> list[Skill]:
    if config is None:
        return []
    try:
        root = Path(config.hierarchy_root)
        if not root.is_absolute():
            root = Path.cwd() / root

        def _load() -> list[Skill]:
            return [
                skill
                for skill in SkillStore(root).iter_skills()
                if skill.product == product_id
            ]

        return await asyncio.to_thread(_load)
    except Exception as e:
        log.debug("skill lookup skipped for evidence retrieval: %s", e)
        return []


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt=left.prompt + right.prompt,
        completion=left.completion + right.completion,
    )


def _no_evidence_error(result, config: AnvayConfig) -> CouncilNoEvidence:
    gate = config.ingestion.quality_gate_threshold
    seed_count = getattr(result, "seed_count", 0)
    filtered_by_gate = getattr(result, "filtered_by_gate", 0)
    if result is not None and seed_count and filtered_by_gate:
        best = getattr(result, "best_score_before_gate", None)
        best_text = "unknown" if best is None else f"{best:.3g}"
        return CouncilNoEvidence(
            user_message=(
                "Council stopped before planning because the retrieval quality gate "
                "filtered every candidate evidence chunk."
            ),
            detail=(
                f"quality_gate_threshold={gate:g} filtered all reranked hits "
                f"(best_score={best_text})"
            ),
        )
    coverage = getattr(result, "coverage", None)
    if coverage is not None and getattr(coverage, "missing_facets", None):
        return CouncilNoEvidence(
            user_message=(
                "Council stopped before planning because the evidence engine could not "
                "cover the requested product context."
            ),
            detail=f"missing evidence facets: {', '.join(coverage.missing_facets)}",
        )
    return CouncilNoEvidence(
        user_message=(
            "Council stopped before planning because no evidence chunks were found. "
            "Sync source content, then run the council again."
        ),
        detail="retrieval found no candidate chunks; sync source content before running council",
    )
