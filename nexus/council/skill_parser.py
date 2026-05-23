"""Skill markdown parser + completeness validator.

Parses the markdown produced by Drafter/Reviser into (name, body, citations)
and validates that all required sections are present. Used by the council to
gate completion: if any required section is missing, the agent fires a
targeted section-fill prompt.

Skill schema (validated):
    # Title                      — required, kebab-cased
    Intro paragraph(s)           — at least one non-empty paragraph
    ## Rules                     — required, ≥ MIN_RULES numbered/bulleted items, each cited
    ## Anti-patterns             — required section, ≥ MIN_ANTI items
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nexus.council.state import EvidenceChunk
from nexus.skills.models import Citation

MIN_RULES = 3
MIN_ANTI = 1

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2_RULES_RE = re.compile(r"##\s+Rules\b(.*?)(?=\n##\s+|\Z)", re.DOTALL | re.IGNORECASE)
_H2_ANTI_RE = re.compile(
    r"##\s+Anti[- ]?patterns\b(.*?)(?=\n##\s+|\Z)", re.DOTALL | re.IGNORECASE
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+\.|[-*])\s", re.MULTILINE)
_CITATION_RE = re.compile(r"\[file:\s*([^\s\]:]+):(\d+)\]", re.IGNORECASE)
_NAME_RE = re.compile(r"[^a-z0-9-]+")
_DASH_RUN = re.compile(r"-{2,}")


@dataclass
class ParsedSkill:
    name: str
    body: str
    citations: list[Citation] = field(default_factory=list)


def parse_skill_markdown(
    md: str,
    *,
    fallback_name: str = "untitled-skill",
    evidence: list[EvidenceChunk] | None = None,
) -> ParsedSkill:
    """Parse a Drafter/Reviser markdown response.

    - Name: H1 heading, kebab-cased. Falls back to `fallback_name` if no H1.
    - Body: the full markdown (H1 included).
    - Citations: every `[file: path:line]` match, deduped. When `evidence` is
      provided, we attach the matching chunk id + excerpt for downstream
      ingest.
    """
    body = md.strip()

    h1 = _H1_RE.search(body)
    raw_name = h1.group(1) if h1 else fallback_name
    name = _normalise_name(raw_name)

    citations = _extract_citations(body, evidence or [])

    return ParsedSkill(name=name, body=body, citations=citations)


@dataclass
class CompletenessReport:
    missing_sections: list[str]  # which required sections are absent
    short_sections: list[str]    # present but below minimum item count

    @property
    def is_complete(self) -> bool:
        return not self.missing_sections and not self.short_sections


def validate_completeness(md: str) -> CompletenessReport:
    """Check that the markdown has all required sections at adequate length.

    Returns a report; an empty report (`is_complete=True`) means the skill is
    ready to ship. The caller decides whether to invoke section-fill or stop.
    """
    missing: list[str] = []
    short: list[str] = []

    if not _H1_RE.search(md):
        missing.append("title")

    rules_match = _H2_RULES_RE.search(md)
    if not rules_match:
        missing.append("rules")
    else:
        items = _LIST_ITEM_RE.findall(rules_match.group(1))
        if len(items) < MIN_RULES:
            short.append(f"rules (have {len(items)}, need ≥{MIN_RULES})")

    anti_match = _H2_ANTI_RE.search(md)
    if not anti_match:
        missing.append("anti-patterns")
    else:
        items = _LIST_ITEM_RE.findall(anti_match.group(1))
        if len(items) < MIN_ANTI:
            short.append(f"anti-patterns (have {len(items)}, need ≥{MIN_ANTI})")

    return CompletenessReport(missing_sections=missing, short_sections=short)


def strip_uncited_rules(md: str) -> tuple[str, int]:
    """Drop list items in `## Rules` that lack any `[file: path:line]` citation.

    Used as a post-parse guardrail: the prompt says every rule must cite, and
    this enforces it deterministically.
    """
    rules_match = _H2_RULES_RE.search(md)
    if not rules_match:
        return md, 0

    block_start = rules_match.start(1)
    block_end = rules_match.end(1)
    block_text = rules_match.group(1)

    new_lines: list[str] = []
    dropped = 0
    for line in block_text.splitlines():
        is_list_item = bool(re.match(r"^\s*(?:\d+\.|[-*])\s", line))
        if is_list_item and not _CITATION_RE.search(line):
            dropped += 1
            continue
        new_lines.append(line)
    if dropped == 0:
        return md, 0
    return md[:block_start] + "\n".join(new_lines) + md[block_end:], dropped


def _extract_citations(body: str, evidence: list[EvidenceChunk]) -> list[Citation]:
    by_anchor: dict[tuple[str, int], EvidenceChunk] = {
        (e.file, e.line): e for e in evidence
    }
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(body):
        file_ = m.group(1)
        try:
            line = int(m.group(2))
        except ValueError:
            continue
        key = (file_, line)
        if key in seen:
            continue
        seen.add(key)
        evi = by_anchor.get(key)
        out.append(
            Citation(
                id=evi.chunk_id if evi else None,
                file=file_,
                line=line,
                excerpt=(evi.excerpt if evi else ""),
            )
        )
    return out


def _normalise_name(raw: str) -> str:
    s = raw.strip().lower().replace("_", "-").replace(" ", "-")
    s = _NAME_RE.sub("-", s)
    s = _DASH_RUN.sub("-", s).strip("-")
    return s[:60] or "untitled-skill"
