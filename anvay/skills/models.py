"""Skills data model.

A skill is the unit of validated guidance Anvay serves to AI agents. Skills
live as Markdown files with YAML frontmatter, version-controlled in a Git repo.
The product remains the tenancy boundary; tier metadata describes how skills
compose inside that product.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: str | None = None
    file: str
    line: int
    excerpt: str = ""


class AppliesTo(BaseModel):
    files: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)


SkillTier = Literal[
    "product_master",
    "application",
    "domain",
    "interface",
    "tech_stack",
    "quality_security",
]


class SkillCoverage(BaseModel):
    repos: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


EvalStatus = Literal["not_run", "passed", "failed", "repaired"]


class Provenance(BaseModel):
    council_session: str | None = None
    validated_by: str
    validated_at: str  # ISO-8601
    evidence_chunks: list[str] = Field(default_factory=list)
    adversary_critique: str | None = None
    revision_count: Literal[0, 1] = 0


class Skill(BaseModel):
    """One curated, human-approved skill scoped to a single product."""

    name: str
    description: str = ""
    product: str
    tier: SkillTier = "domain"
    parent: str | None = None
    related: list[str] = Field(default_factory=list)
    coverage: SkillCoverage = Field(default_factory=SkillCoverage)
    version: int = 1
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    eval_status: EvalStatus = "not_run"
    eval_summary: str = ""
    eval_failures: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    signals_used: list[str] = Field(default_factory=list)
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    provenance: Provenance
    body: str = ""  # markdown body, not in frontmatter

    @property
    def id(self) -> str:
        return f"{self.product}/{self.name}"


class Critique(BaseModel):
    severity: Literal["blocking", "major", "minor"]
    issues: list[dict] = Field(default_factory=list)
    recommendation: str = ""


class SkillProposal(BaseModel):
    """Council output queued for human review."""

    id: str
    name: str
    description: str = ""
    tier: SkillTier = "domain"
    parent: str | None = None
    related: list[str] = Field(default_factory=list)
    coverage: SkillCoverage = Field(default_factory=SkillCoverage)
    body: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    eval_status: EvalStatus = "not_run"
    eval_summary: str = ""
    eval_failures: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    signals_used: list[str] = Field(default_factory=list)
    adversary_critique: Critique | None = None
    status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    created_at: str  # ISO-8601
    approved_by: str | None = None
    approved_at: str | None = None


# ---------------------------------------------------------------- confidence formula


def compute_confidence(*, citations: list[Citation], paragraphs: int, revision_count: int) -> float:
    """confidence = citation_density * critic_passes."""
    if paragraphs <= 0:
        return 0.0
    citation_density = min(len(citations) / paragraphs, 1.0)
    critic_passes = 1.0 if revision_count == 0 else 0.7
    return round(citation_density * critic_passes, 4)
