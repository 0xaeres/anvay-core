"""Single generic Anvay product skill template.

Every council run produces one product-scoped Agent Skill proposal named after
the product, e.g. `forge-skill`. Older approved multi-skill files remain
readable in the store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvay.skills.models import SkillTier

PRODUCT_SKILL_SUFFIX = "skill"
MAX_SKILL_NAME_CHARS = 64


REQUIRED_PRODUCT_SKILL_SECTIONS = (
    "Use This Skill When",
    "Product Snapshot",
    "Product Language",
    "Capabilities And Workflows",
    "System Map",
    "Data Model",
    "Interfaces And Contracts",
    "Invariants And Constraints",
    "How To Use The Knowledge Base",
    "How To Work In This Product",
    "Security And Secrets",
    "Known Traps",
    "Freshness And Evidence",
)

OPTIONAL_PRODUCT_SKILL_SECTIONS = (
    "Testing Strategy",
    "Common Change Patterns",
)

CITED_PRODUCT_SKILL_SECTIONS = (
    "Product Snapshot",
    "Product Language",
    "Capabilities And Workflows",
    "System Map",
    "Data Model",
    "Interfaces And Contracts",
    "Invariants And Constraints",
    "Security And Secrets",
    "Known Traps",
    "Freshness And Evidence",
)

PRODUCT_SKILL_RETRIEVAL_QUERY = (
    "product overview architecture data model interfaces contracts workflows "
    "testing security secrets knowledge base RAG retrieval MCP APIs schemas "
    "runtime boundaries invariants constraints debug review change patterns"
)


@dataclass(frozen=True)
class CatalogSkill:
    suffix: str
    tier: SkillTier
    purpose: str
    description: str
    topics: tuple[str, ...]
    retrieval_suffix: str
    sections: tuple[str, ...]
    cited_sections: tuple[str, ...]


SKILL_CATALOG: tuple[CatalogSkill, ...] = (
    CatalogSkill(
        suffix=PRODUCT_SKILL_SUFFIX,
        tier="product_master",
        purpose=(
            "Single product orientation skill covering product purpose, language, "
            "system map, contracts, data model, workflow, security, traps, and KB use."
        ),
        description=(
            "Use for product orientation, grounded development, review, debugging, "
            "and deciding when to query the product knowledge base."
        ),
        topics=(
            "product",
            "architecture",
            "data model",
            "interfaces",
            "workflows",
            "testing",
            "security",
            "knowledge base",
        ),
        retrieval_suffix=PRODUCT_SKILL_RETRIEVAL_QUERY,
        sections=REQUIRED_PRODUCT_SKILL_SECTIONS,
        cited_sections=CITED_PRODUCT_SKILL_SECTIONS,
    ),
)

REQUIRED_SECTIONS_BY_TIER: dict[SkillTier, tuple[str, ...]] = {
    "product_master": REQUIRED_PRODUCT_SKILL_SECTIONS,
    "application": REQUIRED_PRODUCT_SKILL_SECTIONS,
    "domain": REQUIRED_PRODUCT_SKILL_SECTIONS,
    "interface": REQUIRED_PRODUCT_SKILL_SECTIONS,
    "tech_stack": REQUIRED_PRODUCT_SKILL_SECTIONS,
    "quality_security": REQUIRED_PRODUCT_SKILL_SECTIONS,
}

CITED_SECTIONS_BY_TIER: dict[SkillTier, tuple[str, ...]] = {
    "product_master": CITED_PRODUCT_SKILL_SECTIONS,
    "application": CITED_PRODUCT_SKILL_SECTIONS,
    "domain": CITED_PRODUCT_SKILL_SECTIONS,
    "interface": CITED_PRODUCT_SKILL_SECTIONS,
    "tech_stack": CITED_PRODUCT_SKILL_SECTIONS,
    "quality_security": CITED_PRODUCT_SKILL_SECTIONS,
}


def catalog_plan(product_id: str, topic: str = ""):
    """Build deterministic single-skill plan for a product."""
    from anvay.council.state import SkillPlanItem

    topics = [*SKILL_CATALOG[0].topics, topic]
    return [
        SkillPlanItem(
            name=fixed_skill_name(product_slug(product_id), PRODUCT_SKILL_SUFFIX),
            description=SKILL_CATALOG[0].description,
            tier="product_master",
            purpose=SKILL_CATALOG[0].purpose,
            parent=None,
            related=[],
            coverage={"topics": [t for t in topics if t]},
        )
    ]


def fixed_skill_name(product_slug_value: str, suffix: str) -> str:
    suffix = suffix.strip("-") or PRODUCT_SKILL_SUFFIX
    max_slug_chars = MAX_SKILL_NAME_CHARS - len(suffix) - 1
    slug = (product_slug_value[:max_slug_chars].strip("-") or "product")
    return f"{slug}-{suffix}"


def product_slug(product_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", product_id.lower()).strip("-")
    return slug or "product"
