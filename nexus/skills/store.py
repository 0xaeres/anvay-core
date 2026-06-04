"""Filesystem store for skills — Agent Skills dirs + legacy Markdown files.

New approved skills use Agent Skills layout:
`<product>/<name>/SKILL.md`. Legacy flat files like
`<product>/<name>.skill.md` remain readable. The single product-master skill
is stored as `<product>/product-skill.md`.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from nexus.skills.models import AppliesTo, Provenance, Skill, SkillCoverage


class SkillStore:
    """Read/write product-scoped skill files on a local clone of the skills repo."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ read

    def load(self, relative_path: str) -> Skill:
        return self.load_path(self.root / relative_path)

    def load_path(self, path: Path) -> Skill:
        post = frontmatter.load(str(path))
        meta = _coerce_dates(dict(post.metadata))
        return _build_skill(meta, post.content)

    def iter_skills(self) -> list[Skill]:
        out: list[Skill] = []
        for path in self._skill_paths():
            try:
                out.append(self.load_path(path))
            except (KeyError, ValueError):
                # Skip malformed / legacy org-library files that no longer parse.
                continue
        return out

    # ------------------------------------------------------------ write

    def save(self, skill: Skill, relative_path: str | None = None) -> Path:
        rel = relative_path or SkillStore.relative_path_for(skill)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content=skill.body, **_dump_frontmatter(skill))
        out = frontmatter.dumps(
            post,
            handler=frontmatter.YAMLHandler(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        path.write_text(out, encoding="utf-8")
        return path

    def delete_product(self, product_id: str) -> int:
        deleted = 0
        for path in list(self._skill_paths()):
            try:
                skill = self.load_path(path)
            except (KeyError, ValueError):
                continue
            if skill.product != product_id:
                continue
            path.unlink(missing_ok=True)
            if path.name == "SKILL.md":
                with suppress(OSError):
                    path.parent.rmdir()
            deleted += 1

        product_dir = self.root / product_id
        if product_dir.exists():
            for child in sorted(product_dir.rglob("*"), reverse=True):
                if child.is_dir():
                    with suppress(OSError):
                        child.rmdir()
            with suppress(OSError):
                product_dir.rmdir()
        return deleted

    # ------------------------------------------------------------ paths

    @staticmethod
    def relative_path_for(skill: Skill) -> str:
        if skill.name == "product-skill":
            return f"{skill.product}/product-skill.md"
        return f"{skill.product}/{skill.name}/SKILL.md"

    def _skill_paths(self) -> list[Path]:
        paths = [
            *self.root.rglob("SKILL.md"),
            *self.root.rglob("*.skill.md"),
            *self.root.rglob("product-skill.md"),
        ]
        return sorted({p.resolve(): p for p in paths}.values())


# ---------------------------------------------------------------- builders


def _build_skill(meta: dict, body: str) -> Skill:
    metadata = meta.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("nexus_product"):
        return _build_agent_skill(meta, metadata, body)

    # `product` is required; legacy org-scoped files lacked it and are skipped
    # by iter_skills() via the try/except wrap.
    product = meta.get("product")
    if not product:
        raise ValueError("skill frontmatter missing 'product'")
    return Skill(
        name=meta["name"],
        description=str(meta.get("description", "")),
        product=product,
        tier=_coerce_tier(meta),
        parent=meta.get("parent"),
        related=list(meta.get("related") or meta.get("composes_with") or []),
        coverage=SkillCoverage(**(meta.get("coverage") or {})),
        version=int(meta.get("version", 1)),
        confidence=float(meta.get("confidence", 0.0)),
        eval_status=str(meta.get("eval_status", "not_run")),
        eval_summary=str(meta.get("eval_summary", "")),
        eval_failures=list(meta.get("eval_failures") or []),
        quality_score=float(meta.get("quality_score", 0.0)),
        signals_used=list(meta.get("signals_used") or []),
        applies_to=AppliesTo(**(meta.get("applies_to") or {})),
        provenance=Provenance(**meta["provenance"]),
        body=body,
    )


def _dump_frontmatter(skill: Skill) -> dict:
    return {
        "name": skill.name,
        "description": skill.description,
        "compatibility": {
            "agents": ["codex", "claude", "cursor", "continue"],
            "format": "agent-skills",
        },
        "metadata": {
            "nexus_product": skill.product,
            "nexus_tier": skill.tier,
            "nexus_parent": skill.parent,
            "nexus_related": skill.related,
            "nexus_coverage": skill.coverage.model_dump(),
            "nexus_version": skill.version,
            "nexus_confidence": skill.confidence,
            "nexus_eval_status": skill.eval_status,
            "nexus_eval_summary": skill.eval_summary,
            "nexus_eval_failures": skill.eval_failures,
            "nexus_quality_score": skill.quality_score,
            "nexus_signals_used": skill.signals_used,
            "nexus_applies_to": skill.applies_to.model_dump(),
            "nexus_provenance": skill.provenance.model_dump(),
        },
    }


# silence ruff F401 for yaml (frontmatter uses it transitively)
_ = yaml


def _coerce_dates(meta: dict[str, Any]) -> dict[str, Any]:
    """YAML parses ISO-8601 datetimes natively; coerce back to strings for Pydantic."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, datetime | date):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = _coerce_dates(v)
        else:
            out[k] = v
    return out


def _coerce_tier(meta: dict) -> str:
    tier = meta.get("tier")
    if tier:
        return str(tier)
    kind = str(meta.get("kind", "")).lower()
    scope = str(meta.get("scope", "")).lower()
    if kind == "master" or (kind == "product_master" and scope == "product"):
        return "product_master"
    if kind in {"product_domain", "domain"}:
        return "domain"
    if kind in {"tech_stack", "language"}:
        return "tech_stack"
    if kind in {"security", "quality_security"}:
        return "quality_security"
    return "domain"


def _build_agent_skill(meta: dict, metadata: dict, body: str) -> Skill:
    return Skill(
        name=meta["name"],
        description=str(meta.get("description", "")),
        product=str(metadata["nexus_product"]),
        tier=str(metadata.get("nexus_tier") or "domain"),
        parent=metadata.get("nexus_parent"),
        related=list(metadata.get("nexus_related") or []),
        coverage=SkillCoverage(**(metadata.get("nexus_coverage") or {})),
        version=int(metadata.get("nexus_version", 1)),
        confidence=float(metadata.get("nexus_confidence", 0.0)),
        eval_status=str(metadata.get("nexus_eval_status", "not_run")),
        eval_summary=str(metadata.get("nexus_eval_summary", "")),
        eval_failures=list(metadata.get("nexus_eval_failures") or []),
        quality_score=float(metadata.get("nexus_quality_score", 0.0)),
        signals_used=list(metadata.get("nexus_signals_used") or []),
        applies_to=AppliesTo(**(metadata.get("nexus_applies_to") or {})),
        provenance=Provenance(**metadata["nexus_provenance"]),
        body=body,
    )
