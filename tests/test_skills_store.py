from pathlib import Path

from nexus.skills.models import AppliesTo, Provenance, Skill
from nexus.skills.store import SkillStore


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    skill = Skill(
        name="example",
        description="Use for testing Agent Skills storage.",
        product="forge",
        tier="tech_stack",
        parent="forge-master",
        related=["forge/domain"],
        coverage={"repos": ["api"], "topics": ["testing"]},
        version=2,
        confidence=0.5,
        applies_to=AppliesTo(files=["**/*.py"], contexts=["code-review"]),
        provenance=Provenance(
            validated_by="me@example.com",
            validated_at="2026-05-18T00:00:00Z",
            evidence_chunks=["c1", "c2"],
            revision_count=1,
        ),
        body="# Example\n\nBody here.\n",
    )
    path = store.save(skill)
    assert path.exists()
    assert path.name == "SKILL.md"
    assert path.parent.name == "example"
    assert path.parent.parent.name == "forge"

    loaded = store.load_path(path)
    assert loaded.name == "example"
    assert loaded.description == "Use for testing Agent Skills storage."
    assert loaded.product == "forge"
    assert loaded.tier == "tech_stack"
    assert loaded.parent == "forge-master"
    assert loaded.related == ["forge/domain"]
    assert loaded.coverage.repos == ["api"]
    assert loaded.coverage.topics == ["testing"]
    assert loaded.confidence == 0.5
    assert loaded.applies_to.files == ["**/*.py"]
    assert loaded.body.strip() == "# Example\n\nBody here.".strip()
    assert loaded.provenance.revision_count == 1


def test_loads_legacy_flat_skill_file(tmp_path: Path) -> None:
    legacy = tmp_path / "forge" / "legacy.skill.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "---\n"
        "name: legacy\n"
        "description: Legacy readable skill.\n"
        "product: forge\n"
        "tier: domain\n"
        "confidence: 0.7\n"
        "applies_to: {files: [], contexts: []}\n"
        "provenance:\n"
        "  validated_by: me\n"
        "  validated_at: 2026-05-18T00:00:00Z\n"
        "---\n"
        "# Legacy\n",
        encoding="utf-8",
    )
    loaded = SkillStore(tmp_path).load_path(legacy)
    assert loaded.name == "legacy"
    assert loaded.description == "Legacy readable skill."
    assert loaded.product == "forge"


def test_iter_loads_flat_product_skill_file(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    skill = Skill(
        name="product-skill",
        description="Use for product orientation and grounded development.",
        product="forge",
        tier="product_master",
        parent=None,
        related=[],
        coverage={},
        version=1,
        confidence=0.8,
        applies_to=AppliesTo(),
        provenance=Provenance(
            validated_by="me@example.com",
            validated_at="2026-05-18T00:00:00Z",
            evidence_chunks=["c1"],
            revision_count=0,
        ),
        body="# product-skill\n\nBody here.\n",
    )
    path = store.save(skill)

    assert path == tmp_path / "forge" / "product-skill.md"
    assert [(s.product, s.name) for s in store.iter_skills()] == [
        ("forge", "product-skill")
    ]


def test_iter_skips_legacy_files_without_product(tmp_path: Path) -> None:
    """Legacy org-library files (no `product:` in frontmatter) are skipped."""
    legacy = tmp_path / "shared" / "legacy.skill.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "---\nname: legacy\nkind: language\nscope: org\n---\n# Legacy\n",
        encoding="utf-8",
    )
    store = SkillStore(tmp_path)
    assert store.iter_skills() == []
