from anvay.council.skill_catalog import MAX_SKILL_NAME_CHARS, SKILL_CATALOG, catalog_plan


def test_catalog_has_single_product_skill() -> None:
    assert [item.tier for item in SKILL_CATALOG] == ["product_master"]

    plan = catalog_plan("My Product", "overview")
    assert [item.name for item in plan] == ["my-product-skill"]
    assert all(item.description for item in plan)
    assert plan[0].parent is None
    assert plan[0].related == []


def test_product_skill_name_is_agent_skill_safe() -> None:
    plan = catalog_plan("x" * 100, "")
    assert all(len(item.name) <= MAX_SKILL_NAME_CHARS for item in plan)


def test_product_skill_name_slugifies_product_id() -> None:
    plan = catalog_plan("Forge AMM/API", "")
    assert [item.name for item in plan] == ["forge-amm-api-skill"]
