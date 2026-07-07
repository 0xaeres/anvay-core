"""Routing shape of the single product-skill council graph."""

from anvay.config import (
    AnvayConfig,
    EnrichCfg,
    IngestionCfg,
    ModelCfg,
    ModelsCfg,
    ServerCfg,
    StorageCfg,
    VectorStoreCfg,
)
from anvay.council.graph import build_graph
from anvay.council.state import initial_state


def _make_cfg() -> AnvayConfig:
    m = ModelCfg(provider="deepinfra", model="x")
    return AnvayConfig(
        skills_repo="git@example:repo.git",
        connectors=[],
        vector_store=VectorStoreCfg(),
        models=ModelsCfg(
            council=m,
            light=m,
            embedding=ModelCfg(provider="jina-local", model="j", url="http://x"),
            reranker=ModelCfg(provider="jina-local", model="j", url="http://x"),
        ),
        ingestion=IngestionCfg(enrich_chunks=EnrichCfg()),
        server=ServerCfg(),
        storage=StorageCfg(),
    )


def test_initial_state_revision_zero() -> None:
    state = initial_state(
        session_id="cs_t",
        product_id="p",
        topic="t",
        config_path="x",
    )
    assert state["revision_count"] == 0
    assert state["critique"] is None
    assert state["proposal"] is None
    assert state["evidence"] == []
    assert state["proposals"] == []


def test_build_graph_has_skill_nodes() -> None:
    """Smoke: graph has the bounded product-skill council nodes."""
    from anvay.council.graph import CouncilHandles

    handles = CouncilHandles.__new__(CouncilHandles)
    handles.retrieval = None  # type: ignore[assignment]
    handles.chat_planner = None  # type: ignore[assignment]
    handles.chat_evaluator = None  # type: ignore[assignment]
    handles.chat_repair = None  # type: ignore[assignment]
    handles.chat_synthesizer = None  # type: ignore[assignment]

    graph = build_graph(_make_cfg(), handles)
    assert "planner" in graph.nodes
    assert "synthesizer" in graph.nodes
    assert "repair" in graph.nodes
    assert "skill_eval" in graph.nodes
    assert "finalizer" in graph.nodes
    # Expert fanout removed: skill generation is one synthesis call grounded in
    # deterministic KB artifacts.
    assert "architect" not in graph.nodes
    assert "domain_expert" not in graph.nodes
    assert "quality_expert" not in graph.nodes
    assert "experts" not in graph.nodes
    assert "judge" not in graph.nodes
    assert "targeted_callback" not in graph.nodes


def test_route_after_eval_sends_failures_to_repair() -> None:
    from anvay.council.graph import _route_after_eval
    from anvay.council.state import SkillEvalResult

    state = {
        "eval_results": [
            SkillEvalResult(skill_name="a", status="failed"),
        ],
        "eval_repair_attempts": 0,
    }
    assert _route_after_eval(state) == "repair"


def test_route_after_eval_sends_pass_to_finalizer() -> None:
    from anvay.council.graph import _route_after_eval
    from anvay.council.state import SkillEvalResult

    state = {
        "eval_results": [SkillEvalResult(skill_name="a", status="passed")],
        "eval_repair_attempts": 0,
    }
    assert _route_after_eval(state) == "finalizer"


def test_route_after_eval_stops_at_attempt_cap() -> None:
    from anvay.council.graph import MAX_EVAL_REPAIR_ATTEMPTS, _route_after_eval
    from anvay.council.state import SkillEvalResult

    state = {
        "eval_results": [SkillEvalResult(skill_name="a", status="failed")],
        "eval_repair_attempts": MAX_EVAL_REPAIR_ATTEMPTS,
    }
    # Budget exhausted — proceed to finalizer even though a draft still fails.
    assert _route_after_eval(state) == "finalizer"


def test_route_after_eval_uses_latest_verdict_per_skill() -> None:
    from anvay.council.graph import _route_after_eval
    from anvay.council.state import SkillEvalResult

    # eval_results is append-only; a later 'passed' supersedes an earlier 'failed'.
    state = {
        "eval_results": [
            SkillEvalResult(skill_name="a", status="failed"),
            SkillEvalResult(skill_name="a", status="passed"),
        ],
        "eval_repair_attempts": 1,
    }
    assert _route_after_eval(state) == "finalizer"
