from pathlib import Path

from anvay.council.queue import ProposalQueue
from anvay.skills.models import Citation, SkillProposal


def _make_proposal(name: str = "demo", confidence: float = 0.5) -> SkillProposal:
    return SkillProposal(
        id=f"prop_{name}",
        name=name,
        description=f"Use for {name} tests.",
        tier="domain",
        body="# Demo\n\nBody [file: a/b.py:10].",
        citations=[Citation(file="a/b.py", line=10, excerpt="x")],
        confidence=confidence,
        status="pending",
        created_at="2026-05-18T00:00:00Z",
    )


def test_enqueue_then_list_returns_proposal(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "proposals.db")
    proposal = _make_proposal()
    proposal.eval_status = "passed"
    proposal.eval_summary = "Eval passed."
    proposal.quality_score = 0.91
    proposal.signals_used = ["sig_1"]
    queue.enqueue(
        proposal,
        session_id="cs_x",
        product_id="forge",
    )
    pending = queue.list(status="pending")
    assert len(pending) == 1
    assert pending[0]["name"] == "demo"
    assert pending[0]["description"] == "Use for demo tests."
    assert pending[0]["tier"] == "domain"
    assert pending[0]["citations"] == [
        {"id": None, "file": "a/b.py", "line": 10, "excerpt": "x"}
    ]
    assert pending[0]["eval_status"] == "passed"
    assert pending[0]["eval_summary"] == "Eval passed."
    assert pending[0]["quality_score"] == 0.91
    assert pending[0]["signals_used"] == ["sig_1"]


def test_list_filters_by_product(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "p.db")
    queue.enqueue(
        _make_proposal("a"), session_id="s1", product_id="forge"
    )
    queue.enqueue(
        _make_proposal("b"), session_id="s2", product_id="atlas"
    )
    assert {p["name"] for p in queue.list(product_id="forge")} == {"a"}
    assert {p["name"] for p in queue.list(product_id="atlas")} == {"b"}


def test_update_status_transitions(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "u.db")
    p = _make_proposal()
    queue.enqueue(p, session_id="s", product_id="forge")
    assert queue.update_status(p.id, status="rejected", actor="reviewer@x")
    row = queue.get(p.id)
    assert row is not None
    assert row["status"] == "rejected"
    assert row["approved_by"] == "reviewer@x"
    assert row["approved_at"]


def test_record_and_get_session(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "s.db")
    queue.record_session(
        session_id="cs_demo",
        product_id="forge",
        topic="overview",
        proposal_id="prop_demo",
        proposal_ids=["prop_demo", "prop_api"],
        deliberation=[{"agent": "archaeologist", "body": "found stuff"}],
        costs=[{"agent": "archaeologist", "prompt_tokens": 100, "completion_tokens": 50}],
        started_at="2026-05-18T00:00:00Z",
        completed_at="2026-05-18T00:00:42Z",
    )
    s = queue.get_session("cs_demo")
    assert s is not None
    assert s["topic"] == "overview"
    assert s["proposal_ids"] == ["prop_demo", "prop_api"]
    assert s["deliberation"] == [{"agent": "archaeologist", "body": "found stuff"}]
    assert s["costs"][0]["prompt_tokens"] == 100


def test_list_sessions_orders_newest_first(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "ls.db")
    for i, ts in enumerate(["2026-05-18T00:00:00Z", "2026-05-18T01:00:00Z"]):
        queue.record_session(
            session_id=f"cs_{i}",
            product_id="forge",
            topic=f"topic_{i}",
            proposal_id=None,
            deliberation=[],
            costs=[],
            started_at=ts,
            completed_at=ts,
        )
    sessions = queue.list_sessions(product_id="forge")
    assert [s["id"] for s in sessions] == ["cs_1", "cs_0"]


def test_records_skill_signals_and_eval_results_product_scoped(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "quality.db")
    sig = queue.record_skill_signal(
        product_id="forge",
        source_type="rejection",
        skill_name="forge-master",
        proposal_id="prop_1",
        session_id="cs_1",
        text="Too generic.",
        metadata={"actor": "reviewer"},
    )
    queue.record_skill_signal(
        product_id="atlas",
        source_type="mcp_outcome",
        skill_name="atlas-master",
        text="Worked.",
    )
    queue.record_eval_run(
        run_id="run_1",
        session_id="cs_1",
        product_id="forge",
        suite_version="skill-quality-v1",
        status="partial",
        summary="1/2 passed",
    )
    queue.record_eval_result(
        run_id="run_1",
        session_id="cs_1",
        product_id="forge",
        skill_name="forge-master",
        status="failed",
        summary="Failed eval.",
        failures=["Missing citations."],
        quality_score=0.25,
        attempts=1,
        signals_used=[sig],
    )

    signals = queue.list_skill_signals(product_id="forge")
    assert len(signals) == 1
    assert signals[0]["id"] == sig
    assert signals[0]["metadata"] == {"actor": "reviewer"}

    results = queue.list_eval_results(product_id="forge")
    assert len(results) == 1
    assert results[0]["skill_name"] == "forge-master"
    assert results[0]["failures"] == ["Missing citations."]
    assert results[0]["signals_used"] == [sig]

    session = queue.get_session("cs_1")
    assert session is None


def test_delete_product_removes_quality_records(tmp_path: Path) -> None:
    queue = ProposalQueue(tmp_path / "delete.db")
    queue.record_skill_signal(product_id="forge", source_type="mcp_outcome", text="bad")
    queue.record_eval_run(
        run_id="run_1",
        session_id="cs_1",
        product_id="forge",
        suite_version="skill-quality-v1",
        status="failed",
    )
    queue.record_eval_result(
        run_id="run_1",
        session_id="cs_1",
        product_id="forge",
        skill_name="forge-master",
        status="failed",
    )

    queue.delete_product("forge")

    assert queue.list_skill_signals(product_id="forge") == []
    assert queue.list_eval_results(product_id="forge") == []
