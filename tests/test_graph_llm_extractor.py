from __future__ import annotations

import pytest

from anvay.graph.extractor import extract_resource_graph
from anvay.graph.llm_extractor import merge_llm_graph_facts, parse_llm_graph_facts
from anvay.ingest.models import ResourceRef


def test_parse_llm_graph_facts_filters_and_caps() -> None:
    raw = """
    {
      "facts": [
        {
          "subject": {"name": "read_token", "label": "Function"},
          "predicate": "calls",
          "object": {"name": "load_policy", "label": "Function"},
          "evidence": "read_token calls load_policy",
          "start_line": 2,
          "end_line": 3,
          "confidence": 0.9
        },
        {
          "subject": {"name": "weak", "label": "Function"},
          "predicate": "CALLS",
          "object": {"name": "maybe", "label": "Function"},
          "evidence": "weak maybe",
          "start_line": 4,
          "end_line": 4,
          "confidence": 0.2
        }
      ]
    }
    """

    facts = parse_llm_graph_facts(raw, confidence_floor=0.65, max_facts=1)

    assert len(facts) == 1
    assert facts[0].predicate == "CALLS"
    assert facts[0].subject.name == "read_token"


def test_parse_llm_graph_facts_accepts_string_entity_shorthand() -> None:
    facts = parse_llm_graph_facts(
        """
        {
          "facts": [{
            "subject": "test_product_skill_validation.py",
            "predicate": "MENTIONS",
            "object": "validate_skill_markdown",
            "evidence": "test imports validate_skill_markdown",
            "start_line": 1,
            "end_line": 4,
            "confidence": 0.86
          }]
        }
        """,
        confidence_floor=0.65,
        max_facts=10,
    )

    assert facts[0].subject.label == "CodeFile"
    assert facts[0].object.label == "Function"


def test_parse_llm_graph_facts_rejects_contains_edge() -> None:
    with pytest.raises(ValueError, match="invalid LLM graph payload"):
        parse_llm_graph_facts(
            """
            {
              "facts": [{
                "subject": "anvay/ingest/enricher.py",
                "predicate": "CONTAINS",
                "object": "ContextualEnricher",
                "evidence": "class ContextualEnricher",
                "start_line": 38,
                "end_line": 39,
                "confidence": 0.91
              }]
            }
            """,
            confidence_floor=0.65,
            max_facts=10,
        )


def test_parse_llm_graph_facts_synthesizes_missing_evidence() -> None:
    facts = parse_llm_graph_facts(
        """
        {
          "facts": [{
            "subject": {"name": "run_ingest", "label": "Function"},
            "predicate": "DEPENDS_ON",
            "object": {"name": "EmbedderClient", "label": "Class"},
            "start_line": 187,
            "end_line": 191,
            "confidence": 1
          }]
        }
        """,
        confidence_floor=0.65,
        max_facts=10,
    )

    assert facts[0].evidence == "run_ingest DEPENDS_ON EmbedderClient"


def test_parse_llm_graph_facts_keeps_valid_facts_when_later_fact_invalid() -> None:
    facts = parse_llm_graph_facts(
        """
        {
          "facts": [
            {
              "subject": "pipeline.py",
              "predicate": "MENTIONS",
              "object": "run_ingest",
              "start_line": 1,
              "end_line": 2,
              "confidence": 0.9
            },
            {
              "subject": "pipeline.py",
              "predicate": "HALLUCINATES",
              "object": "nope",
              "start_line": 1,
              "end_line": 2,
              "confidence": 0.9
            }
          ]
        }
        """,
        confidence_floor=0.65,
        max_facts=10,
    )

    assert len(facts) == 1
    assert facts[0].object.name == "run_ingest"


def test_parse_llm_graph_facts_salvages_complete_facts_from_truncated_json() -> None:
    raw = """
    {
      "facts": [
        {
          "subject": "embedder.py",
          "predicate": "MENTIONS",
          "object": "EmbedderClient",
          "evidence": "EmbedderClient is referenced",
          "start_line": 1,
          "end_line": 2,
          "confidence": 0.88
        },
        {
          "subject": "unfinished.py",
          "predicate": "MENTIONS",
          "object": "broken",
          "evidence": "unterminated
    """

    facts = parse_llm_graph_facts(raw, confidence_floor=0.65, max_facts=10)

    assert len(facts) == 1
    assert facts[0].subject.name == "embedder.py"


def test_parse_llm_graph_facts_rejects_bad_json_and_unknown_edge() -> None:
    with pytest.raises(ValueError, match="invalid LLM graph payload"):
        parse_llm_graph_facts("{nope", confidence_floor=0.65, max_facts=10)

    with pytest.raises(ValueError, match="invalid LLM graph payload"):
        parse_llm_graph_facts(
            """
            {
              "facts": [{
                "subject": {"name": "a", "label": "Function"},
                "predicate": "HALLUCINATES",
                "object": {"name": "b", "label": "Function"},
                "evidence": "x",
                "start_line": 1,
                "end_line": 1,
                "confidence": 0.9
              }]
            }
            """,
            confidence_floor=0.65,
            max_facts=10,
        )


def test_merge_llm_graph_facts_adds_source_backed_edge() -> None:
    content = "def read_token():\n    return load_policy()\n\ndef load_policy():\n    return {}\n"
    resource = ResourceRef(source_id="local:test", uri="app.py", mime="text/x-python")
    base = extract_resource_graph(
        product_id="prod",
        source_key="src",
        resource=resource,
        content=content,
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    facts = parse_llm_graph_facts(
        """
        {
          "facts": [{
            "subject": {"name": "read_token", "label": "Function"},
            "predicate": "CALLS",
            "object": {"name": "load_policy", "label": "Function"},
            "evidence": "return load_policy()",
            "start_line": 2,
            "end_line": 2,
            "confidence": 0.92
          }]
        }
        """,
        confidence_floor=0.65,
        max_facts=10,
    )

    merged = merge_llm_graph_facts(
        base=base,
        product_id="prod",
        source_key="src",
        resource=resource,
        facts=facts,
        indexed_at="2026-01-01T00:00:00+00:00",
    )

    call_edges = [edge for edge in merged.edges if edge.type == "CALLS"]
    assert len(call_edges) == 1
    assert any(ref.anchor == "app.py:2" for ref in call_edges[0].source_refs)
