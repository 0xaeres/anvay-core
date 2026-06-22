"""Bounded LLM graph fact extraction for code/docs resources."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from anvay.config import GraphIngestionCfg
from anvay.graph.models import GraphEdge, GraphExtraction, GraphNode, SourceRef
from anvay.ingest.models import ResourceRef
from anvay.llm.client import ChatClient

log = logging.getLogger(__name__)

_GRAPH_NS = uuid.UUID("5db79a04-43f6-4414-a94b-7d28fdf56f95")
_ALLOWED_LABELS = {
    "APIEndpoint",
    "Class",
    "CodeFile",
    "Config",
    "DBTable",
    "Document",
    "FeatureFlag",
    "Function",
    "Module",
    "ProductFlow",
    "Runbook",
    "Service",
    "Test",
}
_ALLOWED_EDGES = {
    "CALLS",
    "CONSTRAINS",
    "DEPENDS_ON",
    "DOCUMENTS",
    "IMPLEMENTS",
    "MENTIONS",
    "PART_OF_FLOW",
}


class LLMEntity(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=64)
    resource_uri: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @field_validator("label")
    @classmethod
    def allowed_label(cls, value: str) -> str:
        if value not in _ALLOWED_LABELS:
            raise ValueError(f"unsupported graph label: {value}")
        return value


class LLMGraphFact(BaseModel):
    subject: LLMEntity
    predicate: str = Field(min_length=1, max_length=64)
    object: LLMEntity
    evidence: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_shorthand_entities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("subject", "object"):
            entity = data.get(key)
            if isinstance(entity, str):
                data[key] = {"name": entity, "label": _infer_entity_label(entity)}
        if not data.get("evidence"):
            subject = _entity_name(data.get("subject"))
            obj = _entity_name(data.get("object"))
            predicate = str(data.get("predicate") or "RELATED_TO").upper()
            data["evidence"] = f"{subject} {predicate} {obj}".strip()
        return data

    @field_validator("predicate")
    @classmethod
    def allowed_predicate(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _ALLOWED_EDGES:
            raise ValueError(f"unsupported graph edge type: {value}")
        return normalized


class LLMGraphPayload(BaseModel):
    facts: list[LLMGraphFact] = Field(default_factory=list)


async def extract_bounded_llm_graph(
    *,
    chat: ChatClient,
    cfg: GraphIngestionCfg,
    product_id: str,
    source_key: str,
    resource: ResourceRef,
    content: str,
    base: GraphExtraction,
    indexed_at: str | None = None,
) -> GraphExtraction:
    """Return deterministic graph plus validated, source-anchored LLM facts."""
    response = await chat.chat(
        [
            {
                "role": "system",
                "content": (
                    "Extract only source-supported code/docs graph facts. "
                    "Return JSON object {\"facts\": [...]}. "
                    "Allowed labels: "
                    + ", ".join(sorted(_ALLOWED_LABELS))
                    + ". Allowed predicates: "
                    + ", ".join(sorted(_ALLOWED_EDGES))
                    + ". Each fact needs subject, predicate, object, evidence, "
                    "subject and object must be objects with name and label, not strings. "
                    "start_line, end_line, confidence. Do not infer facts not "
                    "anchored in the supplied source."
                ),
            },
            {
                "role": "user",
                "content": _source_prompt(resource=resource, content=content),
            },
        ],
        json_mode=True,
        max_tokens=1800,
        stream=False,
    )
    facts = parse_llm_graph_facts(
        response.content,
        confidence_floor=cfg.confidence_floor,
        max_facts=cfg.max_facts_per_resource,
    )
    return merge_llm_graph_facts(
        base=base,
        product_id=product_id,
        source_key=source_key,
        resource=resource,
        facts=facts,
        indexed_at=indexed_at,
    )


def parse_llm_graph_facts(
    raw: str,
    *,
    confidence_floor: float,
    max_facts: int,
) -> list[LLMGraphFact]:
    """Parse and filter LLM graph facts. Invalid payloads fail closed."""
    try:
        facts = _load_fact_payload(raw)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
        raise ValueError(f"invalid LLM graph payload: {e}") from e
    return _filter_facts(facts, confidence_floor=confidence_floor, max_facts=max_facts)


def _filter_facts(
    facts: Sequence[LLMGraphFact],
    *,
    confidence_floor: float,
    max_facts: int,
) -> list[LLMGraphFact]:
    out: list[LLMGraphFact] = []
    for fact in facts:
        if fact.confidence < confidence_floor:
            continue
        if fact.end_line < fact.start_line:
            continue
        out.append(fact)
        if len(out) >= max_facts:
            break
    return out


def _load_fact_payload(raw: str) -> list[LLMGraphFact]:
    text = _strip_json_fence(raw.strip())
    try:
        payload = json.loads(text)
        return _facts_from_payload(payload)
    except json.JSONDecodeError:
        partial = _load_partial_facts(text)
        if partial:
            return partial
        raise


def _facts_from_payload(payload: Any) -> list[LLMGraphFact]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    raw_facts = payload.get("facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError("facts must be a list")
    facts: list[LLMGraphFact] = []
    errors: list[ValidationError] = []
    for item in raw_facts:
        try:
            facts.append(LLMGraphFact.model_validate(item))
        except ValidationError as e:
            errors.append(e)
    if facts:
        return facts
    if errors:
        raise errors[0]
    return []


def _load_partial_facts(text: str) -> list[LLMGraphFact]:
    facts_key = text.find('"facts"')
    if facts_key < 0:
        return []
    array_start = text.find("[", facts_key)
    if array_start < 0:
        return []
    decoder = json.JSONDecoder()
    idx = array_start + 1
    facts: list[LLMGraphFact] = []
    while idx < len(text):
        while idx < len(text) and text[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break
        try:
            item, idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        try:
            facts.append(LLMGraphFact.model_validate(item))
        except ValidationError:
            continue
    return facts


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def merge_llm_graph_facts(
    *,
    base: GraphExtraction,
    product_id: str,
    source_key: str,
    resource: ResourceRef,
    facts: Sequence[LLMGraphFact],
    indexed_at: str | None = None,
) -> GraphExtraction:
    now = indexed_at or datetime.now(UTC).isoformat()
    nodes = {node.stable_id: node for node in base.nodes}
    aliases = _aliases(nodes.values())
    edges = {edge.stable_id: edge for edge in base.edges}

    for fact in facts:
        ref = _source_ref(product_id, source_key, resource, fact, now)
        from_id = _node_id_for_entity(
            fact.subject,
            product_id=product_id,
            resource=resource,
            nodes=nodes,
            aliases=aliases,
            ref=ref,
            now=now,
        )
        to_id = _node_id_for_entity(
            fact.object,
            product_id=product_id,
            resource=resource,
            nodes=nodes,
            aliases=aliases,
            ref=ref,
            now=now,
        )
        existing_edge = next(
            (
                edge
                for edge in edges.values()
                if edge.from_id == from_id and edge.to_id == to_id and edge.type == fact.predicate
            ),
            None,
        )
        if existing_edge is not None:
            existing_edge.source_refs = _merge_refs(existing_edge.source_refs, [ref])
            existing_edge.confidence = max(existing_edge.confidence, fact.confidence)
            existing_edge.properties.setdefault("evidence", fact.evidence)
            continue
        edge_id = _edge_id(product_id, from_id, fact.predicate, to_id, ref.anchor)
        edges[edge_id] = GraphEdge(
            product_id=product_id,
            stable_id=edge_id,
            type=fact.predicate,
            from_id=from_id,
            to_id=to_id,
            properties={"evidence": fact.evidence},
            source_refs=[ref],
            confidence=fact.confidence,
            extraction_method="llm",
            last_seen=now,
        )

    return base.model_copy(
        update={
            "nodes": sorted(nodes.values(), key=lambda n: n.stable_id),
            "edges": sorted(edges.values(), key=lambda e: e.stable_id),
        }
    )


def graph_llm_supported_resource(resource: ResourceRef) -> bool:
    uri = resource.uri.lower()
    return uri.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".md", ".mdx"))


def _infer_entity_label(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs")):
        return "CodeFile"
    if lowered.endswith((".md", ".mdx")):
        return "Document"
    if name.isupper() and "_" in name:
        return "Config"
    if "/" in name or "." in name:
        return "Module"
    return "Function"


def _entity_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name
    return "unknown"


def _source_prompt(*, resource: ResourceRef, content: str) -> str:
    numbered = "\n".join(
        f"{line_no}: {line}" for line_no, line in enumerate(content.splitlines(), start=1)
    )
    return (
        f"resource_uri: {resource.uri}\n"
        f"mime: {resource.mime}\n\n"
        "Source with 1-based line numbers:\n"
        f"{numbered[:24000]}"
    )


def _node_id_for_entity(
    entity: LLMEntity,
    *,
    product_id: str,
    resource: ResourceRef,
    nodes: dict[str, GraphNode],
    aliases: dict[str, str],
    ref: SourceRef,
    now: str,
) -> str:
    key = _norm(entity.name)
    existing = aliases.get(key)
    if existing:
        node = nodes[existing]
        node.source_refs = _merge_refs(node.source_refs, [ref])
        node.confidence = max(node.confidence, 0.8)
        return existing
    resource_uri = entity.resource_uri or resource.uri
    stable_id = _sid(entity.label.lower(), product_id, resource_uri, entity.name)
    node = GraphNode(
        product_id=product_id,
        stable_id=stable_id,
        labels=[entity.label],
        properties={
            "name": entity.name,
            "resource_uri": resource_uri,
            "start_line": entity.start_line or ref.start_line,
            "end_line": entity.end_line or ref.end_line,
        },
        source_refs=[ref],
        confidence=0.8,
        extraction_method="llm",
        last_seen=now,
    )
    nodes[stable_id] = node
    aliases[key] = stable_id
    return stable_id


def _aliases(nodes: Sequence[GraphNode]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in nodes:
        props = node.properties
        values = [
            node.stable_id,
            props.get("name"),
            props.get("path"),
            props.get("title"),
            props.get("key"),
            props.get("resource_uri"),
            props.get("normalized_path"),
        ]
        for value in values:
            if isinstance(value, str) and value:
                out.setdefault(_norm(value), node.stable_id)
    return out


def _source_ref(
    product_id: str,
    source_key: str,
    resource: ResourceRef,
    fact: LLMGraphFact,
    now: str,
) -> SourceRef:
    _ = now
    return SourceRef(
        product_id=product_id,
        source_key=source_key,
        source_id=resource.source_id,
        resource_uri=resource.uri,
        anchor=f"{resource.uri}:{fact.start_line}",
        start_line=fact.start_line,
        end_line=fact.end_line,
    )


def _merge_refs(left: list[SourceRef], right: list[SourceRef]) -> list[SourceRef]:
    refs = {ref.anchor: ref for ref in left}
    for ref in right:
        refs[ref.anchor] = ref
    return list(refs.values())


def _sid(kind: str, product_id: str, *parts: str) -> str:
    return ":".join([kind, product_id, *[str(p).strip() for p in parts if str(p).strip()]])


def _edge_id(product_id: str, from_id: str, edge_type: str, to_id: str, anchor: str) -> str:
    raw = f"{product_id}|{from_id}|{edge_type}|{to_id}|{anchor}"
    return f"edge:{uuid.uuid5(_GRAPH_NS, raw)}"


def _norm(value: str) -> str:
    return value.strip().lower().replace("\\", "/")
