"""Deterministic structural summaries derived from graph extraction."""

from __future__ import annotations

from collections import Counter

from nexus.graph.models import GraphExtraction
from nexus.ingest.models import Chunk, ChunkKind, ResourceRef

SUMMARY_CONTEXT_PATH = "Graph summary"
COMMUNITY_CONTEXT_PATH = "Graph community summary"


def graph_summary_chunk(
    *, product_id: str, resource: ResourceRef, extraction: GraphExtraction
) -> Chunk | None:
    """Build one source-backed structural summary chunk for graph/vector retrieval."""
    if not extraction.nodes and not extraction.edges:
        return None
    lines = [
        f"Structural summary for {resource.uri}.",
        _node_summary(extraction),
        _edge_summary(extraction),
        _symbol_summary(extraction),
    ]
    content = "\n".join(line for line in lines if line).strip()
    if not content:
        return None
    return Chunk(
        product_id=product_id,
        resource=resource,
        content=content,
        start_line=0,
        end_line=0,
        kind=ChunkKind.DOC,
        context_path=SUMMARY_CONTEXT_PATH,
        context_summary="Source-backed structural graph summary for retrieval.",
    )


def is_summary_chunk(chunk: Chunk) -> bool:
    return chunk.start_line == 0 and chunk.end_line == 0 and chunk.context_path == SUMMARY_CONTEXT_PATH


def graph_community_summary_chunk(
    *, product_id: str, resource: ResourceRef, extraction: GraphExtraction
) -> Chunk | None:
    """Build a compact connected-subgraph memory chunk for broad graph search."""
    if not extraction.edges:
        return None
    summary_lines = [
        _flow_summary(extraction),
        _runtime_summary(extraction),
        _doc_code_summary(extraction),
    ]
    if not any(summary_lines):
        return None
    lines = [f"Graph community for {resource.uri}.", *summary_lines]
    content = "\n".join(line for line in lines if line).strip()
    if not content:
        return None
    return Chunk(
        product_id=product_id,
        resource=resource,
        content=content,
        start_line=0,
        end_line=1,
        kind=ChunkKind.DOC,
        context_path=COMMUNITY_CONTEXT_PATH,
        context_summary="Source-backed graph community summary for retrieval.",
    )


def is_community_summary_chunk(chunk: Chunk) -> bool:
    return chunk.start_line == 0 and chunk.end_line == 1 and chunk.context_path == COMMUNITY_CONTEXT_PATH


def _node_summary(extraction: GraphExtraction) -> str:
    labels = Counter(label for node in extraction.nodes for label in node.labels)
    if not labels:
        return ""
    parts = ", ".join(f"{name}={count}" for name, count in sorted(labels.items()))
    return f"Graph nodes: {parts}."


def _edge_summary(extraction: GraphExtraction) -> str:
    edges = Counter(edge.type for edge in extraction.edges)
    if not edges:
        return ""
    parts = ", ".join(f"{name}={count}" for name, count in sorted(edges.items()))
    return f"Graph relationships: {parts}."


def _symbol_summary(extraction: GraphExtraction) -> str:
    names: list[str] = []
    for node in extraction.nodes:
        props = node.properties
        name = props.get("name") or props.get("path") or props.get("title") or props.get("key")
        if isinstance(name, str) and name:
            names.append(name)
    if not names:
        return ""
    sample = ", ".join(names[:24])
    return f"Important entities: {sample}."


def _flow_summary(extraction: GraphExtraction) -> str:
    names = _names_by_id(extraction)
    parts = []
    for edge in extraction.edges:
        if edge.type not in {"CALLS", "DEPENDS_ON", "HANDLES", "IMPLEMENTS", "PART_OF_FLOW"}:
            continue
        parts.append(f"{names.get(edge.from_id, edge.from_id)} -{edge.type}-> {names.get(edge.to_id, edge.to_id)}")
    if not parts:
        return ""
    return "Flows: " + "; ".join(parts[:16]) + "."


def _runtime_summary(extraction: GraphExtraction) -> str:
    names = _names_by_id(extraction)
    parts = []
    for edge in extraction.edges:
        if edge.type not in {"READS", "WRITES", "CONSTRAINS"}:
            continue
        parts.append(f"{names.get(edge.from_id, edge.from_id)} -{edge.type}-> {names.get(edge.to_id, edge.to_id)}")
    if not parts:
        return ""
    return "Runtime/config/data: " + "; ".join(parts[:16]) + "."


def _doc_code_summary(extraction: GraphExtraction) -> str:
    names = _names_by_id(extraction)
    parts = []
    for edge in extraction.edges:
        if edge.type not in {"DOCUMENTS", "MENTIONS", "COVERS"}:
            continue
        parts.append(f"{names.get(edge.from_id, edge.from_id)} -{edge.type}-> {names.get(edge.to_id, edge.to_id)}")
    if not parts:
        return ""
    return "Docs/tests: " + "; ".join(parts[:16]) + "."


def _names_by_id(extraction: GraphExtraction) -> dict[str, str]:
    out = {}
    for node in extraction.nodes:
        props = node.properties
        name = props.get("name") or props.get("path") or props.get("title") or props.get("key")
        if isinstance(name, str) and name:
            out[node.stable_id] = name
    return out
