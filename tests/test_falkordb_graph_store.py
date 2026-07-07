from __future__ import annotations

import os
import uuid

import pytest
from falkordb.edge import Edge
from falkordb.node import Node

from anvay.config import GraphStoreCfg
from anvay.graph.extractor import extract_resource_graph
from anvay.graph.store import FalkorGraphStore, _query_result_to_graph
from anvay.ingest.models import ResourceRef


class FakeQueryResult:
    def __init__(self, result_set):
        self.result_set = result_set


def test_falkordb_result_conversion_preserves_nodes_edges_and_source_refs() -> None:
    source_refs_js = (
        '[{"product_id":"prod","source_key":"source","source_id":"local:test",'
        '"resource_uri":"app.py","anchor":"app.py:1","start_line":1,"end_line":5}]'
    )
    source = Node(
        labels=["Function"],
        properties={
            "product_id": "prod",
            "stable_id": "symbol:prod:app.py:hello:Function",
            "name": "hello",
            "resource_uri": "app.py",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": source_refs_js,
        },
    )
    dest = Node(
        labels=["Module"],
        properties={
            "product_id": "prod",
            "stable_id": "module:prod:shared",
            "name": "shared",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": "[]",
        },
    )
    edge = Edge(
        source,
        "IMPORTS",
        dest,
        properties={
            "product_id": "prod",
            "stable_id": "edge:1",
            "from_id": "symbol:prod:app.py:hello:Function",
            "to_id": "module:prod:shared",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": source_refs_js,
        },
    )

    converted = _query_result_to_graph(FakeQueryResult([[source, dest, [edge]]]))

    assert [node.stable_id for node in converted.nodes] == [
        "module:prod:shared",
        "symbol:prod:app.py:hello:Function",
    ]
    assert converted.edges[0].type == "IMPORTS"
    assert converted.edges[0].from_id == "symbol:prod:app.py:hello:Function"
    assert converted.nodes[1].source_refs[0].anchor == "app.py:1"


def test_falkordb_result_conversion_handles_path_nodes_and_relationships() -> None:
    source = Node(
        labels=["Function"],
        properties={
            "product_id": "prod",
            "stable_id": "symbol:prod:app.py:read_token:Function",
            "name": "read_token",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": "[]",
        },
    )
    dest = Node(
        labels=["Function"],
        properties={
            "product_id": "prod",
            "stable_id": "symbol:prod:app.py:load_policy:Function",
            "name": "load_policy",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": "[]",
        },
    )
    edge = Edge(
        source,
        "CALLS",
        dest,
        properties={
            "product_id": "prod",
            "stable_id": "edge:call",
            "from_id": "symbol:prod:app.py:read_token:Function",
            "to_id": "symbol:prod:app.py:load_policy:Function",
            "confidence": 1.0,
            "extraction_method": "deterministic",
            "last_seen": "2026-01-01T00:00:00+00:00",
            "freshness": 1.0,
            "status": "active",
            "source_refs_js": "[]",
        },
    )

    converted = _query_result_to_graph(FakeQueryResult([[[source, dest], [edge]]]))

    assert [edge.stable_id for edge in converted.edges] == ["edge:call"]
    assert converted.paths == [
        {
            "node_ids": [
                "symbol:prod:app.py:read_token:Function",
                "symbol:prod:app.py:load_policy:Function",
            ],
            "edge_ids": ["edge:call"],
        }
    ]


@pytest.mark.asyncio
async def test_falkordb_store_contract_live() -> None:
    host = os.environ.get("ANVAY_FALKORDB_HOST")
    if not host:
        pytest.skip("set ANVAY_FALKORDB_HOST to run FalkorDB contract test")

    cfg = GraphStoreCfg(
        host=host,
        port=int(os.environ.get("ANVAY_FALKORDB_PORT", "6379")),
        username=os.environ.get("ANVAY_FALKORDB_USERNAME") or None,
        password=os.environ.get("ANVAY_FALKORDB_PASSWORD") or None,
        ssl=os.environ.get("ANVAY_FALKORDB_SSL") == "1",
        graph_prefix="anvay_test",
    )
    store = FalkorGraphStore(cfg)
    product_id = f"contract-{uuid.uuid4().hex[:8]}"
    resource = ResourceRef(
        source_id="local:contract",
        uri="app.py",
        mime="text/x-python",
    )
    extraction = extract_resource_graph(
        product_id=product_id,
        source_key="source",
        resource=resource,
        content=(
            "import os\n\n"
            "def hello():\n"
            "    token = os.environ.get('TOKEN_SECRET')\n"
            "    return token or 'missing'\n"
        ),
    )

    try:
        await store.ensure_schema()
        fact_ids = await store.upsert_resource_graph(extraction)
        assert fact_ids
        seed = next(node.stable_id for node in extraction.nodes if "file:" in node.stable_id)
        traversal = await store.traverse(
            product_id=product_id,
            seed_ids=[seed],
            edge_types=["CONTAINS", "DECLARES", "READS"],
            max_depth=2,
            limit=10,
        )
        assert traversal.nodes
        assert any(edge.type == "READS" for edge in traversal.edges)
        assert await store.retire_resource_graph(
            product_id=product_id,
            fact_ids=fact_ids[:1],
        ) == 1
        await store.delete_product(product_id=product_id)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_traverse_uses_directed_arrow_for_directional_types() -> None:
    """All-directional edge types → outward `->` pattern; mixed → undirected."""

    class _CapturingGraph:
        def __init__(self):
            self.queries: list[str] = []

        async def ro_query(self, query, params, timeout=None):
            self.queries.append(query)
            return FakeQueryResult([])

    cfg = GraphStoreCfg(host="x", graph_prefix="anvay_test")
    store = FalkorGraphStore(cfg)
    fake = _CapturingGraph()
    store._graph = lambda product_id: fake  # type: ignore[assignment]

    async def _noop_schema(product_id):
        return None

    store._ensure_product_schema = _noop_schema  # type: ignore[assignment]

    await store.traverse(
        product_id="p", seed_ids=["s1"], edge_types=["CALLS", "IMPORTS"], max_depth=2
    )
    assert fake.queries[-1].rstrip().count("]->(n)") == 1

    await store.traverse(
        product_id="p", seed_ids=["s1"], edge_types=["CALLS", "CONTAINS"], max_depth=2
    )
    # Mixed directional + structural → undirected close.
    assert "]-(n)" in fake.queries[-1] and "]->(n)" not in fake.queries[-1]

    await store.traverse(
        product_id="p", seed_ids=["s1"], edge_types=["CALLS", "IMPORTS"], max_depth=2
    )
    assert ":CALLS|IMPORTS" in fake.queries[-1]
