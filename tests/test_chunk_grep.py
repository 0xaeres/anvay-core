from __future__ import annotations

import pytest

from nexus.retrieval.chunk_grep import grep_indexed_chunks


class _FakeIndexer:
    async def iter_chunk_payloads(self, *, product_id, vector_kind, batch_size):
        assert product_id == "demo"
        assert batch_size == 256
        if vector_kind == "code":
            yield "c1", {
                "resource_uri": "github:org/repo/nexus/api/routes/council.py",
                "start_line": 40,
                "context_path": "create_session",
                "content": "def create_session():\n    run_council()\n    return session_id\n",
            }
        else:
            yield "d1", {
                "resource_uri": "github:org/repo/README.md",
                "start_line": 10,
                "context_path": "Council",
                "content": "The council drafts proposals.\nHumans approve skills.\n",
            }


@pytest.mark.asyncio
async def test_chunk_grep_finds_payload_match_with_line_offset() -> None:
    hits = await grep_indexed_chunks(
        indexer=_FakeIndexer(),
        product_id="demo",
        query="runtime council flow",
    )

    assert hits
    assert hits[0].chunk_id == "c1"
    assert hits[0].file.endswith("nexus/api/routes/council.py")
    assert hits[0].line == 41
    assert "run_council" in hits[0].excerpt
