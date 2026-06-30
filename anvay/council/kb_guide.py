"""Static 'How To Use The Knowledge Base' skill section.

The MCP toolset Anvay serves is fixed (see `anvay/mcp_server/tools.py`), so the
routing guidance that tells an agent *when and how to query the KB/graph* is a
deterministic template — never LLM-authored. The synthesizer splices this in
verbatim, so the section never drifts and costs no tokens.

This is the skill's reason to exist: a product overview plus a lookup table into
the knowledge base. The overview is synthesized from evidence; this section is
the lookup table.
"""

from __future__ import annotations

KB_SECTION_TITLE = "How To Use The Knowledge Base"

# Body only — the synthesizer/template owns the `## How To Use The Knowledge Base`
# heading. `{product}` is interpolated with the product id.
_KB_USAGE_BODY = """\
This skill is a map, not the territory. Trust it for orientation — product
purpose, the system map, contracts, invariants, and where things live. For
anything you are about to change, read, or assert as current, treat the
knowledge base as source of truth and query it: code moves faster than this
skill.

Query the KB over MCP. Pick the tool by question shape:

- **Orientation / "which skill applies"** — `find_skills` then `get_skill`.
  Start here to load product context before deeper retrieval.
- **General "how does X work" / open-ended evidence** — `evidence_search_corpus`.
  Multi-channel retrieval (hybrid + grep + repo-map + graph-local + summaries),
  coverage-assessed. The default for most questions.
- **Relational / "what calls what", "what depends on this", impact** —
  `ask_product_graph`. Walks the per-product knowledge graph and synthesizes a
  cited answer. Use when the question is about connections, not a single file.
- **Exact symbol / constant / string lookup** — `grep_corpus`. Deterministic
  match against indexed chunks when you already know the token.
- **Symbol definition / signature lookup** — `query_code_context`.
- **Semantic / paraphrased search of code + docs** — `hybrid_search_corpus`
  (dense + BM25 + rerank) when you want the low-level retrieval directly.

Rules of thumb:
- Cite file:line from retrieved evidence, not from this skill, when you make a
  concrete claim about current code.
- If this skill and the KB disagree, the KB wins and this skill is stale —
  report it via `report_outcome` so it gets re-validated.
- Prefer `ask_product_graph` for "why/how are these connected" and
  `evidence_search_corpus` for "show me the relevant code".
"""


def kb_usage_section(product_id: str) -> str:
    """Render the canonical KB-usage section body for a product."""
    return _KB_USAGE_BODY.replace("{product}", product_id).strip()


def kb_usage_fill(product_id: str) -> str:
    """Heading + body, suitable for deterministic section splicing."""
    return f"## {KB_SECTION_TITLE}\n{kb_usage_section(product_id)}\n"
