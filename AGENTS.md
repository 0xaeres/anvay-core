# Nexus — agent & contributor context

This is the Python backend for **Nexus**, a sovereign, MCP-native **context
engine**: it ingests an org's code + docs, runs a multi-agent LLM Council to draft
curated **skill files** (human-approved), serves them via MCP to any AI client,
and ships an **Assistant layer** that queries and acts on Jira/Confluence. The
sibling repo `../nexus-ui/` is the Next.js web UI.

## Read first

| Doc | What it is |
|---|---|
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | **The contributor guide — start here.** Concepts, code map, end-to-end traces, local dev, recipes, glossary. |
| [`ENGINEERING.md`](./ENGINEERING.md) | The formal spec — architecture, data model, API contracts, ADRs. |
| [`docs/`](./docs/) | Per-slice delivery status (`SLICE-*-STATUS.md`); the Assistant design (`ASSISTANT-LAYER.md`). |
| [`README.md`](./README.md) | Setup + quickstart. |

## The three invariants — never break these

1. **Product = root entity.** Every resource carries `product_id`; Qdrant shards
   on it, Neo4j filters on it. Code that crosses the product boundary is a
   tenancy bug.
2. **Skills compose, they don't override.** `composes_with` frontmatter defines
   the prerequisite chain; serve-time assembles `master + domain + org standards`.
3. **Humans approve, agents draft.** The Council and the Assistant write
   *proposals*. Nothing becomes real — a skill file, a Jira write — without an
   explicit human confirm.

## Conventions

- **Python 3.13+, managed by `uv`.** `from __future__ import annotations` at the
  top of every module.
- **Pydantic** for data crossing process boundaries; **dataclasses** for
  in-memory only; plain dicts only in tests.
- **Async by default** for anything touching the network or significant I/O.
- Import order stdlib → third-party → `nexus.*` (ruff/isort enforces).
- No `print()` — use `log = logging.getLogger(__name__)`.
- **Curate, don't proxy** (ADR-012): never expose a raw downstream MCP tool
  catalogue to an agent LLM — wrap it in a small curated facade.

## Before you commit

```bash
uv run ruff check nexus tests        # lint — must be clean
uv run pytest -q                     # tests — must be green (152 at last count)
```

- Add a test for every new public leaf function and every new API route.
- One logical change per commit; imperative subject; tag PRs with the slice.

## Don't

- Don't add runtime dependencies without discussion — the dep set is deliberate.
- Don't seed demo/placeholder products — the system boots empty; users onboard
  their own product via the wizard.
- Don't skip the proposal/approval step for any write action (Invariant 3).
- Don't store secrets in plaintext — OAuth tokens are Fernet-encrypted
  (`NEXUS_TOKEN_KEY`).
