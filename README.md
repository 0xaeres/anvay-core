# Nexus — Context Engine for Engineering Teams

Nexus is a **sovereign, MCP-native context engine** for your codebase. It ingests your code and docs, runs an LLM Council to draft curated skill files with human approval, and serves those skills back via MCP to any AI client (Claude, Cursor, Continue, etc.).

Every AI tool your team uses gets grounded in *your* actual code and conventions — not hallucinated from general training data.

It also ships an **Assistant layer** — a conversational + action interface that queries Jira and Confluence and takes human-confirmed actions (create subtasks, transition issues, update docs), reachable from coding agents (via MCP) and a chat panel in the web UI. See [`docs/ASSISTANT-LAYER.md`](./docs/ASSISTANT-LAYER.md).

```
Your codebase + docs
        │
        ▼
  Nexus ingests (MCP connectors: GitHub, Jira, Confluence)
        │
        ▼
  LLM Council drafts skill files  ←  human reviews + approves in the UI
        │
        ▼
  Skills served via MCP to Claude / Cursor / Continue / any agent
        │
        ▼
  Agents give grounded, cited, org-specific answers
```

---

## What is a skill file?

A skill file is a plain Markdown + YAML document that tells an agent *how to work in your codebase*: patterns to follow, pitfalls to avoid, architectural context, domain vocabulary. Skills are versioned, human-ratified, and composable.

```yaml
---
kind: master          # one master per product; describes the product itself
product: my-api
version: 3
confidence: 0.91
composes_with: []
provenance:
  validated_by: alice@example.com
  validated_at: 2026-05-18T00:00:00Z
---

# My API — Master Skill

## Architecture
The API is a FastAPI + PostgreSQL service…

## Domain Vocabulary
| Term | Meaning |
|---|---|
| Workspace | A tenant-scoped container for all resources |

## Positive Patterns
Always use `get_or_404` for resource lookups. See [src/deps.py:42].

## Anti-Patterns
Never access `db.session` outside a dependency — session lifetime is managed by the DI container.
```

---

## Local setup (Apple Silicon dev)

### Prerequisites

```bash
# Python env
uv sync                               # installs everything from uv.lock

# One-time: local model servers
brew install llama.cpp ollama
mkdir -p models
# Download into models/:
#   jina-embeddings-v4.Q4_K_M.gguf   (embedder)
#   jina-reranker-v3.Q4_K_M.gguf     (reranker)
# From: https://huggingface.co/jinaai
```

### Configure

```bash
cp nexus.yaml.example nexus.yaml
cp .env.example .env
```

Edit `nexus.yaml`:
- `skills_repo` — a Git repo where approved skills are pushed (e.g. `git@github.com:myorg/nexus-skills.git`)
- `connectors` — add your GitHub org/repos, Confluence spaces, Jira project keys

Edit `.env`:
- `DEEPINFRA_API_KEY` — council + PR review LLMs (get one at deepinfra.com)
- `GITHUB_TOKEN` — for the GitHub connector
- `GITHUB_WEBHOOK_SECRET` — for PR review automation (can leave blank for local dev)

### Start infrastructure

```bash
# Qdrant (vector store) + Neo4j (graph) + Langfuse (tracing) + Postgres
docker compose up -d

# Local model servers (runs on Metal — keep these terminals open)
make services-up
```

### Run the API

```bash
uv run uvicorn nexus.api.app:app --port 8000 --reload
# → http://localhost:8000/health  {"status":"ok"}
```

### Run the UI

```bash
cd ../nexus-ui
npm install
npm run dev
# → http://localhost:3000
```

On first boot there are no products. The app opens the onboarding wizard at `http://localhost:3000/onboarding` — create your first product there.

---

## Docker (all-in-one)

```bash
docker compose --profile full up -d
```

This brings up Qdrant, Neo4j, Langfuse, Postgres, **and** the Nexus API. The UI still runs from `nexus-ui/` with `npm run dev`.

> **Apple Silicon:** llama.cpp embedding/reranker services always run on the host (Metal acceleration). For Linux + NVIDIA, point `models.embedding.url` / `models.reranker.url` at any OpenAI-compatible server.

---

## End-to-end flow

### 1. Onboard a product via the UI

Visit `http://localhost:3000/onboarding` — the 4-step wizard:
1. Name your product
2. Connect sources (GitHub repo, Confluence space, etc.)
3. Trigger ingestion (watch the live sync log)
4. Start a council session to draft the master skill

### 2. Onboard via CLI (scriptable)

```bash
# Ingest a local codebase
uv run nexus ingest --product <your-product-id> --path /path/to/repo

# Draft a skill via Council
uv run nexus council draft \
  --product <your-product-id> \
  --topic "authentication middleware" \
  --kind product_domain

# Approve from the UI
open http://localhost:3000/p/<your-product-id>/proposals
```

Replace `<your-product-id>` with the product ID you created in the UI (e.g. `my-api`, `backend`, whatever you named it).

### 3. Use Nexus from Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nexus": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/nexus",
        "run", "nexus-mcp-server",
        "--product", "<your-product-id>"
      ],
      "env": {
        "NEXUS_CONFIG": "/absolute/path/to/nexus/nexus.yaml"
      }
    }
  }
}
```

Claude Desktop will now call `find_skills`, `query_code_context`, and `hybrid_search_corpus` against your indexed codebase.

---

## Continuous automation

Once the daemon is running (`make daemon-up`), Nexus watches your configured sources and automatically:

- Re-indexes changed files within ~5 seconds of a push
- Posts structured PR review comments citing relevant skills
- Generates release notes on tag push

These require `GITHUB_TOKEN` and a configured webhook endpoint.

---

## The Assistant — query and act on Jira/Confluence

The Assistant layer adds a conversational + action interface on top of Nexus's knowledge:

- **Ask** — "summarize JIRA-1234", "search Confluence for the on-call runbook"
- **Act** — "break JIRA-1234 into subtasks" drafts a plan you confirm before anything is written

Two ways to use it:

- **Web UI** — the chat panel at `/p/<product>/assistant`
- **Coding agents** — the Nexus MCP server exposes `assistant_ask`, `assistant_get_jira_issue`, `assistant_confirm_action`, etc., so Claude Desktop / Cursor can query and act mid-task

Live Jira/Confluence access requires the Atlassian integration (`atlassian.enabled` in `nexus.yaml`) and per-user OAuth — without it the Assistant runs against stubbed data. Writes always require explicit human confirmation. Full design: [`docs/ASSISTANT-LAYER.md`](./docs/ASSISTANT-LAYER.md).

---

## Project layout

```
nexus/
├── nexus/
│   ├── api/          FastAPI routes (/products, /council, /skills, /assistant, /auth, …)
│   ├── ingest/       Chunking, embedding, indexing pipeline
│   ├── retrieval/    5-stage RAG: sparse+dense → classifier → HyDE → RRF → rerank
│   ├── council/      LangGraph multi-agent council (6 agents)
│   ├── assistant/    Assistant layer — agent loop, capabilities, action proposals
│   ├── auth/         Per-user OAuth (Atlassian) + token encryption
│   ├── skills/       Skill models, store, seed files
│   ├── connectors/   MCP client (stdio + remote) + local_fs connector
│   ├── graph/        Neo4j GraphRAG layer
│   ├── mcp_server/   MCP server (stdio) — what Claude Desktop connects to
│   ├── tasks/        PR review + changelog task runners
│   ├── daemon.py     Continuous index daemon
│   └── config.py     nexus.yaml loader
├── evals/            RAGAS + code-retrieval eval runners + golden set
├── tests/            104 unit + integration tests
├── scripts/          resilience-smoke.sh, model download helpers
├── nexus.yaml.example
└── docker-compose.yml
```

---

## Quality gates

```bash
uv run pytest                                               # 104 tests
uv run python -m evals.run_ragas --golden evals/golden.jsonl
uv run python -m evals.run_code_eval --golden evals/golden.jsonl
bash scripts/resilience-smoke.sh
```

| Metric | Gate |
|---|---|
| `faithfulness` | ≥ 0.85 |
| `answer_relevancy` | ≥ 0.80 |
| `context_recall` | ≥ 0.75 |
| `nDCG@10` | ≥ 0.75 |
| `Recall@10` | ≥ 0.80 |
| Pairwise preference | ≥ 0.85 |

CI (`.github/workflows/ci.yml`) runs lint + tests + RAGAS regression on every PR and fails if faithfulness drops > 5% from baseline.

---

## Documentation

| File | What it covers |
|---|---|
| [`AGENTS.md`](./AGENTS.md) | Quick orientation for AI agents & new contributors — invariants, conventions, commit checks. |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | **New contributor guide** — code map, end-to-end traces, dev workflow, recipes. Start here. |
| [`ENGINEERING.md`](./ENGINEERING.md) | Full architecture spec, data model, ADRs, API surface |
| [`INTEGRATION.md`](./INTEGRATION.md) | UI ↔ backend cutover map |
| [`docs/UI-CUTOVER-STATUS.md`](./docs/UI-CUTOVER-STATUS.md) | End-to-end demo walkthrough |
| [`docs/SLICE-*-STATUS.md`](./docs/) | Per-slice delivery notes |
| [`docs/ASSISTANT-LAYER.md`](./docs/ASSISTANT-LAYER.md) | Design — conversational + action layer over Jira/Confluence |
| [`docs/SLICE-8-STATUS.md`](./docs/SLICE-8-STATUS.md) | Assistant layer — delivery status (Increment 1 shipped) |
| [`../nexus-ui/DESIGN.md`](../nexus-ui/DESIGN.md) | UI design system rules |

---

## License

Proprietary — internal use only.
