# Contributing to Nexus — A New Developer's Guide

> **Audience:** software engineers with basic familiarity with LLMs (what a prompt is, what a token is) and RAG (the idea of "embed, store, retrieve").
>
> **Goal:** by the end of this document, you should be able to (a) navigate this codebase, (b) understand every major technology choice, (c) trace any request end-to-end, (d) run things locally, and (e) make your first contribution.
>
> **Style:** we explain every non-obvious concept the first time it appears. If you find a term that isn't introduced, please open an issue — it's a doc bug.

---

## Table of contents

0. [Before you start](#0-before-you-start)
1. [Orientation — what Nexus is and isn't](#1-orientation--what-nexus-is-and-isnt)
2. [Concepts you'll need](#2-concepts-youll-need)
   - 2.1 [MCP — Model Context Protocol](#21-mcp--model-context-protocol)
   - 2.2 [Daemons and background processes](#22-daemons-and-background-processes)
   - 2.3 [RAG techniques used here](#23-rag-techniques-used-here)
   - 2.4 [Vector stores and graph stores](#24-vector-stores-and-graph-stores)
   - 2.5 [LangGraph and multi-agent orchestration](#25-langgraph-and-multi-agent-orchestration)
   - 2.6 [Server-Sent Events (SSE)](#26-server-sent-events-sse)
   - 2.7 [Tenant isolation via sharding](#27-tenant-isolation-via-sharding)
   - 2.8 [Circuit breakers and degradation](#28-circuit-breakers-and-degradation)
   - 2.9 [What is a "skill" in Nexus](#29-what-is-a-skill-in-nexus)
3. [The mental model — three invariants](#3-the-mental-model--three-invariants)
4. [How the pieces talk — bytes on the wire](#4-how-the-pieces-talk--bytes-on-the-wire)
5. [Backend code map (`nexus/`)](#5-backend-code-map-nexus)
6. [Frontend code map (`nexus-ui/`)](#6-frontend-code-map-nexus-ui)
7. [End-to-end traces](#7-end-to-end-traces)
8. [Local development workflow](#8-local-development-workflow)
9. [Hands-on tour — exercises to learn by tinkering](#9-hands-on-tour--exercises-to-learn-by-tinkering)
10. [Recipes — common contribution tasks](#10-recipes--common-contribution-tasks)
11. [Testing](#11-testing)
12. [Code conventions](#12-code-conventions)
13. [Glossary — quick reference](#13-glossary--quick-reference)
14. [Further reading](#14-further-reading)

---

## 0. Before you start

### Knowledge we assume

- **Python** — async/await, type hints, packages, pip/uv. Pydantic helps but is explained below.
- **TypeScript / React** — hooks, components, props. Next.js App Router is briefly explained.
- **HTTP basics** — REST routes, status codes, JSON bodies. Webhooks and SSE are explained below.
- **LLMs (basic)** — you know what a prompt and a completion are. You've probably called OpenAI or Anthropic APIs.
- **RAG (basic)** — you know that text gets *embedded* into vectors, stored in a vector DB, and looked up by cosine similarity.

### Knowledge we explain

- MCP (Model Context Protocol)
- LangGraph (multi-agent orchestration)
- HyDE, RRF, reranking, BM25, cross-encoders
- GraphRAG
- Tree-sitter
- Circuit breakers
- Server-Sent Events
- Daemons and background tasks in Python/asyncio
- Webhook HMAC signing
- Custom sharding in Qdrant

If anything else trips you up, that's a documentation bug — open an issue.

### Time investment

| Outcome | Time |
|---|---|
| Read this doc once | ~1.5 hours |
| Get the system running locally | ~30 minutes (excludes GGUF download) |
| Make a trivial change with a test, lint clean | ~2 hours total for someone new |
| Internalize the council code path | a couple of days of poking |

---

## 1. Orientation — what Nexus is and isn't

### One paragraph

Nexus is a **context engine** for engineering organizations. It reads your team's code and docs, runs a panel of LLM "agents" (the **Council**) to draft *curated guidance files* about your codebase, and then serves those files to any AI client (Claude Desktop, Cursor, Continue, etc.) via **MCP** (explained in §2.1). Humans approve every file before it's served. The result: every AI tool your team uses is grounded in *your* code, not in generic training data.

### What this is *not*

- **Not a coding agent** — Nexus doesn't write your code. It tells *other* AI tools how to work in your codebase.
- **Not a chat product** — there's no chat interface. The UI is for managing skills, sources, and council sessions.
- **Not a SaaS** — sovereign by design. You self-host. Your code never leaves your infrastructure (with the exception of cloud LLM API calls, which you can swap for local models).
- **Not auto-merge** — every skill file requires a human approval click. The Council *drafts*; humans *validate*.

### The two repos

```
~/Desktop/projects/
├── nexus/        ← Python backend, CLI, MCP server, ingestion daemon (this repo)
└── nexus-ui/     ← Next.js 16 web app (sibling repo)
```

They communicate over plain HTTP + Server-Sent Events. FastAPI runs on `:8000`, Next.js on `:3000`. No shared type package — Pydantic models in Python and TypeScript types in `lib/types.ts` are kept aligned by convention (with TypeScript checking and tests catching drift).

### The technology stack

| Layer | Tool | Why |
|---|---|---|
| Backend HTTP | **FastAPI** | async-first, Pydantic-integrated, auto-generated OpenAPI |
| Backend lang | **Python 3.13+** | rich LLM ecosystem; we use `uv` for env management |
| Frontend | **Next.js 16** (App Router) + **React 19** | SSR + RSC + file-based routing |
| Frontend lang | **TypeScript 5** | type-checked client-side |
| Styling | **Tailwind v4** + shadcn-style primitives + Radix | utility-first; design tokens live in `app/globals.css` |
| LLM orchestration | **LangGraph** | stateful multi-agent graphs, persistence built-in |
| Embeddings | **Jina v4** via llama.cpp (Metal on macOS) | high-quality, runs locally |
| Reranker | **Jina reranker v3** via llama.cpp | cross-encoder; final ranking stage |
| Vector store | **Qdrant** | supports custom sharding (we shard by `product_id`) |
| Graph store | **Neo4j** | for relational queries — GraphRAG |
| Local light LLM | **Ollama** | HyDE, classifier, relation extraction |
| Cloud LLM | **DeepInfra** (default) — swappable | council reasoning, PR review, changelog |
| Observability | **Langfuse** + **OpenTelemetry** | LLM traces + costs |
| Storage | **SQLite** for registry / queue / checkpoints, **Postgres** for Langfuse |

If a tool name is unfamiliar, jump to §2 (Concepts) — they're all explained there.

---

## 2. Concepts you'll need

### 2.1 MCP — Model Context Protocol

**MCP** ([modelcontextprotocol.io](https://modelcontextprotocol.io)) is Anthropic's open standard for connecting AI clients (Claude Desktop, Cursor, etc.) to external **tools** and **resources** (databases, APIs, file systems, custom services). Think of it as **"USB-C for AI"**: one protocol, many providers and consumers.

There are two roles:

| Role | What it does | Example |
|---|---|---|
| **MCP server** | Exposes tools/resources over a transport (stdio or HTTP) | `nexus-mcp-server`, `mcp-github-server` |
| **MCP client** | Discovers servers, subscribes to their tools, invokes them | Claude Desktop, Cursor |

The protocol is JSON-RPC over stdio (most common) or HTTP. A server advertises:

- **Tools** — callable functions with JSON Schemas, e.g. `find_skills(query: str) → list[Skill]`.
- **Resources** — readable URIs, e.g. `nexus://skills/my-api/master`.

#### Nexus is *both* an MCP server and an MCP client

Most projects are only one. Nexus is both:

- **As an MCP server** (`nexus/mcp_server/server.py`): exposes 3 tools (`find_skills`, `query_code_context`, `hybrid_search_corpus`) and a meta-skill resource. This is what Claude Desktop connects to.
- **As an MCP client** (`nexus/connectors/mcp_client.py`): connects *to* connector servers (GitHub MCP server, Jira MCP server, etc.) to pull your code and docs into the index.

That dual role is what gives the project its name — Nexus sits at the *nexus* between source connectors and AI clients.

#### Why MCP and not REST?

For ingestion: connectors publish *notifications* when a resource changes (file pushed, ticket updated). Real-time, persistent connections fit better than REST polling. MCP gives us that for free.

For serving skills: AI clients are MCP-native already. Speaking MCP means zero adapter code on the client side.

### 2.2 Daemons and background processes

A **daemon** (Unix term, pronounced "demon" or "day-mon") is a long-running background process that doesn't have a user-interactive shell — it just sits there doing work in response to events. Web servers, cron, syslog are all daemons.

In Nexus, `nexus/daemon.py` is a long-running asyncio task that:

1. **Bootstrap phase** — does a one-time full ingest across all configured connectors at startup.
2. **Watch phase** — loops forever over `connector_manager.updates()`, an async iterator that yields resource-change events. For each event, it re-indexes that single resource incrementally.

```python
# Stripped-down version of what daemon.py does
async def run_daemon(config, product_id):
    manager = ConnectorManager(config.connectors)
    await manager.bootstrap_all(product_id)        # full ingest
    async for update in manager.updates():         # never returns
        await reindex_resource(update, product_id) # incremental
```

The daemon is what makes the system **continuously updated** rather than batch-indexed. It survives connector crashes (the manager reconnects with backoff) and process restarts (Qdrant + Neo4j are persistent).

You start it with `make daemon-up` or by importing and running `run_daemon(...)` from your own entry point.

### 2.3 RAG techniques used here

You already know the basics: chunk text → embed → store → retrieve nearest by cosine similarity. Nexus uses several refinements that boost quality. Each is implemented as one leaf module under `nexus/retrieval/`.

#### a) BM25 (sparse retrieval)

**Dense embeddings** capture semantic similarity ("car" ≈ "automobile"). They miss **lexical exact matches** — function names, error strings, rare identifiers like `xyzzy_2024`. That's where BM25 shines.

**BM25** is a 1990s information-retrieval algorithm that scores a document against a query based on term frequency (how often each query word appears in the doc) and inverse document frequency (how rare each query word is across the corpus). It's "sparse" because each document is represented as a sparse vector of TF-IDF weights — most positions are zero.

We compute sparse vectors at ingest time using `fastembed`'s BM25 encoder, store them in Qdrant in a *named vector* alongside the dense one, and let Qdrant rank by BM25 server-side at query time.

#### b) Hybrid retrieval + Reciprocal Rank Fusion (RRF)

We always run **both** dense and sparse retrieval. Then we need to combine the two ranked lists into one.

**Reciprocal Rank Fusion (RRF)** is a deceptively simple fusion algorithm. For each result, sum `1 / (k + rank_in_list_i)` across all lists, where `k=60` is a constant. Rank 1 contributes `1/61`, rank 2 contributes `1/62`, etc. Documents that rank highly in *any* list float to the top, and the algorithm is robust to score-scale differences (dense scores are in [0,1], BM25 scores are unbounded — RRF doesn't care).

Implementation: 30 lines in `nexus/retrieval/hybrid.py`.

#### c) HyDE — Hypothetical Document Embeddings

The query "how do I validate a JWT?" and the document `function validateJWT(token) {...}` are *not* close in embedding space — the query is in question-form, the doc is code. **HyDE** fixes this:

1. Send the query to a small LLM with the prompt "Write a hypothetical document that would answer this question."
2. The LLM emits something like `function validateJWT(token) { /* check signature, expiry, ... */ }`.
3. Embed *that*, not the original query.

The hallucinated document is much closer in embedding space to the real one. Our HyDE module (`nexus/retrieval/hyde.py`) uses Ollama for this since it needs to be cheap and fast.

#### d) Reranking with a cross-encoder

A bi-encoder (what produces our embeddings) maps query and document to *separate* vectors, then compares with cosine. Fast but lossy.

A **cross-encoder** processes query and document *together* in a single transformer pass and emits a single relevance score. Much higher quality but expensive — you can't do this over millions of documents.

The standard trick: use the bi-encoder for **first-stage retrieval** (top ~50 candidates), then the cross-encoder to **rerank** those 50 down to top ~10. We use Jina reranker v3 via llama.cpp's `/reranking` endpoint. See `nexus/retrieval/reranker.py`.

#### e) GraphRAG

Pure-vector RAG misses relational queries:

> "Which ADR motivated the change in commit abc123?"

This isn't a similarity question — it's a *traversal* question. To answer it you need a graph: `Commit -> implements -> ADR`.

**GraphRAG** indexes triples like `(ADR-007, motivates, commit:abc123)` into a graph database (Neo4j here). At retrieval time, after the vector hits come back, we **expand** them by following graph edges — given a hit on a commit, we look up the ADR it implements and include that too.

Our relation extractor (`nexus/ingest/relation_extractor.py`) uses a light LLM at ingest time to extract triples from doc chunks. The retrieval-side expansion lives in `nexus/retrieval/graph.py`.

#### The full Nexus pipeline (5 stages)

```
User query
   │
   ▼
[1] Classifier ── simple? skip HyDE.                  (retrieval/classifier.py)
   │
   ▼
[2] HyDE        ── generate hypothetical doc          (retrieval/hyde.py)
   │
   ▼
[3a] Dense ───────┐                                   (Qdrant dense vector)
[3b] BM25 ────────┤
[3c] Graph hop ───┤── all merged via RRF              (retrieval/hybrid.py)
   │
   ▼
[4] Rerank      ── cross-encoder picks top-k           (retrieval/reranker.py)
   │
   ▼
[5] Quality gate ─ drop if score < 0.3 (configurable)
   │
   ▼
Top results returned with citations
```

Orchestrated by `nexus/retrieval/pipeline.py`. Each stage is wrapped in a circuit breaker (§2.8).

### 2.4 Vector stores and graph stores

#### Qdrant (vector store)

A **vector database** stores high-dimensional floating-point vectors and supports approximate-nearest-neighbor (ANN) search over them. We use [Qdrant](https://qdrant.tech) because of two features:

- **Named vectors** — a single record can hold both a dense and a sparse vector. We use this for hybrid retrieval (§2.3b).
- **Custom sharding** — you can shard a collection on an arbitrary key. We shard on `product_id` (§2.7) for hard tenant isolation.

Two collections: `nexus_code` and `nexus_text`. A third, `nexus_cache`, holds semantic cache entries.

#### Neo4j (graph store)

A **graph database** stores nodes and edges with properties, queried with Cypher (a SQL-like graph language). We use [Neo4j](https://neo4j.com) for the GraphRAG layer (§2.3e). Nodes are entities (commits, ADRs, tickets, files); edges are extracted relations (`implements`, `references`, `closes`).

Cypher example — find the ADR that motivated a commit:

```cypher
MATCH (c:Commit {id: $commit_id})-[:IMPLEMENTS]->(a:ADR) RETURN a
```

See `nexus/graph/store.py` for the wrapper.

### 2.5 LangGraph and multi-agent orchestration

[**LangGraph**](https://langchain-ai.github.io/langgraph/) is a Python library for building **stateful**, **graph-shaped** workflows around LLM calls. It's designed for cases where you have several agents (or tool-using nodes) that need to pass state, run in parallel, fan-out and fan-in, and resume from a checkpoint after a crash.

The core abstractions:

- **State** — a TypedDict that flows through the graph. Each node can read it and return a partial update that's merged in.
- **Node** — a function `async def f(state) → dict`. The dict is merged into state.
- **Edge** — a directed connection. Can be conditional.
- **Graph** — the assembled DAG. Compile, then `.ainvoke(initial_state)` to run.
- **Checkpointer** — a backend that persists state at every transition. We use `SqliteSaver` so a process crash mid-session can resume from the last checkpoint.

The Nexus council topology:

```
                START
                /  \
        Archaeologist  Domain-Expert    ← run in parallel
                \  /
             Synthesizer                ← reads both outputs
                  │
              Adversary                  ← critiques
                  │
                 END
```

Code:

```python
# nexus/council/graph.py (simplified)
graph = StateGraph(CouncilState)
graph.add_node("archaeologist", archaeologist.run)
graph.add_node("domain_expert", domain_expert.run)
graph.add_node("synthesizer", synthesizer.run)
graph.add_node("adversary", adversary.run)

graph.add_edge(START, "archaeologist")
graph.add_edge(START, "domain_expert")
graph.add_edge("archaeologist", "synthesizer")
graph.add_edge("domain_expert", "synthesizer")
graph.add_edge("synthesizer", "adversary")
graph.add_edge("adversary", END)

compiled = graph.compile(checkpointer=SqliteSaver(...))
```

Each node is a single function in `nexus/council/agents/`. They call retrieval + ChatClient + return partial state. Nothing magic.

### 2.6 Server-Sent Events (SSE)

**Server-Sent Events** is a tiny standard for one-way streaming from a server to a browser over HTTP. It's simpler than WebSockets (one-way, plain text, auto-reconnects), and the browser API is one line:

```js
const source = new EventSource('/council/sessions/abc/stream')
source.onmessage = (e) => console.log(e.data)
```

The server just keeps the connection open and writes lines like:

```
event: message
data: {"agent": "archaeologist", "body": "Found 4 relevant chunks..."}

event: cost
data: {"agent": "archaeologist", "prompt_tokens": 1280}

event: session_end
data: {"proposal_id": "p_..."}
```

Each `\n\n` flushes a message to the client.

We use SSE for:

- **Council deliberation streaming** — `/council/sessions/{id}/stream`.
- **Source sync logs** — `/products/{p}/sources/{s}/log`.

Why SSE over WebSockets? We only need server→client streaming, and SSE is dirt-simple. Server-side: FastAPI's `StreamingResponse`. Client-side: a thin hook at `nexus-ui/lib/hooks/useEventStream.ts`.

### 2.7 Tenant isolation via sharding

If two products share the same Qdrant collection, a query for product A could accidentally retrieve product B's chunks. That's a tenancy bug we cannot have.

Qdrant supports **custom sharding** — you can declare that a collection is sharded on an arbitrary key, and queries that specify the shard key only hit the shard for that tenant. We use `product_id` as the shard key:

```python
# Excerpt from nexus/ingest/indexer.py
client.create_collection(
    collection_name="nexus_code",
    sharding_method=models.ShardingMethod.CUSTOM,
)
# every point has a `product_id` payload field used as the shard key
```

Effect: a query that filters on `product_id == "my-api"` is *physically* impossible to leak hits from other products, because the data isn't on those shards. We get correctness from the database, not just from query filters.

Neo4j tenant isolation works differently — every node has a `product_id` property and every query filters on it. Less strict than Qdrant's physical sharding but adequate for the graph layer.

### 2.8 Circuit breakers and degradation

A **circuit breaker** is a resilience pattern from distributed systems. The idea: if an external dependency is failing, don't keep retrying — *fail fast*, recover gracefully, and probe periodically to see if it's back.

States:

```
  [CLOSED] ── N failures ──→ [OPEN] ── timeout elapsed ──→ [HALF-OPEN]
     ▲                          │                              │
     └──── probe succeeds ──────┴──── probe fails ──────────────┘
```

In `nexus/retrieval/circuit.py` we wrap every external dependency (HyDE LLM, sparse encoder, reranker, etc.) with a breaker. Defaults: 3 failures → open for 30 seconds → half-open probe.

When a breaker is open, the pipeline **degrades gracefully**:

- HyDE breaker open → skip HyDE, embed the raw query
- Reranker breaker open → return the RRF-fused list without rerank
- Graph breaker open → skip the graph hop stage
- Dense breaker open → return BM25 only (or fail loudly)

This is what keeps a single flaky model service from taking down the whole retrieval pipeline.

### 2.9 What is a "skill" in Nexus

A **skill** is a plain Markdown file with a YAML frontmatter, describing patterns / pitfalls / conventions for some scope:

```yaml
---
name: my-auth-middleware
kind: product_domain
scope: product
product: my-api
version: 3
confidence: 0.91
applies_to:
  files: ["src/auth/**", "src/middleware/jwt.ts"]
  contexts: ["code-review", "feature-work"]
composes_with: [master]
provenance:
  council_session: cs_abc123
  validated_by: alice@example.com
  validated_at: 2026-05-18T10:00:00Z
  evidence_chunks: [chunk_id_1, chunk_id_2]
  revision_count: 1
---

# Auth Middleware

## Patterns
- All JWT validation must go through `verifyJWT()` in [src/auth/jwt.ts:42].
- Never read `req.headers.authorization` directly — use `requireAuth()` middleware.

## Anti-patterns
- Don't put auth logic in route handlers.
- Don't store user IDs in `req.cookies`; use `req.session`.
```

Key properties:

- **Versioned** — every skill has a version that bumps on edit.
- **Cited** — every claim has a `[file:line]` anchor that points to real code.
- **Composable** — skills declare `composes_with: [...]` and Nexus assembles bundles at serve time. A request that touches `src/auth/jwt.ts` would receive `[master + my-auth-middleware]`.
- **Confidence-rated** — a float in [0,1] computed at draft time based on citation count, retrieval scores, and adversary critique severity. The UI shows it as a colored bar.
- **Provenance-tracked** — every skill records which council session drafted it, who validated, when, with what evidence.

Skills are stored as files in a git repo (`skills_repo` in `nexus.yaml`) for full history. They're served via MCP as resources (`nexus://skills/{product}/{kind}/{name}`).

> **Naming note:** Anthropic also has a feature called "Skills" for Claude Code. Nexus skills predate that and serve a different purpose — they're *grounded guidance for any agent*, not an Anthropic-specific construct.

---

## 3. The mental model — three invariants

If you remember nothing else, remember these:

### Invariant 1: Product = root entity

Every resource scopes to one `product_id`. Sources, ingested chunks, council sessions, skill files, activity events — all carry `product_id`. The shard key in Qdrant is `product_id`. The Neo4j filter is on `product_id`. Crossing the boundary is a tenancy bug.

**Why:** a Nexus instance might serve many products at one org. The Auth team should never see the Billing team's skills, and vice versa. Multi-product isolation is non-negotiable.

### Invariant 2: Skills compose; they don't override

When an AI client asks "give me the skills for this file," Nexus assembles a bundle by composition:

```
[master skill (the product itself)]
+ [matching product-domain skills (auth middleware, billing math, …)]
+ [matching adopted org standards (TS conventions, OWASP, …)]
```

`composes_with` in the frontmatter defines the explicit prerequisite chain. There is no override — skills always add context, never silence each other.

### Invariant 3: Humans approve; agents draft

The Council writes **proposals** into a SQLite queue (`./data/proposals.db`). Proposals are not skills. A proposal becomes a skill only when a human clicks **Approve** in the UI (or calls `POST /proposals/{id}/approve`). The approval handler writes the file, commits to git, and indexes the skill body so it's searchable.

**Why:** trust. Auto-merge would let one bad council session pollute the org's curated knowledge. The approval workflow turns Nexus into a *human-in-the-loop* system, which is what makes the outputs trustworthy.

---

## 4. How the pieces talk — bytes on the wire

A bird's-eye view of every communication channel:

```
┌────────────────────┐
│  Browser (UI)      │
│  nexus-ui :3000    │
└──┬──────────────┬──┘
   │ HTTP/JSON    │ SSE
   │              │
┌──▼──────────────▼──┐         ┌────────────────┐         ┌────────────────┐
│  FastAPI :8000     │◄────────┤  ProposalQueue │         │   Qdrant       │
│  nexus/api/        │         │  SQLite        │         │   :6333        │
└──┬──────────────┬──┘         └────────────────┘         └───▲────────────┘
   │              │                                            │
   │ stdio        │ async tasks                                │
   │ (subprocess) │                                            │
   │              ▼                                            │
   │       ┌────────────────┐         ┌────────────────┐       │
   │       │ Daemon         ├────────►│ Ingest Pipeline├───────┤
   │       │ daemon.py      │         │ chunker/embed  │       │
   │       └──▲─────────────┘         └────────────────┘       │
   │          │ MCP                                            │
   │          │ stdio                                          │
   │      ┌───┴──────────┐                                     │
   │      │ Connector    │                                     │
   │      │ MCP servers  │                                     │
   │      │ (separate)   │                                     │
   │      └──────────────┘                                     │
   │                                                           │
   │  MCP stdio                                                │
   ▼                                                           │
┌────────────────────┐                                         │
│ Claude Desktop /   │                                         │
│ Cursor / Continue  │                                         │
│ (MCP client)       │                                         │
└─────────┬──────────┘                                         │
          │                                                    │
          │ nexus-mcp-server is launched as a subprocess       │
          │                                                    │
┌─────────▼──────────┐                                         │
│ nexus/mcp_server/  ├─────────────────────────────────────────┘
│ server.py          │  (queries Qdrant for retrieval)
└────────────────────┘
```

Three independent processes commonly run side by side: the FastAPI app, the daemon, and the MCP server (one per AI client). They communicate via:

- **FastAPI → SQLite** (registry, proposal queue, council checkpoints).
- **FastAPI ↔ Qdrant / Neo4j** (read and write).
- **Daemon ← MCP** (connector servers push notifications to the daemon over stdio).
- **MCP server → Qdrant / Neo4j** (read-only at query time).
- **Browser ↔ FastAPI** (HTTP for everything; SSE for streaming).

No message bus, no Kafka, no Redis pub-sub. Everything is direct.

---

## 5. Backend code map (`nexus/`)

We covered the file tree in the prior version of this doc; here we walk through it with explanations woven in.

### `api/` — FastAPI HTTP surface

```
api/
├── app.py        FastAPI() instance; mounts routers; CORS config
├── deps.py       shared Depends() helpers
└── routes/       one file per resource
```

**FastAPI**, briefly: an async web framework where you write `async def` handlers and declare types via Pydantic models. It auto-generates an OpenAPI spec from your type signatures.

**`Depends()`** is FastAPI's dependency-injection mechanism:

```python
@router.get("/sources")
async def list_sources(
    config: NexusConfig = Depends(get_config_dep),
    registry: Registry = Depends(get_registry),
):
    ...
```

Each handler declares what it needs. `get_config_dep` and `get_registry` are factories defined in `deps.py` that yield the singletons. This makes testing trivial — in tests you override the dependency to a fake.

### `ingest/` — turning raw resources into searchable vectors

```
ingest/
├── models.py            ResourceRef, Chunk, EmbeddedChunk
├── chunker.py           tree-sitter for code; heading split for docs
├── enricher.py          optional context summary prepended to chunks
├── embedder.py          calls llama-server /v1/embeddings (Jina v4)
├── relation_extractor.py extracts (s, p, o) triples from doc chunks via light LLM
├── indexer.py           writes chunks to Qdrant with named vectors
├── incremental.py       single-resource path used by the daemon
└── pipeline.py          orchestrator
```

#### Tree-sitter

[**Tree-sitter**](https://tree-sitter.github.io/) is an incremental parser library with grammars for every major programming language. We use it in `chunker.py` to split source code along *semantic boundaries* (functions, classes, methods) instead of naive line-count chunks. A function definition stays in one chunk. The result: every chunk carries a precise `file:line` anchor and represents a coherent unit of code.

Supported languages out of the box: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.rs`, `.go`. Markdown is split on heading hierarchy. Anything else falls back to fixed-size character splits.

#### Why `EmbeddedChunk` is its own model

A `Chunk` is content + metadata. An `EmbeddedChunk` is a chunk + its dense + sparse vectors. Separating them lets us re-embed at different model versions without re-chunking.

### `retrieval/` — the 5-stage pipeline

Covered in detail in §2.3. The orchestrator is `pipeline.py`:

```python
async def retrieve(query, product_id) -> list[Hit]:
    if cache_hit := await self.cache.lookup(query, product_id):
        return cache_hit
    is_complex = self.classifier.classify(query)
    hyde_doc = await self.hyde.expand(query) if is_complex else query
    dense = await self.qdrant.search(embed(hyde_doc), shard_key=product_id)
    sparse = await self.qdrant.search_sparse(sparse_embed(query), shard_key=product_id)
    graph = await self.graph.expand(dense[:5], product_id)
    fused = rrf([dense, sparse, graph], k=60)
    top_k = await self.reranker.rerank(query, fused[:50])
    high_quality = [h for h in top_k if h.score > 0.3]
    await self.cache.store(query, product_id, high_quality)
    return high_quality
```

Every component has a circuit breaker around it. If `reranker` is open, we skip rerank and degrade.

### `council/` — multi-agent LLM orchestration

```
council/
├── state.py        CouncilState — TypedDict flowing through the graph
├── graph.py        StateGraph wiring
├── runner.py       _SessionHub — kick_off, _run_session, stream_events
├── queue.py        SQLite-backed proposal + session queue
├── change_request.py  org-skill change-request routing
└── agents/         one async function per agent
```

#### `_SessionHub`

The heart of the council. It manages:

- A dict of `asyncio.Queue` per active session.
- Each LangGraph node calls `_publish_node_delta(session_id, node_name, delta)` which enqueues a state-update event.
- The `stream_events(session_id)` async generator dequeues and yields them as SSE events.

This is why the UI sees deliberation messages stream in real-time: the LangGraph nodes are producing them while running, and the SSE consumer is reading them off a queue in the same process.

#### Agents

Each agent is a single async function:

```python
# nexus/council/agents/archaeologist.py (sketch)
async def run(state: CouncilState, handles: CouncilHandles) -> dict:
    hits = await handles.retrieval.retrieve(state["topic"], state["product_id"])
    response = await handles.chat_arch.chat(prompt_template(state, hits))
    return {"archaeologist_findings": response.text, "citations": [...]}
```

Returned dicts are merged into `CouncilState` by LangGraph. The next node reads the merged state.

Agents are intentionally **small** — most are under 150 lines. The hard work is in prompts (in `nexus/council/agents/_common.py`) and in retrieval (called via `handles.retrieval`).

### `connectors/` — the MCP client side

```
connectors/
├── manager.py     ConnectorManager — supervises N clients
├── mcp_client.py  async MCP stdio client wrapper
└── local_fs.py    a built-in connector that doesn't go over MCP
```

`ConnectorManager.updates()` is the async iterator the daemon loops over. Each `ManagedConnector` runs an `mcp_client` in a supervisor loop that reconnects with exponential backoff (1s → 30s) on transport failure.

### `graph/` — Neo4j wrapper

```
graph/
└── store.py    GraphStore — upsert_triples, cypher_query, scope_by_product
```

Thin wrapper over the official Neo4j async driver. Adds product_id filtering and a relation-aware upsert.

### `mcp_server/` — what Claude Desktop connects to

```
mcp_server/
├── server.py   MCP stdio server entry; argparse, registers tools/resources
└── tools.py    find_skills, query_code_context, hybrid_search_corpus
```

When you add Nexus to Claude Desktop's config, Claude spawns `nexus-mcp-server` as a subprocess and exchanges JSON-RPC messages over the subprocess's stdin/stdout. Our server registers handlers for `tools/list`, `tools/call`, `resources/list`, `resources/read`, and forwards each call to the corresponding function in `tools.py`.

### `tasks/` — task runners (the "Apply" verb)

```
tasks/
├── pr_review.py    webhook handler → retrieve skills → post a structured PR comment
└── changelog.py    tag-push handler → generate release notes
```

These are the consumers of the curated skills. When a PR opens, `pr_review.py` retrieves the most relevant skills for the diff, sends them to an LLM along with the diff, and posts a comment that cites each skill chip (`[skill: my-auth-middleware]`).

### `assistant/` — the conversational + action layer (Slice 8)

```
assistant/
├── models.py         ActionProposal / Conversation / message models
├── store.py          SQLite — conversations, messages, action proposals, identities
├── capabilities.py   the curated ~8-tool facade the agent LLM sees
├── loop.py           the tool-calling agent loop (the "brain")
├── connector_port.py ReadPort / ActPort Protocols + FakeConnectorPort stub
├── atlassian.py      AtlassianConnectorPort — real Jira/Confluence via Rovo MCP
├── executor.py       runs a confirmed ActionProposal — the only write path
└── factory.py        build_assistant_store / build_connector_port (shared wiring)
```

The Assistant lets a user (or another agent) query Jira/Confluence and *draft*
human-confirmed changes. The cardinal rule — **curate, don't proxy**: the agent
LLM only ever sees the ~8 intent-shaped tools in `capabilities.py`; the 30-40 raw
Atlassian tools stay behind `atlassian.py`. Full design: [`docs/ASSISTANT-LAYER.md`](./docs/ASSISTANT-LAYER.md).

### `auth/` — per-user OAuth

```
auth/
├── atlassian_oauth.py  OAuth 2.1 + PKCE — authorize URL, code exchange, refresh
└── token_cipher.py     Fernet encryption for OAuth tokens at rest
```

Assistant write actions run as the *real user* (ADR-013), so each user connects
their own Atlassian account; tokens are encrypted with a key from `NEXUS_TOKEN_KEY`.

### Webhooks, briefly

A **webhook** is an HTTP POST that an external service (GitHub here) sends to your URL when something happens (PR opened, tag pushed). You publish the URL; they call it.

GitHub signs every webhook with an HMAC of the payload using a shared secret. We verify the signature in `nexus/api/routes/webhooks.py` before processing — otherwise anyone could forge events. If you've never done webhook HMAC verification:

```python
import hmac, hashlib
expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
if not hmac.compare_digest(expected, received_signature):
    raise HTTPException(401)
```

### `cli.py`, `daemon.py`, `config.py`, `registry.py`

| File | Role |
|---|---|
| `cli.py` | Typer-based CLI. `nexus ingest`, `nexus query`, `nexus council draft`. |
| `daemon.py` | The long-running ingest daemon (§2.2). |
| `config.py` | Loads `nexus.yaml`, expands `${ENV_VARS}`, validates via Pydantic. |
| `registry.py` | SQLite store for products, users, runtime-added sources. |

`config.py` is worth reading in full — Pydantic's `BaseSettings` makes it elegant. The whole config schema is type-checked at boot.

### `observability/otel.py`

Sets up OpenTelemetry tracing and Langfuse trace IDs. Every LLM call gets a span; spans propagate through async context.

### `llm/client.py`

`ChatClient` — a thin wrapper over `httpx` that speaks the OpenAI chat completions shape. Used for all council agents. Tracks tokens + cost per call.

---

## 6. Frontend code map (`nexus-ui/`)

### Next.js App Router in 30 seconds

If you've used Next.js' older `pages/` router, the App Router is different:

- Routes are folders inside `app/`. A folder with a `page.tsx` is a route.
- `layout.tsx` wraps all routes below it.
- Folders like `[product]` are **dynamic segments** — they match anything, and the value is passed as a prop.
- Components are **server components by default**. Add `'use client'` at the top of the file to opt into client-side rendering (needed for hooks, effects, event handlers).
- `loading.tsx` next to `page.tsx` renders during route data fetching.

### Directory structure

```
nexus-ui/
├── app/
│   ├── layout.tsx              wraps everything in Shell + providers
│   ├── page.tsx                "/" — redirects based on whether any products exist
│   ├── globals.css             Tailwind @theme block — all design tokens
│   ├── onboarding/             /onboarding
│   ├── settings/org/           /settings/org
│   ├── p/[product]/            all product-scoped routes
│   │   ├── dashboard/
│   │   ├── sources/{,/new,/[name]}
│   │   ├── council/{,/[sessionId]}
│   │   ├── skills/{,/[id]}
│   │   ├── assistant/          Assistant chat panel (Slice 8)
│   │   ├── activity/
│   │   ├── settings/
│   │   └── proposals/
│   └── {dashboard,skills,connectors}/  legacy redirects
│
├── components/
│   ├── screens/   one .tsx per UX screen — owns data fetching
│   ├── shell/     TopBar, SideNav, ProductSwitcher, CommandPalette
│   ├── ui/        shadcn-style primitives — Card, Button, Badge, …
│   ├── skeletons/ loading states
│   └── icons/     BrandIcon wrapper for simple-icons
│
└── lib/
    ├── api/{client,index}.ts    typed HTTP client, 1:1 with FastAPI routes
    ├── hooks/useEventStream.ts  SSE consumer
    ├── types.ts                 shared types (aligned to backend)
    ├── product-context.ts       React context for current product + user + perms
    └── utils.ts                 cn() — class-merge helper
```

### Why no Redux / Zustand / React Query?

The data model is simple: each screen fetches what it needs on mount, refetches after mutations. No global state beyond the product context. Caching pain hasn't appeared yet. If it does, we add React Query — but deliberately, not by default.

### The "current product" context

`lib/product-context.ts` exposes `useProduct()` returning the current product, the current user, RBAC permissions, and a `debugRole` for the demo persona switcher. Every screen reads `currentProductId` and uses it in API calls.

The shell (`components/shell/Shell.tsx`) populates this context from `GET /me` + `GET /products` on app boot. Until those resolve, `loading: true` is set, and screens defensively render placeholder UIs.

### The SSE hook

`lib/hooks/useEventStream.ts` is a single-purpose SSE consumer. Give it a URL, get back `{events, status, error}`. It handles auto-reconnect, named events, and buffer caps. Used by `CouncilSession` (deliberation stream) and `ConnectorDetail` (sync log).

### Design system primitives

Read [`nexus-ui/DESIGN.md`](../nexus-ui/DESIGN.md) for the full rules. Quick rules:

- Use `<H1>`, `<H2>`, `<Body>`, `<Subtle>`, `<Code>` from `components/ui/typography.tsx`. Never raw `<h1>` or `text-lg`.
- Use `<PageHeader>`, `<PageBody>`, `<PageGrid>` from `components/ui/page.tsx` for page chrome.
- All design tokens (`bg-bg`, `text-fg-muted`, `border-border`, etc.) live in `app/globals.css` inside the `@theme inline` block.
- Brand icons (GitHub, Jira, Confluence) via `BrandIcon`. Everything else via lucide-react.

---

## 7. End-to-end traces

These walk through real code paths. Open the referenced files in a second pane and *follow along* — that's how you really learn the code.

### Trace A: User approves a proposal in the UI

```
[Browser]  Proposals.tsx → click Approve
   ↓
[lib/api]  approveProposal(id, actor)
           POST /proposals/{id}/approve  body={"actor":"alice@example.com"}
   ↓
[FastAPI]  nexus/api/routes/proposals.py::approve
           - reads ProposalQueue row
           - calls approve_proposal(...)
   ↓
[approval] nexus/skills/approval.py::approve_proposal
           1. Validates the proposal
           2. SkillStore.save(skill, path)            → writes .skill.md
           3. _embed_skill_body(skill)                → uses EmbedderClient
           4. Indexer.index_chunks([embedded_chunk])  → writes to Qdrant
                with shard_key=product_id (§2.7)
           5. ProposalQueue.mark_approved(id, actor)
           6. git add + commit + push                 → nexus/skills/git.py
   ↓
[Response] {"ok": true, "skill_id": "...", "chunks_indexed": 12}
   ↓
[Browser]  refresh() re-fetches proposals list — approved row disappears
```

**Things to notice:**

- The skill becomes searchable immediately (step 4 indexes its body before the response returns).
- Git is the *source of truth*. The filesystem write is a side effect; the canonical record is the git commit.
- Failure modes: if step 4 succeeds but step 6 fails (network), the skill is searchable but not committed. Next boot, the system will re-sync from git, so the in-memory copy is reconciled. Worst case: a skill is indexed but lost on restart (logged loudly).

### Trace B: User starts a council session and watches it live

```
[Browser]  CouncilLanding → click "Start session"
           POST /products/{id}/council/sessions  body={"topic":"...","skill_kind":"..."}
   ↓
[FastAPI]  routes/council.py::create_session
           - validates input
           - generates session_id
           - calls council.runner.kick_off(...)
           - kick_off:
               1. _SessionHub.register(session_id) — creates asyncio.Queue
               2. asyncio.create_task(_run_session(...)) — schedules in background
               3. returns immediately
           - returns {"session_id":"...","status":"running"}
   ↓
[Browser]  router.push(/p/{id}/council/{session_id})
           CouncilSession.tsx mounts
           const { events } = useEventStream(sessionStreamUrl(session_id))
   ↓
[Browser]  opens EventSource at GET /council/sessions/{sid}/stream
   ↓
[FastAPI]  routes/council.py::stream_session
           - returns StreamingResponse from council.runner.stream_events(sid)
           - stream_events: async generator dequeuing from _SessionHub's queue
   ↓
[LangGraph]   council/graph.py compiled StateGraph executes
              ├─ archaeologist node runs:
              │     ├─ handles.retrieval.retrieve(topic) → hits
              │     ├─ handles.chat_arch.chat(prompt)    → LLM response
              │     └─ returns {archaeologist_findings: ..., citations: [...]}
              │     ↓ LangGraph state merged ↓
              │   _publish_node_delta(sid, "archaeologist", delta)
              │     → enqueues SSE event in _SessionHub queue
              │     → reaches the browser via the open SSE connection
              ├─ domain_expert runs in parallel (same pattern)
              ├─ synthesizer runs after both, produces draft body
              ├─ adversary critiques; if blocking severity, loops back
              │   to synthesizer for one revision
              └─ END — proposal finalized
   ↓
[Queue]    ProposalQueue.enqueue(proposal_id, session_id, status="pending")
   ↓
[SSE]      event: session_end / data: {"proposal_id": "..."}
   ↓
[Browser]  on session_end: router.push to the proposals page
```

**Things to notice:**

- The session runs entirely in the FastAPI process; we don't need a separate worker.
- The `_SessionHub` queue is the critical piece — it decouples the LangGraph executor from the SSE consumer. They run independently; the queue is the bridge.
- LangGraph's SqliteSaver checkpoints state at every node transition. If the FastAPI process crashes, the next boot can resume the session from the last checkpoint.

### Trace C: Daemon detects a file change and re-indexes it

```
[GitHub] developer pushes commit
   ↓
[mcp-github-server] separate MCP server (not this repo) emits a "resource updated" notification
   ↓
[manager]  nexus/connectors/manager.py::ConnectorManager.updates() yields the event
           (the manager is supervising mcp_client over stdio with reconnect-on-failure)
   ↓
[daemon]   nexus/daemon.py watch loop receives the update
           calls reindex_resource(update, product_id, …)
   ↓
[ingest]   nexus/ingest/incremental.py::reindex_resource
           1. delete the old chunks for this resource from Qdrant (by resource_id payload filter)
           2. chunker.chunk_resource() — tree-sitter aware
           3. (if docs) enricher prepends context summary
           4. embedder.embed_batch(chunks) — calls llama-server /v1/embeddings
           5. (if docs) relation_extractor extracts triples
           6. indexer.upsert(embedded_chunks) — Qdrant write with shard_key=product_id
           7. (if triples) graph_store.upsert_triples(triples, product_id) — Neo4j write
           8. semantic_cache.invalidate_for_chunks(...) — purge affected cache entries
   ↓
[Logging] daemon logs duration + chunk count; OTel span closes
```

**Things to notice:**

- Steps 6 + 7 are eventually consistent — Qdrant might be updated 200ms before Neo4j. A query running between them gets vector hits but no graph expansion for the new chunks. Not a correctness bug — just a brief degradation.
- Cache invalidation (step 8) is critical. Without it, the next query might return a cached result that's missing the new chunk.

### Trace D: Claude Desktop calls `find_skills`

```
[Claude Desktop] user runs a task. Claude needs guidance for the codebase.
                 sends MCP request:
                 method=tools/call, name=find_skills, args={query: "JWT validation pattern"}
   ↓
[stdio]  the nexus-mcp-server subprocess receives JSON-RPC over stdin
   ↓
[mcp_server] nexus/mcp_server/server.py routes the call to tools.find_skills
   ↓
[tools]      nexus/mcp_server/tools.py::find_skills
             1. RetrievalContext.retrieve(query, product_id) — the full 5-stage pipeline (§2.3)
             2. group hits by skill_id (some chunks come from skill bodies, others from code)
             3. for each candidate skill: load it from SkillStore
             4. assemble composition: master + matching domain + matching org standards
             5. return [{name, body, confidence, citations, composed_with: [...]}]
   ↓
[stdio]  JSON response written to stdout
   ↓
[Claude] Claude reads the response, uses skill bodies as context for the task
```

**Things to notice:**

- This is the *whole point of the system*. Everything else (ingest, council, approval) feeds this moment.
- The MCP server has no HTTP exposure — it lives in stdio, launched by the client. That's deliberate: it inherits the client's auth, the connection is the lifetime of the subprocess, there's nothing to firewall.
- Retrieval here is the *same pipeline* used by the council during drafting. Skills are searched the same way code is.

---

## 8. Local development workflow

### Setup checklist

```bash
# 1. Backend
cd nexus/
uv sync                                # install Python deps
cp nexus.yaml.example nexus.yaml
cp .env.example .env
# edit .env — at minimum: DEEPINFRA_API_KEY, GITHUB_TOKEN (or leave blank)

# 2. Frontend
cd ../nexus-ui/
npm install

# 3. Infrastructure
cd ../nexus/
docker compose up -d                   # Qdrant + Neo4j + Langfuse + Postgres

# 4. Local model servers (Apple Silicon path)
make services-up                       # llama-server embed + rerank + Ollama

# 5. Run them
# terminal 1
uv run uvicorn nexus.api.app:app --port 8000 --reload
# terminal 2
cd ../nexus-ui/
npm run dev
```

Visit `http://localhost:3000`. First load redirects to `/onboarding`.

### Hot reload

- Backend: `--reload` flag on uvicorn picks up `.py` edits.
- Frontend: Next.js HMR for `.tsx` and `.css` edits.
- `nexus.yaml`: not hot-reloaded; restart uvicorn after edits.

### Useful commands

```bash
# Lint
uv run ruff check nexus tests
uv run ruff format nexus tests

# Type check (frontend)
cd ../nexus-ui && npx tsc --noEmit

# Test
uv run pytest                          # all
uv run pytest -x -q                    # fail fast, quiet
uv run pytest tests/test_chunker.py    # one file
uv run pytest -k approve               # by keyword
uv run pytest -k approve --pdb         # drop into debugger on failure

# Run the daemon
uv run python -m nexus.daemon --product my-api

# Run the CLI
uv run nexus --help

# Run the MCP server (what Claude Desktop launches)
uv run nexus-mcp-server --product my-api
```

### What you need running for each task

| Task | uvicorn | docker | make services-up | DeepInfra key |
|---|---|---|---|---|
| Backend unit tests | ❌ | ❌ | ❌ | ❌ |
| Frontend dev (no real data) | ✅ | ❌ | ❌ | ❌ |
| End-to-end: list products, navigate | ✅ | ❌ | ❌ | ❌ |
| Real ingest (chunks land in Qdrant) | ✅ | ✅ | ✅ | ❌ |
| Real council session | ✅ | ✅ | ✅ | ✅ |
| Real PR review | ✅ + daemon | ✅ | ✅ | ✅ |

The good news: you can do most frontend and backend development without ever running the heavy infra. We mock retrieval/LLM calls in tests.

### Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: nexus.yaml` on boot | no config file | `cp nexus.yaml.example nexus.yaml` |
| All API calls 500 | uvicorn or one of its deps not running | check terminal for stack trace |
| `Backend unreachable` on every UI screen | uvicorn not on `:8000` | start uvicorn |
| CORS error in browser console | next.js port not in CORS allowlist | edit `nexus/api/app.py` allow_origins |
| Council session hangs at "drafting…" | DeepInfra key missing/invalid | check `.env` and uvicorn logs |
| Ingest hangs on embed step | llama-server not running | `make services-up`, then `curl :8080/health` |
| Skill approval succeeds but skill not searchable | `hierarchy_root` path doesn't exist | `mkdir -p ./skills` or change config |
| Qdrant raises 404 on collection | first boot didn't create the collection | restart uvicorn — collections are created lazily on first ingest |

### Debugging tips

1. **Langfuse trace UI** — `http://localhost:3001`. Every LLM call is logged with full prompts, completions, latencies, and costs. Drop into the UI to see what the council saw.
2. **CLI single-query introspection** — `uv run nexus query "your text" -p my-api` prints retrieval results stage-by-stage with scores. Fastest way to debug retrieval quality.
3. **Set `LOG_LEVEL=DEBUG`** before running uvicorn for verbose logs.
4. **`pytest --pdb`** — drops into the interactive debugger at the first failure.
5. **Qdrant dashboard** — `http://localhost:6333/dashboard`. Inspect collections, points, payloads. Great for "is my chunk actually in there?" debugging.
6. **Neo4j browser** — `http://localhost:7474`. Run Cypher queries against the graph layer.

---

## 9. Hands-on tour — exercises to learn by tinkering

These are designed to take you ~15–30 minutes each. Do them in order — they build on each other.

### Exercise 1: Read your way through one trace

Open the file at `nexus/skills/approval.py:38` (the `approve_proposal` function). Read it line by line. Now open `nexus/skills/store.py` and find `SkillStore.save`. Trace what it does.

**Goal:** appreciate that the whole approve-a-proposal flow is ~100 lines across 2 files. Most of the codebase is like this.

### Exercise 2: Run the test suite and break a test on purpose

```bash
uv run pytest -x -q                    # should be all green
```

Now break one test:

```bash
# Edit nexus/ingest/chunker.py and change a constant — say, MAX_CHUNK_SIZE.
# Re-run:
uv run pytest -x
# See the failure. Revert.
```

**Goal:** confirm the test loop works. Notice how fast tests are (<3s for the whole suite).

### Exercise 3: Add a no-op API endpoint

In `nexus/api/routes/dashboard.py`, add a new endpoint:

```python
@router.get("/{product_id}/dashboard/hello")
async def hello(product_id: str) -> dict:
    return {"product": product_id, "message": "hi"}
```

Restart uvicorn. Hit it:

```bash
curl http://localhost:8000/products/my-api/dashboard/hello
```

Now expose it from the frontend. In `nexus-ui/lib/api/index.ts`:

```ts
export const getHello = (productId: string) =>
  api.get<{product: string; message: string}>(`/products/${productId}/dashboard/hello`)
```

Call it from any component, log the result. **Goal:** internalize the round trip.

### Exercise 4: Trace a council session in Langfuse

Spin up Langfuse (already in `docker compose up`). Run a council session via the CLI:

```bash
uv run nexus council draft --product my-api --topic "logging conventions" --kind product_domain
```

Open `http://localhost:3001`. You'll see a trace tree — every LLM call with its prompt, completion, tokens, latency. Click into one. Notice how the prompts include retrieval results.

**Goal:** understand that the council is *just* a sequence of LLM calls — there's no magic.

### Exercise 5: Watch the SSE stream raw

```bash
curl -N http://localhost:8000/council/sessions/<some-session-id>/stream
```

Watch raw SSE events scroll past:

```
event: session_start
data: {"session_id":"...","skill_kind":"..."}

event: message
data: {"agent":"archaeologist","body":"..."}

event: cost
data: {"agent":"archaeologist","prompt_tokens":1240,...}
```

**Goal:** see the wire-level events the UI consumes.

### Exercise 6: Inspect a chunk in Qdrant

After running an ingest:

```bash
curl 'http://localhost:6333/collections/nexus_code/points/scroll' \
  -H 'Content-Type: application/json' \
  -d '{"limit": 3, "with_payload": true, "with_vector": false}'
```

You'll see the actual chunk text, file:line metadata, product_id. **Goal:** demystify what's stored.

### Exercise 7: Make your first real PR

Find a `TODO` in the code, fix it. Or pick something from the list of small contribution ideas in [`docs/SLICE-7-STATUS.md`](./docs/SLICE-7-STATUS.md) ("What's deferred"). Open a PR. Tag it with the slice it touches.

---

## 10. Recipes — common contribution tasks

### Add a new API endpoint

1. Pick the right route file under `nexus/api/routes/`, or create a new one.
2. Write the handler — use `Depends()` for shared state.
3. Register the router in `nexus/api/app.py` if it's new.
4. Add a typed client in `nexus-ui/lib/api/index.ts`.
5. Add response types in `nexus-ui/lib/types.ts` if they don't exist.
6. Use from a screen.

```python
# nexus/api/routes/foo.py
from fastapi import APIRouter, Depends
from nexus.api.deps import get_registry
from nexus.registry import Registry

router = APIRouter(prefix="/foo", tags=["foo"])

@router.get("/{item_id}")
async def get_foo(item_id: str, registry: Registry = Depends(get_registry)) -> dict:
    return {"id": item_id}
```

```python
# nexus/api/app.py
from nexus.api.routes import foo
app.include_router(foo.router)
```

```typescript
// nexus-ui/lib/api/index.ts
export const getFoo = (id: string) => api.get<{id: string}>(`/foo/${id}`)
```

### Add a new UI screen

1. Build `nexus-ui/components/screens/MyScreen.tsx`. Follow the data-fetching pattern in §6.
2. Add the route file `nexus-ui/app/p/[product]/my-screen/page.tsx`:

   ```tsx
   import { MyScreen } from '@/components/screens/MyScreen'
   export default function Page() { return <MyScreen /> }
   ```

3. Add `loading.tsx` next to it with a skeleton.
4. Add a SideNav entry in `components/shell/SideNav.tsx`.
5. Optionally add a CommandPalette entry so `⌘K` can jump to it.

### Add a new council agent

1. New file under `nexus/council/agents/`. Pattern: a single async function `run(state, handles) → dict`.
2. Wire it into the StateGraph in `nexus/council/graph.py::build_graph`.
3. If it needs a dedicated model tier, add the role to `ModelsCfg` in `nexus/config.py` and `nexus.yaml.example`.
4. Update the roster constants in `nexus-ui/lib/types.ts::COUNCIL_ROSTERS`, `COUNCIL_AGENT_LABELS`, `COUNCIL_AGENT_HUES` if the agent appears in the UI.

### Add a new MCP connector

You don't add the connector *here* — you implement it as a separate MCP server in its own repo. Once it speaks the MCP protocol, you reference it in `nexus.yaml`:

```yaml
connectors:
  - name: my-custom
    type: custom-stdio
    command: /path/to/my-mcp-server
    args: ["--config", "/path/to/config.json"]
    watch: true
```

`ConnectorManager` spawns it via stdio on boot.

### Add a new skill kind

1. Add to the enum in `nexus/skills/models.py::SkillKind` (product-scope) or `OrgSkillKind` (org-scope).
2. Mirror in `nexus-ui/lib/types.ts`.
3. Define which agents draft it in `COUNCIL_ROSTERS`.
4. Pick a color in `KIND_COLOR` in the Skills screen.

### Add a retrieval pipeline stage

1. Write a leaf module (e.g. `retrieval/foo.py`) with `async def run(...) → list[Hit]`.
2. Wire it into `RetrievalContext.retrieve()` in `nexus/retrieval/pipeline.py`.
3. Add a circuit breaker entry in `retrieval/circuit.py` if it calls a network service.
4. Re-run `evals/run_code_eval.py` to confirm nDCG@10 didn't regress.

### Debug retrieval quality

1. Run `uv run python -m evals.run_ragas --verbose` — see which golden-set items fail.
2. Drop into CLI: `uv run nexus query "your text" -p my-api` shows stage-by-stage scores.
3. Open Langfuse to see the LLM-judge's reasoning for each failed item.

---

## 11. Testing

### Where tests live

`tests/test_<module>.py` — one file per backend module. 104 tests at time of writing, ~3s to run.

### Testing patterns

| Kind | Pattern | Example |
|---|---|---|
| Unit | direct function call with fake input | `tests/test_chunker.py` |
| Integration (in-process) | FastAPI `TestClient`, real DB on tmpdir | `tests/test_approval.py` |
| Eval (offline) | golden set + LLM judge with thresholds | `evals/run_ragas.py` |

### When to add a test

- **Always** — new public function on a leaf module.
- **Always** — new API route. Happy path + at least one error case.
- **Sometimes** — new UI logic that's nontrivial (e.g. the SSE buffer in CouncilSession).
- **Never** — UI styling. Manual review.

### CI

`.github/workflows/ci.yml`:

1. `ruff check`
2. `pytest`
3. `evals.run_ragas` against the golden set
4. Build fails if faithfulness drops > 5% from `evals/baseline.json`.

---

## 12. Code conventions

### Python

- `from __future__ import annotations` at the top of every module.
- Type-annotate all public functions.
- **Pydantic** for data crossing process boundaries (API requests, DB rows). **Dataclasses** for in-memory only. Plain dicts only in tests.
- Async by default for anything that touches the network or significant filesystem work.
- Import order: stdlib, third-party, `nexus.*`. ruff/isort enforces.
- Single-line comments. Multi-paragraph → docstring at the top of the function.
- No `print()` — use `log = logging.getLogger(__name__)`.

### TypeScript / React

- `'use client'` only when needed. Default to server components.
- Avoid `any` — prefer `unknown` + a type guard.
- No inline `style={{}}` except dynamic per-instance colors.
- One screen per file. Co-locate small helpers; promote to `lib/` only if reused.
- Don't add runtime deps without discussion — `package.json` is intentionally small.

### Commits and PRs

- One logical change per commit.
- Imperative subject: "Add foo endpoint" not "Added foo endpoint".
- Reference the slice doc it touches: `[slice-4] Add /proposals/{id}/edit endpoint`.

---

## 13. Glossary — quick reference

| Term | Definition |
|---|---|
| **Action proposal** | A drafted Jira/Confluence change from the Assistant. Inert until a human confirms it — a sibling of `SkillProposal`. Lives in `nexus/assistant/`. |
| **Agent (council)** | A single LLM-driven node in the LangGraph council. Examples: Archaeologist, Synthesizer, Adversary. |
| **ANN** | Approximate Nearest Neighbor — algorithmic family used by vector DBs to find similar vectors in sublinear time. |
| **Assistant** | The conversational + action layer (Slice 8) — queries Jira/Confluence + corpus and drafts human-confirmed actions. Reachable via MCP tools and the UI chat panel. |
| **Bi-encoder** | An embedding model that maps query and doc into separate vectors, compared with cosine. Fast, less accurate. |
| **BM25** | A sparse retrieval algorithm using term-frequency × inverse-document-frequency. Catches exact lexical matches. |
| **Bootstrap (daemon)** | The first-run phase of the daemon — a full one-time ingest before entering the watch loop. |
| **Chunk** | A unit of text from a resource that gets independently embedded. ~200–1000 tokens. |
| **Circuit breaker** | A resilience pattern: skip a flaky dependency instead of retrying when it's failing. States: CLOSED, OPEN, HALF-OPEN. |
| **composes_with** | A skill frontmatter field declaring which other skills this skill depends on at serve time. |
| **Council** | The multi-agent LLM system that drafts skill proposals. Lives in `nexus/council/`. |
| **Cross-encoder** | A model that processes query + doc *together* in one transformer pass for a single relevance score. Slow, accurate. Used for reranking. |
| **Curate, don't proxy** | The rule that the Assistant's agent LLM sees only a small hand-curated tool facade; raw downstream MCP catalogues stay behind the connector boundary (ADR-012). |
| **Daemon** | A long-running background process. In Nexus, the ingest daemon that watches connectors. |
| **Dense vector** | The output of a neural embedding model. 768-d or 1024-d floats, capturing semantic similarity. |
| **Depends() (FastAPI)** | Dependency-injection mechanism. Each handler declares what it needs; FastAPI provides them. |
| **GraphRAG** | RAG that augments vector retrieval with graph traversal. Lets you answer relational queries. |
| **HMAC** | A keyed-hash message authentication code. Used to verify webhook origin. |
| **HyDE** | Hypothetical Document Embeddings. Generate a fake answer with a small LLM, embed *that* instead of the query. |
| **Indexer** | The Nexus component that writes embedded chunks to Qdrant. |
| **LangGraph** | A library for stateful multi-step LLM workflows. We use it for the council. |
| **MCP** | Model Context Protocol — Anthropic's open standard for connecting AI clients to external tools/resources. |
| **MCP client / server** | Roles in MCP. Nexus is both: client to connectors, server to AI clients. |
| **Master skill** | The top-level skill for a product. Describes the product itself. Composed into every bundle. |
| **Org skill** | A skill in the Org Library (tech stack / language / security). Adopted by products via composition. |
| **Pipeline (retrieval)** | The 5-stage flow: classifier → HyDE → hybrid → graph hop → rerank. Orchestrated by `nexus/retrieval/pipeline.py`. |
| **PKCE** | Proof Key for Code Exchange (RFC 7636) — protects an OAuth authorization-code flow. Used by the Assistant's Atlassian OAuth. |
| **Proposal** | A draft skill in the queue, awaiting human approval. Becomes a real skill on approval. |
| **product_id** | The tenant identifier. Everything is scoped by it. |
| **Provenance** | Metadata recording where a skill came from: which session, who validated, when, with what evidence. |
| **Pydantic** | A Python type-validation library. We use it for every cross-boundary data definition. |
| **Reranker** | A cross-encoder model used in the final retrieval stage to refine the top candidates from earlier stages. |
| **RRF** | Reciprocal Rank Fusion. Combines multiple ranked lists by summing `1/(k+rank)` across lists. |
| **Semantic cache** | A query-result cache keyed on the *embedding* of the query, with cosine threshold 0.92. |
| **Server-Sent Events** | A simple one-way streaming format over HTTP. Used for the council deliberation stream and sync log. |
| **Sharding (custom)** | Qdrant feature to physically partition a collection on an arbitrary key. We shard on `product_id`. |
| **Skill** | A versioned, cited, composable Markdown+YAML file describing patterns/conventions for an area of code. |
| **Sparse vector** | A BM25-style TF-IDF vector. Mostly zeros. Stored alongside dense vectors in Qdrant. |
| **State (LangGraph)** | A TypedDict flowing through the graph. Each node returns partial updates that are merged. |
| **StateGraph** | LangGraph's DAG builder. Add nodes, add edges, compile, run. |
| **stdio (transport)** | Standard input/output. MCP's default transport — JSON-RPC over a subprocess's stdin/stdout. |
| **Tree-sitter** | Incremental parser library with grammars for many languages. We use it for semantic chunking. |
| **Webhook** | An HTTP POST your service receives from an external system when an event occurs. |

---

## 14. Further reading

### Official documentation for the tools we use

| Tool | URL |
|---|---|
| MCP | https://modelcontextprotocol.io/ |
| FastAPI | https://fastapi.tiangolo.com/ |
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| Qdrant | https://qdrant.tech/documentation/ |
| Neo4j (Cypher) | https://neo4j.com/docs/cypher-manual/ |
| Tree-sitter | https://tree-sitter.github.io/ |
| Pydantic | https://docs.pydantic.dev/ |
| Next.js App Router | https://nextjs.org/docs/app |
| Tailwind v4 | https://tailwindcss.com/docs |
| Radix UI | https://www.radix-ui.com/ |
| Ollama | https://ollama.com/library |
| Langfuse | https://langfuse.com/docs |

### Papers / posts worth reading

- **HyDE** — "Precise Zero-Shot Dense Retrieval without Relevance Labels", Gao et al., 2022.
- **RRF** — "Reciprocal rank fusion outperforms condorcet and individual rank learning methods", Cormack et al., SIGIR 2009.
- **BM25** — Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
- **GraphRAG** — Microsoft Research's overview: https://www.microsoft.com/en-us/research/blog/graphrag/
- **Circuit breaker pattern** — Martin Fowler's overview: https://martinfowler.com/bliki/CircuitBreaker.html
- **Reciprocal Rank Fusion in practice** — Elastic's writeup: https://www.elastic.co/blog/ranking-search-results-with-rrf

### Internal docs

| Doc | When |
|---|---|
| [`README.md`](./README.md) | Setup + quickstart |
| [`ENGINEERING.md`](./ENGINEERING.md) | Formal architecture spec, ADRs, full API surface |
| [`INTEGRATION.md`](./INTEGRATION.md) | History of the UI↔backend cutover |
| [`docs/UI-CUTOVER-STATUS.md`](./docs/UI-CUTOVER-STATUS.md) | Demo walkthrough — useful as a smoke test |
| [`docs/SLICE-*-STATUS.md`](./docs/) | Per-slice delivery notes |
| [`../nexus-ui/DESIGN.md`](../nexus-ui/DESIGN.md) | UI design system rules |

---

**Welcome aboard.** If anything in this doc is unclear, that's a bug. Open an issue, send a PR, or both.
