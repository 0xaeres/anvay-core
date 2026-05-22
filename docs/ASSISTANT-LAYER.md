# Design — The Nexus Assistant (Action & Conversational Layer)

**Status:** ✅ Implemented as Slice 8 (Increments 1-4). This document is the design
of record; per-increment delivery notes are in [`SLICE-8-STATUS.md`](./SLICE-8-STATUS.md).
The MS Teams adapter (§14, Phase 2) remains future work.
**Author:** drafted from a research pass on enterprise agentic patterns, May 2026.
**Depends on:** Slices 0–7 (ingestion, retrieval, MCP server, council).

---

## 1. Motivation

Nexus today is **read-mostly**: ingest → council → skills → serve. Connectors are used only to pull data *in* for indexing.

This design adds an **Assistant layer** — a conversational, action-taking interface on top of Nexus's existing knowledge. It lets a user (or another agent):

- Ask about a specific Jira story, or search Confluence (optionally scoped to a space).
- Get LLM-assisted analysis — e.g. "break this story into subtasks", "what's missing from this doc".
- **Take actions** with human confirmation — create Jira subtasks, transition issues, add comments, assign, create/update Confluence pages.

And it must be consumable through **multiple channels** without re-implementing the brain each time:

- **MCP tools** — a coding agent or Claude Desktop calls Nexus's assistant capabilities mid-task (Phase 1).
- **MS Teams bot** — a Teams user gets answers and triggers actions in chat (Phase 2).
- **Nexus UI** — a chat panel in the existing web app (Phase 1).

This is well-trodden ground at enterprise scale (Atlassian Rovo, Glean, Microsoft Copilot). The design below adopts the established patterns and — crucially — fits them onto what Nexus already has, rather than bolting on a parallel system.

---

## 2. Goals and non-goals

### Goals

- A single **agent loop** (the "brain") reused by every channel.
- **Read** capabilities: live Jira/Confluence lookup + Nexus's existing GraphRAG corpus.
- **Write** capabilities: broad Jira/Confluence mutations, gated by **human confirmation**.
- **Per-user identity** — writes are attributed to the real user and respect their permissions.
- Reuse Nexus's existing pieces: MCP server, LangGraph, the proposal/approval pattern, SSE, RBAC, the activity timeline.

### Non-goals (for this slice)

- The MS Teams adapter — designed for here, *built* in Phase 2.
- Slack / other channels — Phase 3.
- Autonomous (no-confirmation) writes — never. Invariant 3 (humans approve) holds.
- Replacing the council. The council authors *skills*; the assistant *uses* skills + live data to answer and act.

---

## 3. Architecture — four layers

The dominant industry pattern for multi-channel agents is a four-layer split. The failure mode everyone warns about is logic leaking into channel adapters, so that Teams and Slack drift apart. We avoid it by keeping every channel **thin** and the brain **shared**.

```
┌──────────────────────────────────────────────────────────────────┐
│ CHANNEL LAYER  — thin adapters, no business logic                  │
│   • MCP tools        (coding agents / Claude Desktop)   [Phase 1]   │
│   • Nexus UI chat    (web panel over SSE)               [Phase 1]   │
│   • MS Teams bot     (Adaptive Cards adapter)           [Phase 2]   │
│        every channel calls the SAME internal endpoint               │
└────────────────────────────────┬───────────────────────────────────┘
                                  │  POST /products/{id}/assistant/messages
┌────────────────────────────────▼───────────────────────────────────┐
│ ASSISTANT LAYER  — the brain                                         │
│   • tool-calling agent loop  (LangGraph: LLM node ↔ tool node)       │
│   • conversation memory      (SQLite, like council sessions)         │
│   • action proposal + HITL   (reuses the proposal-queue pattern)     │
└────────────────────────────────┬───────────────────────────────────┘
                                  │  the agent only ever sees curated tools
┌────────────────────────────────▼───────────────────────────────────┐
│ CAPABILITY LAYER  — a curated, intent-shaped tool facade (~10 tools) │
│   READ:  search_corpus · find_skills                                 │
│          get_jira_issue · search_jira · search_confluence · get_page │
│   ACT:   propose_jira_changes · propose_confluence_update            │
│          confirm_action · reject_action                              │
└────────────────────────────────┬───────────────────────────────────┘
                                  │  dispatch table (static mapping)
┌────────────────────────────────▼───────────────────────────────────┐
│ CONNECTOR LAYER  — existing, extended with an `act` capability       │
│   • read capability   (ingestion — exists today)                    │
│   • act  capability   (write-back — NEW)                             │
│   • downstream: Atlassian Rovo MCP Server (full ~30-40 tool catalog) │
│     → consumed here, NEVER exposed upward to the agent               │
└──────────────────────────────────────────────────────────────────────┘
```

Almost every layer reuses something that already exists. Only the Assistant and Capability layers are genuinely new code.

---

## 4. The tool-curation model — *curate, don't proxy*

> **This section answers the central design risk: context poisoning.**

The Atlassian Rovo MCP Server exposes a large catalog (search, create, update, transition, comment, assign, bulk-manage, summarize, Compass components — 30-40 tools). The naive integration — pass that catalog straight through to the assistant's LLM — has three failure modes:

1. **Token cost** — every tool's JSON Schema sits in the prompt on every turn.
2. **Selection accuracy** — LLM tool-picking degrades sharply as the catalog grows past ~15-20 tools.
3. **Security surface** — every downstream tool becomes invokable.

### The rule: a three-tier tool model

| Tier | What | Who sees it |
|---|---|---|
| **Tier 1 — Raw MCP tools** | The full Atlassian Rovo catalog (and any future connector's catalog) | Only the **Connector layer**. Never the agent. |
| **Tier 2 — Nexus Capability tools** | A curated, hand-defined facade of ~10 intent-shaped tools | The **agent's LLM**. This is the *only* tool list it sees. |
| **Tier 3 — Per-turn subset (optional)** | A cheap classifier loads only the relevant capability group per turn | Refinement; see below |

The Connector layer behaves like a **mini MCP gateway**: it consumes the full downstream catalog but applies *visibility filtering* — only the curated capabilities are reachable, the rest of the catalog stays dark.

### The dispatch table

Each Tier-2 capability maps statically to one or more Tier-1 tools:

| Tier-2 capability (agent sees) | Tier-1 Atlassian tool(s) it dispatches to |
|---|---|
| `search_corpus(query)` | — Nexus retrieval pipeline (no Atlassian) |
| `find_skills(query)` | — existing Nexus MCP tool |
| `get_jira_issue(key)` | `getJiraIssue` |
| `search_jira(query)` | `searchJiraIssuesUsingJql` |
| `search_confluence(query, space?)` | `searchConfluencePages` |
| `get_confluence_page(id)` | `getConfluencePage` |
| `propose_jira_changes(key, instruction)` | *analysis only* — emits an `ActionProposal` |
| `propose_confluence_update(id, instruction)` | *analysis only* — emits an `ActionProposal` |
| `confirm_action(proposal_id)` | on confirm → `createJiraSubtask`, `transitionJiraIssue`, `addJiraComment`, `assignJiraIssue`, `updateConfluencePage`, `createConfluencePage`, … |
| `reject_action(proposal_id)` | none |

### Why this also solves "broader mutations" for free

The user wants broad write support — transitions, comments, assignments, page creation, not just two flagship actions. **None of those become separate agent tools.** They are all folded behind `propose_jira_changes` / `propose_confluence_update`.

The breadth lives *inside* the proposal step: a planning LLM call (given the issue + the user's instruction) emits an **action plan** — a list of typed mutations:

```jsonc
// ActionProposal.plan — produced by propose_jira_changes
[
  { "op": "create_subtask", "summary": "Write migration script", "estimate": "2h" },
  { "op": "create_subtask", "summary": "Add rollback path",       "estimate": "1h" },
  { "op": "add_comment",    "body": "Split per the design review on 2026-05-20." },
  { "op": "transition",     "to": "In Progress" },
  { "op": "assign",         "assignee": "current_user" }
]
```

The agent's tool list stays at ~10 no matter how many mutation *types* we support. Adding a new mutation type later means extending the plan schema + the connector's executor — **zero growth in the agent-visible surface.** This is the core elegance of curate-don't-proxy.

### Tier 3 — optional per-turn subsetting

Even ~10 tools is comfortable for modern LLMs, so Tier 3 is a *later* refinement, not required for Phase 1. If we want it: a heuristic classifier (mirroring the existing `retrieval/classifier.py`) labels the turn as `read` / `jira-action` / `confluence-action` and loads only that group (~4-6 tools). Documented here so the door stays open.

---

## 5. Connector layer — the new `act` capability

Today a connector is read-only: list resources, get a resource, subscribe to updates. We extend the connector abstraction to be **bidirectional**:

```python
class ConnectorCapability(StrEnum):
    READ = "read"   # list / get / subscribe — ingestion (exists today)
    ACT  = "act"    # invoke a write tool — NEW
```

A connector declares which capabilities it supports. The `ConnectorManager` gains:

- `list_act_tools(connector) -> list[ToolSpec]` — the downstream write catalog (Tier 1).
- `invoke_act(connector, tool_name, args, *, as_user) -> ToolResult` — execute one write, on behalf of a specific user (see §6).

### Consuming the Atlassian Rovo MCP Server

The Atlassian Rovo MCP Server is a **remote** MCP server (HTTP/SSE transport at `mcp.atlassian.com`, OAuth 2.1). Nexus's current `connectors/mcp_client.py` only speaks **stdio** (for local connector subprocesses).

**Required work:** add a remote/HTTP-SSE transport mode to the MCP client, with OAuth 2.1 bearer-token auth. This is the single biggest piece of net-new connector plumbing in this slice.

> ⚠️ The legacy endpoint `mcp.atlassian.com/v1/sse` is deprecated after **30 June 2026** — target the current endpoint from day one.

The Atlassian server is registered as a connector of type `atlassian-rovo` in `nexus.yaml`; it is *not* used for ingestion (Nexus's own connectors still do that) — it is purely the `act` + live-read backend.

---

## 6. Per-user OAuth and identity

**Decision:** writes run as the **real user**, via per-user OAuth 2.1 — not a shared service account. This matches how the Atlassian Rovo MCP Server is designed (it always acts within the signed-in user's permissions) and gives correct attribution + permission enforcement for free.

### The flow

```
Settings → "Connect Atlassian account"
   → GET  /auth/atlassian/start          (Nexus builds the authorize URL, PKCE)
   → Atlassian OAuth 2.1 consent screen
   → GET  /auth/atlassian/callback?code= (Nexus exchanges code → access + refresh token)
   → tokens encrypted at rest, keyed by (user_id)
```

### Token use

- Every assistant read/write for a user attaches **that user's** Atlassian token to the downstream MCP call.
- Refresh tokens are used transparently; on hard expiry the assistant returns a *"reconnect your Atlassian account"* message with a link — it never silently fails.
- A user who has not connected gets a friendly prompt the first time they hit a Jira/Confluence capability. Corpus/skill capabilities still work without it.

### Identity per channel

| Channel | How the user is known |
|---|---|
| Nexus UI | the logged-in Nexus user (from `/me`) |
| MCP tools | the MCP server is launched per-user; `--user` (or `NEXUS_USER`) at launch, alongside the existing `--product`. Phase-1 simplification, documented. |
| MS Teams | the Teams/AAD identity, mapped to a Nexus user — Phase 2. |

Token storage is **encrypted** (a new `assistant_identities` table; see §9). Encryption key from env, never committed.

---

## 7. The assistant agent loop

A small **LangGraph** graph — the same library the council already uses, so no new orchestration dependency.

```
        START
          │
   ┌──────▼───────┐      tool calls       ┌──────────────┐
   │  agent (LLM) │──────────────────────►│  tool executor│
   │   node       │◄──────────────────────│  node         │
   └──────┬───────┘    tool results       └──────────────┘
          │  (loop until the LLM emits a final answer)
          ▼
         END
```

- **agent node** — an LLM call with the curated Tier-2 tool list (§4) + conversation history + a system prompt that explains the assistant's scope and the propose-before-write rule.
- **tool executor node** — runs the requested capability, returns the result; for `propose_*` it returns the `ActionProposal` summary, *not* a completed write.
- **Loop** — standard tool-use loop, capped at N iterations (config) to bound cost.
- **Streaming** — each node delta is published over SSE, exactly like the council's `_SessionHub` (§ reuse `nexus/council/runner.py` patterns).
- **Memory** — conversation turns persisted in SQLite; the loop is product-scoped (Invariant 1) — every tool call carries `product_id`.

Reused: `nexus/llm/client.py` (`ChatClient`), `nexus/retrieval/pipeline.py` (as the `search_corpus` tool), the SSE hub pattern.

---

## 8. Action proposals and human-in-the-loop confirmation

This is where the design honors **Invariant 3 — humans approve, agents draft.** The industry-standard write pattern (tool returns "pending approval" → user previews → confirms → execute) is *exactly* Nexus's existing proposal/approval pattern. So we reuse it.

### `ActionProposal` — a sibling of `SkillProposal`

```python
class ActionProposal(BaseModel):
    id: str
    conversation_id: str
    product_id: str
    requested_by: UserId
    target: ActionTarget          # {system: "jira"|"confluence", key/id: ...}
    instruction: str              # the user's natural-language ask
    plan: list[ActionStep]        # typed mutations (see §4)
    preview: str                  # human-readable preview — a task list, or a doc diff
    status: Literal["pending", "confirmed", "rejected", "executed", "failed"]
    created_at: datetime
    confirmed_by: UserId | None
    executed_at: datetime | None
    result: dict | None           # connector response, for the audit trail
```

### The flow

```
user: "break JIRA-412 into subtasks and move it to In Progress"
   │
agent → propose_jira_changes("JIRA-412", instruction)
   │      → planning LLM call → ActionProposal{plan:[...], preview:"..."}
   │      → status=pending, persisted
   ▼
channel renders the PREVIEW + Confirm / Reject
   • UI       → an action-proposal card (extends the Proposals screen)
   • MCP      → returned as structured content; agent shows it; user says confirm
   • Teams    → an Adaptive Card with Confirm / Reject buttons   (Phase 2)
   │
user confirms → confirm_action(proposal_id)
   │      → ConnectorManager.invoke_act(...) for each plan step, as the user
   │      → status=executed, result stored, activity-timeline entry written
   ▼
agent reports back what changed, with links to the created/updated items
```

### Preview format

- **Jira** — a rendered task list / change summary ("2 subtasks, 1 comment, 1 transition").
- **Confluence** — a **diff** between the current page body and the proposed body.

The proposal is **never** auto-confirmed. `confirm_action` is the only path to a write, and it requires an explicit user act in whatever channel raised it. Confirmation is *reactive* — only writes trigger it; reads never do.

---

## 9. Data model additions

Three new SQLite tables (in the registry DB, or a dedicated `assistant.sqlite` — TBD in review):

| Table | Purpose |
|---|---|
| `conversations` | one row per assistant conversation: `id`, `product_id`, `user_id`, `channel`, `created_at`, `last_active_at` |
| `conversation_messages` | turns: `id`, `conversation_id`, `role`, `content`, `tool_calls_js`, `created_at` |
| `action_proposals` | the `ActionProposal` rows (§8) |
| `assistant_identities` | per-user OAuth tokens: `user_id`, `provider`, `access_token_enc`, `refresh_token_enc`, `expires_at`, `scopes` |

`action_proposals` deliberately mirrors the existing `proposals` queue so the UI and audit code can treat both uniformly.

---

## 10. API surface additions

All under the existing FastAPI app, product-scoped per Invariant 1.

| Route | Purpose |
|---|---|
| `POST /products/{id}/assistant/messages` | start or continue a conversation; body `{conversation_id?, text}` |
| `GET  /assistant/conversations/{cid}/stream` | SSE — agent deltas, tool calls, action proposals |
| `GET  /products/{id}/assistant/conversations` | list a user's conversations |
| `GET  /assistant/conversations/{cid}` | full transcript |
| `POST /assistant/actions/{id}/confirm` | execute an `ActionProposal` |
| `POST /assistant/actions/{id}/reject` | discard an `ActionProposal` |
| `GET  /auth/atlassian/start` | begin per-user OAuth (PKCE) |
| `GET  /auth/atlassian/callback` | OAuth code exchange |
| `GET  /products/{id}/assistant/identity` | whether the current user has connected Atlassian |

SSE reuses the `useEventStream` hook and the council's streaming machinery.

---

## 11. MCP server additions — the coding-agent channel

Nexus already runs an MCP server (`nexus/mcp_server/`). We add a **small, curated** set of assistant tools to it — consistent with the curate-don't-proxy rule (§4); we do **not** surface the raw Atlassian catalog here either.

New tools on the Nexus MCP server:

| Tool | Notes |
|---|---|
| `assistant_ask(query)` | runs the agent loop, returns the answer |
| `assistant_get_issue(key)` | live Jira lookup |
| `assistant_search_confluence(query, space?)` | live Confluence search |
| `assistant_propose_changes(target, instruction)` | returns an `ActionProposal` preview |
| `assistant_confirm_action(proposal_id)` | executes it |

A coding agent mid-task can call `assistant_get_issue("JIRA-412")` to ground itself, or `assistant_propose_changes` then surface the preview to *its* user for confirmation. The confirmation responsibility is delegated to the calling agent — Nexus still refuses to write without an explicit `confirm`.

---

## 12. UI additions

| Screen | What |
|---|---|
| `components/screens/Assistant.tsx` | a chat panel — message list + composer, streams over SSE. New route `/p/[product]/assistant`. |
| Action-proposal card | rendered inline in the chat *and* on the existing Proposals screen — Confirm / Reject, with the §8 preview (task list or doc diff). |
| Settings → integrations | a "Connect Atlassian account" button + connection status, driven by `/auth/atlassian/*`. |

All within the existing design system (`DESIGN.md`) — no new dependencies expected beyond a diff renderer for Confluence previews (the `react-diff-viewer` that was already deferred in `INTEGRATION.md`).

---

## 13. Guardrails, RBAC, audit

- **RBAC** — extends the existing `getPerms`. A new `canRunAssistantActions` permission: `product_admin` yes, `sme` read-only (can ask, cannot confirm writes). Org admins configurable.
- **Per-product capability allowlist** — `nexus.yaml` declares which `act` operations are enabled per product. A product can run read-only even if the connector supports writes.
- **Audit trail** — every `ActionProposal` (proposed → confirmed → executed) writes to the existing **activity timeline**. The proposal stores who asked, the plan, who confirmed, and the connector response.
- **Observability** — the agent loop is traced in **Langfuse** like the council: every LLM call, every tool call, token cost.
- **Prompt-injection** — live Jira/Confluence content passes through the existing `retrieval/guard.py` redaction before it reaches the agent LLM, so a malicious ticket body cannot hijack the loop.
- **Loop bound** — the agent loop is capped at N iterations to bound cost and prevent runaway tool-calling.

---

## 14. Phasing

| Phase | Scope |
|---|---|
| **Phase 1** (this slice) | Agent loop · curated capability layer · `act` connector capability + Atlassian Rovo MCP (remote transport) · per-user OAuth · `ActionProposal` + confirm/reject · broad Jira/Confluence mutations · **channels: MCP tools + Nexus UI chat** |
| **Phase 2** | **MS Teams bot adapter** — Bot Framework / Teams SDK; AAD→Nexus user mapping; Adaptive Card confirmations. The brain is unchanged; this is a thin adapter. |
| **Phase 3** | Slack adapter · Tier-3 per-turn tool subsetting · additional connectors (GitHub issues actions, etc.) |

---

## 15. Proposed ADRs (to fold into `ENGINEERING.md`)

1. **Curate, don't proxy, downstream MCP tools.** The agent LLM sees only a hand-defined ~10-tool facade; downstream catalogs (Atlassian Rovo) stay behind the connector boundary. Rationale: tool-selection accuracy, token cost, security surface (§4).
2. **Per-user OAuth for write actions.** Writes run as the real user, never a shared service account. Rationale: correct attribution + permission enforcement; matches the Atlassian server's own model (§6).
3. **Action proposals reuse the proposal/approval pattern.** `ActionProposal` is a sibling of `SkillProposal`; no write happens without an explicit `confirm`. Rationale: consistency with Invariant 3 (§8).
4. **Broad mutations live in the plan, not the tool list.** Mutation types are entries in an `ActionProposal.plan`, not separate agent tools. Rationale: the agent surface stays constant as write coverage grows (§4).

---

## 16. Open questions / risks

1. **MCP remote transport** is net-new client plumbing (HTTP/SSE + OAuth). It is the critical-path technical risk for Phase 1 — prototype it first.
2. **Token encryption key management** — env var for now; revisit if Nexus ever gets a secrets manager.
3. **Conversation retention** — how long do we keep transcripts? Propose 90 days, configurable.
4. **MCP-channel user identity** — Phase 1 ties it to the `--user` launch flag. Acceptable? Or defer the MCP write-channel to Phase 2 and ship Phase 1's MCP channel read-only?
5. **Atlassian rate limits** — the Rovo MCP server has quotas; the connector layer needs a rate-limit-aware retry, distinct from the existing circuit breakers.
6. **Cost** — an agentic loop with tool calls is more expensive than a single retrieval. The loop bound + Langfuse cost tracking mitigate; a per-product budget cap may be worth adding.

---

## 17. Code map — where the new modules land

```
nexus/
├── assistant/                      NEW — the brain
│   ├── loop.py                     LangGraph agent loop
│   ├── capabilities.py             Tier-2 curated tool definitions
│   ├── dispatch.py                 Tier-2 → Tier-1 dispatch table
│   ├── proposals.py                ActionProposal model + queue
│   └── conversations.py            conversation memory store
├── connectors/
│   ├── manager.py                  EXTEND — list_act_tools, invoke_act
│   ├── mcp_client.py               EXTEND — remote HTTP/SSE transport + OAuth
│   └── atlassian.py                NEW — Atlassian Rovo connector binding
├── auth/                           NEW
│   └── atlassian_oauth.py          OAuth 2.1 PKCE flow + token store
├── api/routes/
│   └── assistant.py                NEW — the §10 route surface
└── mcp_server/
    └── tools.py                    EXTEND — the §11 assistant tools

nexus-ui/
├── app/p/[product]/assistant/      NEW route
├── components/screens/
│   └── Assistant.tsx               NEW chat screen
└── lib/api/index.ts                EXTEND — assistant client functions
```

---

## 18. References

- [Atlassian Remote (Rovo) MCP Server](https://github.com/atlassian/atlassian-mcp-server) — read+write Jira/Confluence over MCP, OAuth 2.1, GA Feb 2026.
- [Atlassian — Extend Atlassian into any AI assistant using MCP](https://www.atlassian.com/platform/remote-mcp-server)
- [AWS — Human-in-the-loop confirmation with Bedrock Agents](https://aws.amazon.com/blogs/machine-learning/implement-human-in-the-loop-confirmation-with-amazon-bedrock-agents/) — the propose/preview/confirm pattern.
- [Elastic — Human-in-the-loop AI agents with LangGraph](https://www.elastic.co/search-labs/blog/human-in-the-loop-hitllanggraph-elasticsearch)
- [What is an MCP Gateway](https://dev.to/composiodev/what-is-an-mcp-gateway-and-why-do-enterprise-ai-teams-need-one-in-2026-1lie) — visibility filtering / policy enforcement at the gateway.
- [Microsoft — Agents in Teams overview](https://learn.microsoft.com/en-us/microsoftteams/platform/toolkit/build-an-ai-agent-in-teams) — Teams SDK, MCP-native, for Phase 2.
- [MindStudio — Multi-channel AI agent deployment](https://www.mindstudio.ai/blog/multi-channel-ai-agent-deployment-slack-teams) — the four-layer channel/agent/tool/memory architecture.
