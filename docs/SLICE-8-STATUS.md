# Slice 8 — Assistant Layer Status

Implements [`docs/ASSISTANT-LAYER.md`](./ASSISTANT-LAYER.md). Delivered in increments;
this doc tracks each.

**Progress:** Increment 1 ✅ · Increment 2 ✅ · Increment 3 ✅ · Increment 4 ✅ — **Slice 8 complete.**

## Increment 1 — the backend brain ✅

The Assistant layer, Capability layer, and API surface — runnable and tested
today against a stub connector. The Atlassian remote-MCP transport + per-user
OAuth (the §16 critical-path risk) sit behind clean ports and land in later
increments without touching the brain.

### What's implemented

| Module | Status | Notes |
|---|---|---|
| `nexus/assistant/models.py` | ✅ | `ActionProposal` (sibling of `SkillProposal`), `ActionStep`, `Conversation`, `ConversationMessage`; typed mutation op sets (`JIRA_OPS`, `CONFLUENCE_OPS`). |
| `nexus/assistant/store.py` | ✅ | SQLite store — conversations, messages, action proposals. Mirrors `council/queue.py`. |
| `nexus/assistant/connector_port.py` | ✅ | `ReadPort` / `ActPort` Protocols + `FakeConnectorPort` (deterministic stand-in). |
| `nexus/assistant/capabilities.py` | ✅ | The curated **Tier-2** tool facade — 8 intent-shaped tools. Curate-don't-proxy (ADR). |
| `nexus/assistant/loop.py` | ✅ | The agent loop — sequential JSON-action tool-calling, iteration-capped. |
| `nexus/assistant/executor.py` | ✅ | `execute_proposal` — the only write path; runs after human confirm. |
| `nexus/api/routes/assistant.py` | ✅ | 7 routes — messages, conversations, action confirm/reject. |
| `nexus/config.py` | ✅ | Optional `models.assistant` role; falls back to `council_agents`. |

### API surface (live)

| Route | Purpose |
|---|---|
| `POST /products/{id}/assistant/messages` | start/continue a conversation; runs one agent turn |
| `GET /products/{id}/assistant/conversations` | list conversations |
| `GET /assistant/conversations/{cid}` | full transcript |
| `GET /products/{id}/assistant/actions` | list action proposals |
| `GET /assistant/actions/{id}` | one action proposal |
| `POST /assistant/actions/{id}/confirm` | execute a drafted proposal (the only write path) |
| `POST /assistant/actions/{id}/reject` | discard a drafted proposal |

### Tests / lint

- `tests/test_assistant_store.py` — conversation / message / proposal CRUD (5 tests).
- `tests/test_assistant_loop.py` — agent loop with a scripted fake LLM: read-tool flow,
  propose flow, unknown-tool resilience, iteration cap (4 tests).
- `tests/test_assistant_executor.py` — confirm/execute happy path, failure recording,
  no double-execute (3 tests).
- Full suite: **116 passing** (104 → 116). `ruff check nexus tests` clean.

### Design decisions taken during implementation

- **Plain async loop, not LangGraph.** The design doc (§7) suggested LangGraph for
  consistency with the council. In implementation the assistant turn is strictly
  sequential (LLM ↔ tool), so LangGraph's fan-out/fan-in would be ceremony. The loop
  is a plain async function; durable state lives in `AssistantStore`. (Refines §7.)
- **`confirm_action` is not an agent tool.** Confirmation is a human act exposed as an
  API route, never something the autonomous loop can call. The agent registry is read
  tools + the two `propose_*` tools only. (Refines §4.)
- **JSON-action tool calls**, not native tool-calling — consistent with how the council
  parses JSON from text; no change needed to `ChatClient`.

### Deliberately deferred to later increments

- **Atlassian Rovo connector** — `FakeConnectorPort` is wired in `routes/assistant.py`;
  swap for `AtlassianConnectorPort` when the remote MCP transport lands.
- **Remote MCP transport + per-user OAuth** — Increment 2 (the §16 critical-path risk).
- **SSE streaming** — `POST /messages` is synchronous for now; SSE arrives with the UI.
- **Nexus UI chat panel** — Increment 4.
- **MCP server assistant tools** (coding-agent channel) — Increment 3.
- **Real corpus retrieval** in `search_corpus` — `ToolContext.retrieval` is `None`;
  the tool degrades gracefully. Wire `RetrievalContext` in a later increment.

## Increment 2 — Atlassian connector + per-user OAuth ✅

The §16 critical-path risk, de-risked: the remote MCP transport, OAuth 2.1 PKCE,
and `AtlassianConnectorPort` — all behind clean seams and exercised by tests with
mocked HTTP (no live Atlassian site required).

### What's implemented

| Module | Status | Notes |
|---|---|---|
| `nexus/connectors/remote_mcp.py` | ✅ | `RemoteMCPClient` — JSON-RPC 2.0 over Streamable HTTP; `initialize` / `tools/list` / `tools/call`; handles JSON **and** SSE responses; echoes `Mcp-Session-Id`. |
| `nexus/auth/token_cipher.py` | ✅ | Fernet encryption for tokens at rest; key from `NEXUS_TOKEN_KEY`. |
| `nexus/auth/atlassian_oauth.py` | ✅ | OAuth 2.0 (3LO) + PKCE — verifier/challenge helpers, `authorize_url`, `exchange_code`, `refresh`, `TokenSet`. |
| `nexus/assistant/atlassian.py` | ✅ | `AtlassianConnectorPort` — `ReadPort` + `ActPort` over the Rovo MCP server; per-user token resolution with silent refresh; the curated→Atlassian dispatch table. |
| `nexus/assistant/store.py` | ✅ | `assistant_identities` (encrypted tokens) + `oauth_flows` (single-use PKCE state) tables + methods. |
| `nexus/api/routes/auth.py` | ✅ | `/auth/atlassian/start`, `/auth/atlassian/callback`, `DELETE /auth/atlassian/identity`. |
| `nexus/config.py` | ✅ | `AtlassianCfg` block (optional; `enabled: false` by default). |
| deps + wiring | ✅ | `get_connector_port` returns `AtlassianConnectorPort` when Atlassian is enabled + a token key is set, else `FakeConnectorPort`. `GET /products/{id}/assistant/identity`. |

### Atlassian tool dispatch (curate-don't-proxy in action)

The agent still sees only the 8 curated tools. `AtlassianConnectorPort` is the
single place mapping them to real Rovo MCP tools (verified against Atlassian's
"Supported tools" docs):

| Curated capability / action op | Atlassian Rovo tool |
|---|---|
| `get_jira_issue` | `getJiraIssue` |
| `search_jira` | `searchJiraIssuesUsingJql` |
| `search_confluence` | `searchConfluenceUsingCql` |
| `get_confluence_page` | `getConfluencePage` |
| `create_subtask` | `createJiraIssue` |
| `transition` | `transitionJiraIssue` |
| `add_comment` | `addCommentToJiraIssue` |
| `assign` / `update_field` | `editJiraIssue` |
| `update_page` / `create_page` | `updateConfluencePage` / `createConfluencePage` |

### Tests / lint

- `test_token_cipher.py` (4), `test_atlassian_oauth.py` (6), `test_remote_mcp.py` (5),
  `test_atlassian_connector.py` (7), `test_assistant_identity_store.py` (6).
- Full suite: **143 passing** (116 → 143). `ruff check nexus tests` clean.

### Notes / follow-ups

- Atlassian tool **argument shapes** are best-effort; re-verify against a live
  `tools/list` on first connection to a real site (flagged in `atlassian.py`).
- The `RemoteMCPClient` re-runs `initialize` per cached client; per-user clients
  are cached, so this is once per user — acceptable. Connection pooling is shared.

## Increment 3 — MCP server channel ✅

Curated assistant tools on the Nexus MCP server — so a coding agent (Claude
Desktop, Cursor, …) can query and act on Jira/Confluence mid-task.

### What's implemented

| Module | Status | Notes |
|---|---|---|
| `nexus/assistant/factory.py` | ✅ | `build_assistant_store` / `build_connector_port` — shared by the FastAPI app and the MCP server so wiring lives in one place. `api/deps.py` refactored onto it. |
| `nexus/mcp_server/assistant_tools.py` | ✅ | `AssistantToolState` + 5 curated tools. |
| `nexus/mcp_server/server.py` | ✅ | Registers the 5 assistant tools; new `--user` / `NEXUS_USER` arg drives per-user OAuth attribution. |

### The 5 MCP assistant tools (curate-don't-proxy holds here too)

| MCP tool | Purpose |
|---|---|
| `assistant_ask` | Run the agent loop — answers + may draft (never apply) an action proposal |
| `assistant_get_jira_issue` | Direct live Jira lookup |
| `assistant_search_confluence` | Direct live Confluence search |
| `assistant_list_actions` | List drafted action proposals awaiting confirmation |
| `assistant_confirm_action` | Execute a drafted proposal — the only MCP write path |

A coding agent surfaces an `action_proposal` preview to *its* user and only calls
`assistant_confirm_action` on explicit approval — the human-in-the-loop gesture,
mediated by the calling agent. The raw Atlassian catalogue never reaches any LLM.

### Tests / lint

- `tests/test_mcp_assistant_tools.py` — 7 tests: reads, the ask loop, proposal
  surfacing, list/confirm, error paths.
- Full suite: **150 passing** (143 → 150). `ruff check nexus tests` clean.

## Increment 4 — UI chat panel ✅

The Nexus UI gets a chat panel — the third channel onto the same brain.

### What's implemented (`nexus-ui/`)

| File | Status | Notes |
|---|---|---|
| `components/screens/Assistant.tsx` | ✅ | Chat panel — message list, composer, empty state with example prompts. |
| `app/p/[product]/assistant/{page,loading}.tsx` | ✅ | New route `/p/[product]/assistant`. |
| `lib/types.ts` | ✅ | `ActionProposal`, `ActionStep`, `AssistantTurn`, `AssistantIdentity`. |
| `lib/api/index.ts` | ✅ | `sendAssistantMessage`, `getAssistantIdentity`, `confirmAction`, `rejectAction`, `startAtlassianAuth`. |
| `components/shell/{SideNav,Shell,CommandPalette}.tsx` | ✅ | Nav entry, `g i` chord, command-palette entry. |

### Behaviour

- **Action-proposal cards** render inline in the chat: the rendered plan preview, a
  status badge, and **Confirm & apply / Reject** buttons. Confirm is the only path
  that writes — Invariant 3 holds at the UI too.
- **Connect-Atlassian banner** appears when Atlassian is enabled but the current
  user hasn't connected — links into the `/auth/atlassian/start` flow. When
  Atlassian is not configured, a muted note explains the stub-data mode.

### SSE streaming (ASSISTANT-LAYER.md §10)

`POST /products/{id}/assistant/messages` streams the turn as Server-Sent Events.
The agent loop (`run_turn`) takes an optional `on_event` sink; the route runs the
loop in a background task whose `on_event` callback feeds an `asyncio.Queue` that
the SSE generator drains. Events:

- `start` — `{conversation_id}`
- `message` — live `{type: tool_call|tool_result, tool, …}` progress
- `session_end` — `{reply, action_proposal, iterations}`
- `error` — `{message}`

The UI uses `fetch` (not `EventSource`, so the message travels in the POST body),
parses the SSE frames, and shows each tool call live — `▸ get_jira_issue` with a
spinner that resolves to ✓/✗. Synchronous callers (the MCP `assistant_ask` tool)
pass no `on_event` sink and the loop behaves exactly as before.

### Tests / lint

- `npx tsc --noEmit` (nexus-ui) — clean.
- Backend full suite: **152 passing**; `ruff check nexus tests` clean.
- `test_assistant_loop.py` covers the loop's event emission and the no-sink path.

---

## Slice 8 — summary

Four increments, all shipped. The Assistant layer is a conversational + action
layer over Jira/Confluence, reachable from MCP tools (coding agents) and the Nexus
UI chat panel, with **live SSE streaming** of agent progress. It reuses the
proposal/approval model (Invariant 3), per-user OAuth, RBAC, and the activity
timeline. **152 backend tests passing; UI type-checks clean.** The MS Teams
adapter remains as future work (Phase 2) — see `ASSISTANT-LAYER.md`.
