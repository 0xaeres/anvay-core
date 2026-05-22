"""SQLite store for assistant conversations + action proposals.

Mirrors the pattern of `nexus/council/queue.py` — a thin synchronous wrapper;
SQLite is fast enough that async is unnecessary.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from nexus.assistant.models import (
    ActionProposal,
    ActionStep,
    ActionTarget,
    Conversation,
    ConversationMessage,
    MessageRole,
    ProposalStatus,
)
from nexus.auth.atlassian_oauth import TokenSet
from nexus.auth.token_cipher import TokenCipher

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    product_id     TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    channel        TEXT NOT NULL DEFAULT 'ui',
    title          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_product
    ON conversations(product_id, last_active_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_name       TEXT,
    tool_args_js    TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv
    ON conversation_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS action_proposals (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    requested_by    TEXT NOT NULL,
    target_js       TEXT NOT NULL,
    instruction     TEXT NOT NULL,
    plan_js         TEXT NOT NULL,
    preview         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    confirmed_by    TEXT,
    executed_at     TEXT,
    result_js       TEXT,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_product
    ON action_proposals(product_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_conv
    ON action_proposals(conversation_id);

-- Per-user OAuth tokens (encrypted at rest). See docs/ASSISTANT-LAYER.md §6.
CREATE TABLE IF NOT EXISTS assistant_identities (
    user_id           TEXT NOT NULL,
    provider          TEXT NOT NULL,
    access_token_enc  TEXT NOT NULL,
    refresh_token_enc TEXT,
    scope             TEXT NOT NULL DEFAULT '',
    expires_at        TEXT NOT NULL,
    connected_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, provider)
);

-- Transient OAuth flow state (state -> PKCE verifier), consumed by the callback.
CREATE TABLE IF NOT EXISTS oauth_flows (
    state         TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


class AssistantStore:
    def __init__(self, db_path: Path, *, cipher: TokenCipher | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # When None, the per-user OAuth identity feature is disabled (no key set).
        self._cipher = cipher
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------ conversations

    def create_conversation(self, conv: Conversation) -> Conversation:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO conversations
                   (id, product_id, user_id, channel, title, created_at, last_active_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    conv.id,
                    conv.product_id,
                    conv.user_id,
                    conv.channel,
                    conv.title,
                    conv.created_at,
                    conv.last_active_at,
                ),
            )
        return conv

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return Conversation(**dict(row)) if row else None

    def list_conversations(self, *, product_id: str) -> list[Conversation]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE product_id = ? "
                "ORDER BY last_active_at DESC",
                (product_id,),
            ).fetchall()
        return [Conversation(**dict(r)) for r in rows]

    def touch_conversation(self, conversation_id: str, *, title: str | None = None) -> None:
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE conversations
                   SET last_active_at = ?,
                       title = CASE WHEN title = '' AND ? IS NOT NULL THEN ? ELSE title END
                   WHERE id = ?""",
                (ts, title, title, conversation_id),
            )

    # ------------------------------------------------------------ messages

    def add_message(self, msg: ConversationMessage) -> ConversationMessage:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO conversation_messages
                   (id, conversation_id, role, content, tool_name, tool_args_js, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    msg.id,
                    msg.conversation_id,
                    msg.role.value,
                    msg.content,
                    msg.tool_name,
                    json.dumps(msg.tool_args) if msg.tool_args is not None else None,
                    msg.created_at,
                ),
            )
        return msg

    def list_messages(self, conversation_id: str) -> list[ConversationMessage]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_messages WHERE conversation_id = ? "
                "ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    # ------------------------------------------------------------ action proposals

    def save_proposal(self, proposal: ActionProposal) -> ActionProposal:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO action_proposals
                   (id, conversation_id, product_id, requested_by, target_js,
                    instruction, plan_js, preview, status, created_at,
                    confirmed_by, executed_at, result_js, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    proposal.id,
                    proposal.conversation_id,
                    proposal.product_id,
                    proposal.requested_by,
                    proposal.target.model_dump_json(),
                    proposal.instruction,
                    json.dumps([s.model_dump() for s in proposal.plan]),
                    proposal.preview,
                    proposal.status.value,
                    proposal.created_at,
                    proposal.confirmed_by,
                    proposal.executed_at,
                    json.dumps(proposal.result) if proposal.result is not None else None,
                    proposal.error,
                ),
            )
        return proposal

    def get_proposal(self, proposal_id: str) -> ActionProposal | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM action_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        return _row_to_proposal(row) if row else None

    def list_proposals(
        self, *, product_id: str | None = None, status: str | None = None
    ) -> list[ActionProposal]:
        sql = "SELECT * FROM action_proposals WHERE 1=1"
        args: list = []
        if product_id:
            sql += " AND product_id = ?"
            args.append(product_id)
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def update_proposal(
        self,
        proposal_id: str,
        *,
        status: ProposalStatus,
        confirmed_by: str | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> bool:
        executed_at = (
            datetime.now(UTC).isoformat()
            if status in (ProposalStatus.EXECUTED, ProposalStatus.FAILED)
            else None
        )
        with self._conn() as conn:
            cur = conn.execute(
                """UPDATE action_proposals
                   SET status = ?,
                       confirmed_by = COALESCE(?, confirmed_by),
                       executed_at = COALESCE(?, executed_at),
                       result_js = COALESCE(?, result_js),
                       error = COALESCE(?, error)
                   WHERE id = ?""",
                (
                    status.value,
                    confirmed_by,
                    executed_at,
                    json.dumps(result) if result is not None else None,
                    error,
                    proposal_id,
                ),
            )
            return cur.rowcount > 0


    # ------------------------------------------------------------ identities

    @property
    def identities_enabled(self) -> bool:
        """True when a token-encryption key is configured."""
        return self._cipher is not None

    def _require_cipher(self) -> TokenCipher:
        if self._cipher is None:
            raise RuntimeError(
                "token encryption is not configured — set the NEXUS_TOKEN_KEY env "
                "var (generate one with TokenCipher.generate_key())"
            )
        return self._cipher

    def save_identity(self, user_id: str, *, provider: str, token: TokenSet) -> None:
        cipher = self._require_cipher()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO assistant_identities
                   (user_id, provider, access_token_enc, refresh_token_enc,
                    scope, expires_at, connected_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    user_id,
                    provider,
                    cipher.encrypt(token.access_token),
                    cipher.encrypt(token.refresh_token) if token.refresh_token else None,
                    token.scope,
                    token.expires_at,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_identity(self, user_id: str, *, provider: str) -> TokenSet | None:
        cipher = self._require_cipher()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM assistant_identities WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
        if not row:
            return None
        refresh_enc = row["refresh_token_enc"]
        return TokenSet(
            access_token=cipher.decrypt(row["access_token_enc"]),
            refresh_token=cipher.decrypt(refresh_enc) if refresh_enc else None,
            scope=row["scope"],
            expires_at=row["expires_at"],
        )

    def delete_identity(self, user_id: str, *, provider: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM assistant_identities WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------ oauth flows

    def save_oauth_flow(
        self, state: str, *, provider: str, code_verifier: str, user_id: str
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO oauth_flows
                   (state, provider, code_verifier, user_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (state, provider, code_verifier, user_id, datetime.now(UTC).isoformat()),
            )

    def pop_oauth_flow(self, state: str) -> dict | None:
        """Fetch and delete a pending OAuth flow — single use, anti-replay."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM oauth_flows WHERE state = ?", (state,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM oauth_flows WHERE state = ?", (state,))
        return dict(row)


def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
    d = dict(row)
    args = d.pop("tool_args_js", None)
    return ConversationMessage(
        id=d["id"],
        conversation_id=d["conversation_id"],
        role=MessageRole(d["role"]),
        content=d["content"],
        tool_name=d.get("tool_name"),
        tool_args=json.loads(args) if args else None,
        created_at=d["created_at"],
    )


def _row_to_proposal(row: sqlite3.Row) -> ActionProposal:
    d = dict(row)
    result = d.pop("result_js", None)
    return ActionProposal(
        id=d["id"],
        conversation_id=d["conversation_id"],
        product_id=d["product_id"],
        requested_by=d["requested_by"],
        target=ActionTarget(**json.loads(d["target_js"])),
        instruction=d["instruction"],
        plan=[ActionStep(**s) for s in json.loads(d["plan_js"] or "[]")],
        preview=d["preview"],
        status=ProposalStatus(d["status"]),
        created_at=d["created_at"],
        confirmed_by=d.get("confirmed_by"),
        executed_at=d.get("executed_at"),
        result=json.loads(result) if result else None,
        error=d.get("error"),
    )
