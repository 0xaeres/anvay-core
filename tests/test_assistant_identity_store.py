"""AssistantStore — per-user OAuth identity storage + OAuth flow state."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nexus.assistant.store import AssistantStore
from nexus.auth.atlassian_oauth import TokenSet
from nexus.auth.token_cipher import TokenCipher


def _token(expires_in: int = 3600) -> TokenSet:
    return TokenSet(
        access_token="PLAINTEXT_ACCESS_xyz",
        refresh_token="PLAINTEXT_REFRESH_xyz",
        scope="read:jira-work",
        expires_at=(datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat(),
    )


def _store(tmp_path: Path) -> AssistantStore:
    return AssistantStore(
        tmp_path / "assistant.db", cipher=TokenCipher(TokenCipher.generate_key())
    )


def test_identity_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.identities_enabled is True
    store.save_identity("alice", provider="atlassian", token=_token())

    got = store.get_identity("alice", provider="atlassian")
    assert got is not None
    assert got.access_token == "PLAINTEXT_ACCESS_xyz"
    assert got.refresh_token == "PLAINTEXT_REFRESH_xyz"
    assert not got.is_expired()


def test_tokens_are_encrypted_at_rest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_token())

    raw = (
        sqlite3.connect(tmp_path / "assistant.db")
        .execute("SELECT access_token_enc, refresh_token_enc FROM assistant_identities")
        .fetchone()
    )
    # The plaintext must not appear anywhere in the stored columns.
    assert "PLAINTEXT_ACCESS_xyz" not in (raw[0] or "")
    assert "PLAINTEXT_REFRESH_xyz" not in (raw[1] or "")


def test_identity_feature_disabled_without_cipher(tmp_path: Path) -> None:
    store = AssistantStore(tmp_path / "assistant.db")  # no cipher
    assert store.identities_enabled is False
    with pytest.raises(RuntimeError):
        store.save_identity("alice", provider="atlassian", token=_token())
    with pytest.raises(RuntimeError):
        store.get_identity("alice", provider="atlassian")


def test_delete_identity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_token())
    assert store.delete_identity("alice", provider="atlassian") is True
    assert store.get_identity("alice", provider="atlassian") is None
    assert store.delete_identity("alice", provider="atlassian") is False


def test_oauth_flow_is_single_use(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_oauth_flow(
        "state-abc", provider="atlassian", code_verifier="verifier-1", user_id="alice"
    )
    flow = store.pop_oauth_flow("state-abc")
    assert flow is not None
    assert flow["code_verifier"] == "verifier-1"
    assert flow["user_id"] == "alice"
    # second pop returns nothing — anti-replay
    assert store.pop_oauth_flow("state-abc") is None


def test_get_identity_for_unknown_user(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get_identity("nobody", provider="atlassian") is None
