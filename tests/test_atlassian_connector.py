"""AtlassianConnectorPort — capability/action → Atlassian tool dispatch,
per-user token resolution, and silent refresh."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nexus.assistant.atlassian import AtlassianConnectorPort
from nexus.assistant.connector_port import ConnectorUnavailable
from nexus.assistant.models import ActionStep, ActionTarget
from nexus.assistant.store import AssistantStore
from nexus.auth.atlassian_oauth import TokenSet
from nexus.auth.token_cipher import TokenCipher
from nexus.config import AtlassianCfg


class FakeRemoteClient:
    """Stands in for RemoteMCPClient. Invokes the token provider per call,
    exactly as the real client does in its request headers."""

    def __init__(self, token_provider):
        self._token_provider = token_provider
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        await self._token_provider()  # exercises identity resolution + refresh
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": '{"ok": true}'}], "isError": False}

    async def aclose(self) -> None:
        pass


class FakeOAuth:
    def __init__(self) -> None:
        self.refresh_calls = 0

    async def refresh(self, *, refresh_token: str) -> TokenSet:
        self.refresh_calls += 1
        return TokenSet(
            access_token="refreshed-AT",
            refresh_token="refreshed-RT",
            scope="read",
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        )


def _store(tmp_path: Path) -> AssistantStore:
    return AssistantStore(
        tmp_path / "assistant.db", cipher=TokenCipher(TokenCipher.generate_key())
    )


def _valid_token() -> TokenSet:
    return TokenSet(
        access_token="AT",
        refresh_token="RT",
        scope="read",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )


def _port(store: AssistantStore, oauth=None) -> tuple[AtlassianConnectorPort, list]:
    created: list[FakeRemoteClient] = []

    def factory(token_provider):
        client = FakeRemoteClient(token_provider)
        created.append(client)
        return client

    port = AtlassianConnectorPort(
        cfg=AtlassianCfg(enabled=True),
        store=store,
        oauth=oauth or FakeOAuth(),
        client_factory=factory,
    )
    return port, created


def test_get_jira_issue_maps_to_getJiraIssue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_valid_token())
    port, created = _port(store)

    asyncio.run(port.get_jira_issue("PROJ-1", as_user="alice"))

    assert created[0].calls[0] == ("getJiraIssue", {"issueIdOrKey": "PROJ-1"})


def test_search_confluence_builds_cql_with_space(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_valid_token())
    port, created = _port(store)

    asyncio.run(port.search_confluence("retry logic", space="ENG", as_user="alice"))

    name, args = created[0].calls[0]
    assert name == "searchConfluenceUsingCql"
    assert 'text ~ "retry logic"' in args["cql"]
    assert 'space = "ENG"' in args["cql"]


def test_execute_create_subtask_maps_to_createJiraIssue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_valid_token())
    port, created = _port(store)

    step = ActionStep(op="create_subtask", args={"summary": "Sub A"})
    target = ActionTarget(system="jira", key="PROJ-1")
    res = asyncio.run(port.execute_step(step, target=target, as_user="alice"))

    name, args = created[0].calls[0]
    assert name == "createJiraIssue"
    assert args["fields"]["parent"]["key"] == "PROJ-1"
    assert args["fields"]["summary"] == "Sub A"
    assert args["fields"]["issuetype"]["name"] == "Subtask"
    assert res["applied"] is True


def test_execute_transition_comment_assign_map_correctly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_identity("alice", provider="atlassian", token=_valid_token())
    port, created = _port(store)
    target = ActionTarget(system="jira", key="PROJ-2")

    asyncio.run(
        port.execute_step(
            ActionStep(op="transition", args={"to": "Done"}), target=target, as_user="alice"
        )
    )
    asyncio.run(
        port.execute_step(
            ActionStep(op="add_comment", args={"body": "hi"}), target=target, as_user="alice"
        )
    )
    asyncio.run(
        port.execute_step(
            ActionStep(op="assign", args={"assignee": "bob"}), target=target, as_user="alice"
        )
    )
    # one client is cached per user, so all three calls land on created[0]
    tools = [name for name, _ in created[0].calls]
    assert tools == ["transitionJiraIssue", "addCommentToJiraIssue", "editJiraIssue"]


def test_missing_identity_raises_connector_unavailable(tmp_path: Path) -> None:
    store = _store(tmp_path)  # no identity saved for "bob"
    port, _created = _port(store)

    try:
        asyncio.run(port.get_jira_issue("X-1", as_user="bob"))
        raise AssertionError("expected ConnectorUnavailable")
    except ConnectorUnavailable as e:
        assert "connected" in str(e).lower()


def test_expired_token_is_refreshed_and_persisted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    expired = TokenSet(
        access_token="OLD",
        refresh_token="RT",
        scope="read",
        expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    )
    store.save_identity("alice", provider="atlassian", token=expired)
    oauth = FakeOAuth()
    port, _created = _port(store, oauth=oauth)

    asyncio.run(port.get_jira_issue("X-1", as_user="alice"))

    assert oauth.refresh_calls == 1
    refreshed = store.get_identity("alice", provider="atlassian")
    assert refreshed is not None
    assert refreshed.access_token == "refreshed-AT"
