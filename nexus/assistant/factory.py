"""Construction helpers for the Assistant layer.

Shared by the FastAPI app (`nexus/api/deps.py`) and the MCP server
(`nexus/mcp_server/assistant_tools.py`) so connector/store wiring — and the
"real Atlassian vs. stub" decision — lives in exactly one place.
"""

from __future__ import annotations

import logging
import os

from nexus.assistant.store import AssistantStore
from nexus.auth.token_cipher import TokenCipher, TokenCipherError
from nexus.config import NexusConfig

log = logging.getLogger(__name__)


def build_assistant_store(config: NexusConfig) -> AssistantStore:
    """An AssistantStore with token encryption enabled iff NEXUS_TOKEN_KEY is set."""
    cipher: TokenCipher | None = None
    key = os.environ.get("NEXUS_TOKEN_KEY")
    if key:
        try:
            cipher = TokenCipher(key)
        except TokenCipherError as e:
            log.warning("NEXUS_TOKEN_KEY invalid — per-user OAuth disabled: %s", e)
    return AssistantStore(
        config.storage.proposal_queue.parent / "assistant.db", cipher=cipher
    )


def build_connector_port(config: NexusConfig, store: AssistantStore):
    """The Assistant's read/act backend.

    Returns the real `AtlassianConnectorPort` when Atlassian is enabled AND a
    token-encryption key is configured; otherwise the stub `FakeConnectorPort`,
    so the Assistant is always runnable.
    """
    if config.atlassian.enabled and store.identities_enabled:
        from nexus.assistant.atlassian import AtlassianConnectorPort
        from nexus.auth.atlassian_oauth import AtlassianOAuth

        log.info("assistant: using AtlassianConnectorPort (live)")
        return AtlassianConnectorPort(
            cfg=config.atlassian, store=store, oauth=AtlassianOAuth(config.atlassian)
        )
    from nexus.assistant.connector_port import FakeConnectorPort

    log.info("assistant: using FakeConnectorPort (stub — Atlassian not configured)")
    return FakeConnectorPort()
