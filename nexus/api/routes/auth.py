"""Per-user Atlassian OAuth — see docs/ASSISTANT-LAYER.md §6.

The PKCE flow: `/auth/atlassian/start` mints a `state` + code verifier and returns
the Atlassian consent URL; `/auth/atlassian/callback` consumes the `state`,
exchanges the code, and stores the user's encrypted tokens.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from nexus.api.deps import get_assistant_store, get_atlassian_oauth, get_config_dep
from nexus.assistant.store import AssistantStore
from nexus.auth.atlassian_oauth import (
    AtlassianOAuth,
    AtlassianOAuthError,
    generate_code_verifier,
    generate_state,
)
from nexus.config import NexusConfig

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/atlassian", tags=["auth"])

_PROVIDER = "atlassian"


def _guard(config: NexusConfig, store: AssistantStore) -> None:
    if not config.atlassian.enabled:
        raise HTTPException(
            status_code=409, detail="Atlassian integration is disabled in nexus.yaml"
        )
    if not store.identities_enabled:
        raise HTTPException(
            status_code=409,
            detail="token encryption is not configured — set the NEXUS_TOKEN_KEY env var",
        )


@router.get("/start")
async def start(
    actor: str = "admin",
    store: AssistantStore = Depends(get_assistant_store),
    config: NexusConfig = Depends(get_config_dep),
    oauth: AtlassianOAuth = Depends(get_atlassian_oauth),
) -> dict:
    """Begin the OAuth flow. Returns the Atlassian consent URL for the UI to open."""
    _guard(config, store)
    state = generate_state()
    verifier = generate_code_verifier()
    store.save_oauth_flow(
        state, provider=_PROVIDER, code_verifier=verifier, user_id=actor
    )
    return {"authorize_url": oauth.authorize_url(state=state, verifier=verifier)}


@router.get("/callback")
async def callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    store: AssistantStore = Depends(get_assistant_store),
    config: NexusConfig = Depends(get_config_dep),
    oauth: AtlassianOAuth = Depends(get_atlassian_oauth),
) -> RedirectResponse:
    """OAuth redirect target. Exchanges the code and stores the user's tokens."""
    redirect = config.atlassian.post_auth_redirect

    if error:
        log.warning("atlassian oauth: provider returned error: %s", error)
        return RedirectResponse(url=f"{redirect}?atlassian=error")
    if not code or not state:
        return RedirectResponse(url=f"{redirect}?atlassian=error")

    flow = store.pop_oauth_flow(state)
    if not flow:
        log.warning("atlassian oauth: unknown or replayed state")
        return RedirectResponse(url=f"{redirect}?atlassian=error")

    try:
        token = await oauth.exchange_code(code=code, verifier=flow["code_verifier"])
    except AtlassianOAuthError as e:
        log.warning("atlassian oauth: code exchange failed: %s", e)
        return RedirectResponse(url=f"{redirect}?atlassian=error")

    store.save_identity(flow["user_id"], provider=_PROVIDER, token=token)
    log.info("atlassian oauth: connected %s", flow["user_id"])
    return RedirectResponse(url=f"{redirect}?atlassian=connected")


@router.delete("/identity")
async def disconnect(
    actor: str = "admin",
    store: AssistantStore = Depends(get_assistant_store),
) -> dict:
    """Disconnect an Atlassian account."""
    removed = store.delete_identity(actor, provider=_PROVIDER)
    return {"ok": True, "removed": removed}
