"""Atlassian OAuth 2.0 (3LO) with PKCE — see docs/ASSISTANT-LAYER.md §6.

Implemented by hand rather than via a library: the flow is small, and avoiding a
new dependency keeps the surface tight. PKCE (RFC 7636) protects the code
exchange even though Atlassian is a confidential client.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx

# --------------------------------------------------------------------------
# PKCE helpers
# --------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def generate_code_verifier() -> str:
    """A high-entropy, URL-safe PKCE code verifier (RFC 7636 §4.1)."""
    return _b64url(os.urandom(32))


def code_challenge(verifier: str) -> str:
    """The S256 challenge for a verifier (RFC 7636 §4.2)."""
    return _b64url(hashlib.sha256(verifier.encode()).digest())


def generate_state() -> str:
    """An anti-CSRF `state` value for the authorization request."""
    return _b64url(os.urandom(16))


# --------------------------------------------------------------------------
# Token model
# --------------------------------------------------------------------------


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None
    scope: str
    expires_at: str  # ISO-8601

    def is_expired(self, *, skew_s: int = 60) -> bool:
        exp = datetime.fromisoformat(self.expires_at)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        return datetime.now(UTC) >= exp - timedelta(seconds=skew_s)

    @classmethod
    def from_response(cls, payload: dict) -> TokenSet:
        expires_in = int(payload.get("expires_in", 3600))
        return cls(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            scope=payload.get("scope", ""),
            expires_at=(datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat(),
        )


class AtlassianOAuthError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------


class AtlassianOAuth:
    """Stateless helper around the Atlassian authorize + token endpoints.

    `cfg` is the `AtlassianCfg` block from nexus.yaml. `http_client` is injectable
    so the exchange/refresh calls can be exercised in tests without a network.
    """

    def __init__(self, cfg, *, http_client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def authorize_url(self, *, state: str, verifier: str) -> str:
        """The URL to send the user to for consent."""
        params = {
            "audience": "api.atlassian.com",
            "client_id": self.cfg.client_id,
            "scope": " ".join(self.cfg.scopes),
            "redirect_uri": self.cfg.redirect_uri,
            "state": state,
            "response_type": "code",
            "prompt": "consent",
            "code_challenge": code_challenge(verifier),
            "code_challenge_method": "S256",
        }
        return f"{self.cfg.authorize_url}?{urlencode(params)}"

    async def exchange_code(self, *, code: str, verifier: str) -> TokenSet:
        """Trade an authorization code for tokens."""
        return await self._token_request(
            {
                "grant_type": "authorization_code",
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "code": code,
                "redirect_uri": self.cfg.redirect_uri,
                "code_verifier": verifier,
            }
        )

    async def refresh(self, *, refresh_token: str) -> TokenSet:
        """Use a refresh token to get a fresh access token."""
        return await self._token_request(
            {
                "grant_type": "refresh_token",
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "refresh_token": refresh_token,
            }
        )

    async def _token_request(self, form: dict) -> TokenSet:
        try:
            resp = await self._client.post(self.cfg.token_url, data=form)
        except httpx.HTTPError as e:
            raise AtlassianOAuthError(f"token request failed: {e}") from e
        if resp.status_code != 200:
            raise AtlassianOAuthError(
                f"token endpoint returned {resp.status_code}: {resp.text[:300]}"
            )
        return TokenSet.from_response(resp.json())
