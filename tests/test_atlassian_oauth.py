"""Atlassian OAuth — PKCE helpers + the code-exchange / refresh flow."""

from __future__ import annotations

import asyncio

import httpx

from nexus.auth.atlassian_oauth import (
    AtlassianOAuth,
    code_challenge,
    generate_code_verifier,
    generate_state,
)
from nexus.config import AtlassianCfg


def test_pkce_verifier_is_high_entropy_and_urlsafe() -> None:
    v = generate_code_verifier()
    assert 43 <= len(v) <= 128
    assert "=" not in v  # base64url, unpadded
    assert generate_code_verifier() != generate_code_verifier()


def test_code_challenge_is_deterministic_s256() -> None:
    v = generate_code_verifier()
    assert code_challenge(v) == code_challenge(v)
    assert "=" not in code_challenge(v)
    assert code_challenge(v) != code_challenge(generate_code_verifier())


def test_state_is_unique() -> None:
    assert generate_state() != generate_state()


def test_authorize_url_carries_pkce_params() -> None:
    cfg = AtlassianCfg(client_id="cid-123", redirect_uri="http://cb")
    oauth = AtlassianOAuth(cfg)
    verifier = generate_code_verifier()
    url = oauth.authorize_url(state="st8", verifier=verifier)
    assert "client_id=cid-123" in url
    assert "state=st8" in url
    assert "code_challenge_method=S256" in url
    assert code_challenge(verifier) in url
    assert "response_type=code" in url


def test_exchange_code_sends_correct_form_and_parses_tokens() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "access_token": "AT",
                "refresh_token": "RT",
                "expires_in": 3600,
                "scope": "read:jira-work",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    oauth = AtlassianOAuth(
        AtlassianCfg(client_id="cid", client_secret="sec"), http_client=client
    )
    token = asyncio.run(oauth.exchange_code(code="auth-code", verifier="my-verifier"))

    assert token.access_token == "AT"
    assert token.refresh_token == "RT"
    assert not token.is_expired()
    assert "grant_type=authorization_code" in captured["body"]
    assert "code_verifier=my-verifier" in captured["body"]
    assert "code=auth-code" in captured["body"]


def test_refresh_uses_refresh_grant() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200, json={"access_token": "AT2", "refresh_token": "RT2", "expires_in": 3600}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    oauth = AtlassianOAuth(AtlassianCfg(client_id="cid"), http_client=client)
    token = asyncio.run(oauth.refresh(refresh_token="old-rt"))

    assert token.access_token == "AT2"
    assert "grant_type=refresh_token" in captured["body"]
    assert "refresh_token=old-rt" in captured["body"]
