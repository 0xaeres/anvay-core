"""Minimal GitHub REST client for repo creation.

Scoped tight — we only need `POST /user/repos` and `POST /orgs/{org}/repos`.
Anything beyond that lives elsewhere (the GitHub connector, the webhook handler).
"""

from __future__ import annotations

import httpx

GITHUB_API = "https://api.github.com"


class GitHubAPIError(RuntimeError):
    """Raised on any non-2xx response from the GitHub REST API."""

    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


async def create_repo(
    *,
    token: str,
    name: str,
    org: str | None = None,
    private: bool = True,
    description: str = "Anvay skills repository — org-wide product skills + shared standards.",
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Create a GitHub repo. Returns the parsed repo object on success.

    Uses `auto_init=true` so the repo is immediately cloneable (it gets an
    initial README commit on the default branch).
    """
    url = f"{GITHUB_API}/orgs/{org}/repos" if org else f"{GITHUB_API}/user/repos"
    payload = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": True,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        resp = await client.post(url, json=payload, headers=headers)
    finally:
        if owns_client:
            await client.aclose()
    if resp.status_code >= 300:
        raise GitHubAPIError(resp.status_code, resp.text)
    return resp.json()
