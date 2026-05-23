"""Tests for the skills-repo bootstrap flow.

Covers the SetupKV store, the starter-pack file enumeration, the GitHub client
(mocked), and the bootstrap orchestrator end-to-end against a local bare repo
acting as the "GitHub remote."
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from git import Repo

from nexus.setup import SetupKV, bootstrap_skills_repo, starter_pack_root
from nexus.setup.github_api import GitHubAPIError, create_repo

# ---------- SetupKV ---------------------------------------------------------


def test_setup_kv_set_get_delete(tmp_path: Path) -> None:
    kv = SetupKV(tmp_path / "data.db")
    assert kv.get("skills_repo") is None
    kv.set("skills_repo", "https://github.com/me/x.git")
    assert kv.get("skills_repo") == "https://github.com/me/x.git"
    kv.set("skills_repo", "https://github.com/me/y.git")  # upsert
    assert kv.get("skills_repo") == "https://github.com/me/y.git"
    kv.delete("skills_repo")
    assert kv.get("skills_repo") is None


def test_setup_kv_creates_db_parent_dir(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "deeper" / "data.db"
    SetupKV(db)
    assert db.parent.is_dir()


# ---------- starter pack discovery ------------------------------------------


def test_starter_pack_root_exists() -> None:
    root = starter_pack_root()
    assert root.is_dir()
    files = list(root.rglob("*.skill.md"))
    assert len(files) >= 13  # 6 languages + 6 tech_stack + 1 security


def test_starter_pack_has_expected_kinds() -> None:
    root = starter_pack_root()
    kinds = {p.parent.name for p in root.rglob("*.skill.md")}
    assert {"language", "tech_stack", "security"}.issubset(kinds)


# ---------- GitHub API client -----------------------------------------------


def test_create_repo_user_endpoint() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = request.read().decode()
        return httpx.Response(
            201, json={"clone_url": "https://github.com/u/nexus-skills.git"}
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    repo = asyncio.run(
        create_repo(token="tok", name="nexus-skills", client=client)
    )
    assert repo["clone_url"] == "https://github.com/u/nexus-skills.git"
    assert captured["url"].endswith("/user/repos")
    assert captured["headers"]["authorization"] == "Bearer tok"
    import json as _json
    assert _json.loads(captured["json"])["auto_init"] is True


def test_create_repo_org_endpoint() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(201, json={"clone_url": "https://x.git"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    asyncio.run(create_repo(token="t", name="nx", org="acme", client=client))
    assert "/orgs/acme/repos" in captured["url"]


def test_create_repo_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text='{"message":"name already exists"}')

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    with pytest.raises(GitHubAPIError) as exc:
        asyncio.run(create_repo(token="t", name="x", client=client))
    assert exc.value.status == 422


# ---------- bootstrap orchestrator end-to-end -------------------------------


def _init_local_remote(path: Path) -> str:
    """Create a bare local repo to act as the "GitHub remote", pre-populated
    with a single commit on `main` so it's cloneable like an auto_init repo."""
    bare = path / "remote.git"
    Repo.init(bare, bare=True, initial_branch="main")
    # Seed with an initial commit via an intermediate working clone.
    seed = path / "seed"
    work = Repo.clone_from(str(bare), str(seed))
    work.git.checkout("-b", "main")
    (seed / "README.md").write_text("# Skills\n")
    work.git.add(A=True)
    work.index.commit("initial")
    work.remotes.origin.push("main")
    return str(bare)


def test_bootstrap_existing_repo_seeds_starter_pack(tmp_path: Path) -> None:
    remote_url = _init_local_remote(tmp_path)

    result = asyncio.run(
        bootstrap_skills_repo(mode="existing", existing_repo_url=remote_url)
    )
    assert result.created_repo is False
    assert result.files_seeded >= 13
    assert result.commit_sha is not None
    assert result.skills_repo_url == remote_url

    # Verify the remote actually received the commit + files.
    verify_dir = tmp_path / "verify"
    verify = Repo.clone_from(remote_url, str(verify_dir))
    shared = Path(verify.working_tree_dir) / "shared"
    assert shared.is_dir()
    seeded = list(shared.rglob("*.skill.md"))
    assert len(seeded) == result.files_seeded


def test_bootstrap_existing_repo_idempotent_on_second_run(tmp_path: Path) -> None:
    remote_url = _init_local_remote(tmp_path)
    first = asyncio.run(
        bootstrap_skills_repo(mode="existing", existing_repo_url=remote_url)
    )
    second = asyncio.run(
        bootstrap_skills_repo(mode="existing", existing_repo_url=remote_url)
    )
    assert first.files_seeded > 0
    assert second.files_seeded == 0  # nothing new to copy
    assert second.commit_sha is None


def test_bootstrap_create_mode_calls_github_then_seeds(tmp_path: Path) -> None:
    remote_url = _init_local_remote(tmp_path)
    fake_repo = {"clone_url": remote_url}

    with patch(
        "nexus.setup.bootstrap.create_repo", new=AsyncMock(return_value=fake_repo)
    ) as mock_create:
        result = asyncio.run(
            bootstrap_skills_repo(
                mode="create",
                github_token="tok",
                github_org="acme",
                repo_name="nexus-skills",
            )
        )
    mock_create.assert_awaited_once()
    kwargs = mock_create.await_args.kwargs
    assert kwargs["org"] == "acme"
    assert kwargs["name"] == "nexus-skills"
    assert result.created_repo is True
    assert result.files_seeded >= 13


def test_bootstrap_create_requires_token() -> None:
    with pytest.raises(Exception) as exc:
        asyncio.run(bootstrap_skills_repo(mode="create"))
    assert "github_token" in str(exc.value)


def test_bootstrap_existing_requires_url() -> None:
    with pytest.raises(Exception) as exc:
        asyncio.run(bootstrap_skills_repo(mode="existing"))
    assert "existing_repo_url" in str(exc.value)


def test_bootstrap_unknown_mode_rejected() -> None:
    with pytest.raises(Exception) as exc:
        asyncio.run(bootstrap_skills_repo(mode="garbage"))
    assert "unknown mode" in str(exc.value)
