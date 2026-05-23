"""Route-level tests for /setup/* endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from git import Repo

from nexus.api.app import app
from nexus.api.deps import get_setup_kv
from nexus.config import get_config
from nexus.setup import SetupKV


@pytest.fixture
def client(tmp_path: Path):
    kv = SetupKV(tmp_path / "registry.db")
    app.dependency_overrides[get_setup_kv] = lambda: kv
    # Force the resolver to ignore the developer's local nexus.yaml so each
    # test starts from a true "no setup yet" baseline.
    cfg = get_config()
    saved = cfg.skills_repo
    cfg.skills_repo = ""
    try:
        yield TestClient(app), kv
    finally:
        cfg.skills_repo = saved
        app.dependency_overrides.pop(get_setup_kv, None)


def _bare_remote(tmp_path: Path) -> str:
    bare = tmp_path / "remote.git"
    Repo.init(bare, bare=True, initial_branch="main")
    seed = tmp_path / "seed"
    work = Repo.clone_from(str(bare), str(seed))
    work.git.checkout("-b", "main")
    (seed / "README.md").write_text("# Skills\n")
    work.git.add(A=True)
    work.index.commit("initial")
    work.remotes.origin.push("main")
    return str(bare)


def test_status_reports_unconfigured_by_default(client) -> None:
    c, _ = client
    r = c.get("/setup/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is False
    assert body["skills_repo_url"] is None
    assert body["source"] is None


def test_status_reports_runtime_source_after_kv_set(client) -> None:
    c, kv = client
    kv.set("skills_repo", "https://github.com/me/x.git")
    body = c.get("/setup/status").json()
    assert body == {
        "configured": True,
        "skills_repo_url": "https://github.com/me/x.git",
        "source": "runtime",
    }


def test_post_existing_repo_persists_to_kv(client, tmp_path: Path) -> None:
    c, kv = client
    remote = _bare_remote(tmp_path)
    r = c.post(
        "/setup/skills-repo",
        json={"mode": "existing", "existing_repo_url": remote},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills_repo_url"] == remote
    assert body["files_seeded"] >= 13
    assert body["created_repo"] is False
    assert kv.get("skills_repo") == remote


def test_post_create_mode_without_token_returns_400(client, monkeypatch) -> None:
    c, _ = client
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = c.post("/setup/skills-repo", json={"mode": "create"})
    assert r.status_code == 400
    assert "GITHUB_TOKEN" in r.json()["detail"]


def test_post_create_mode_with_token_invokes_bootstrap(
    client, tmp_path: Path, monkeypatch
) -> None:
    c, _ = client
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    remote = _bare_remote(tmp_path)
    fake_repo = {"clone_url": remote}
    with patch(
        "nexus.setup.bootstrap.create_repo", new=AsyncMock(return_value=fake_repo)
    ):
        r = c.post(
            "/setup/skills-repo",
            json={"mode": "create", "github_org": "acme", "repo_name": "nexus-skills"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created_repo"] is True
    assert body["files_seeded"] >= 13


def test_post_bad_mode_returns_400(client) -> None:
    c, _ = client
    r = c.post("/setup/skills-repo", json={"mode": "nonsense"})
    assert r.status_code == 422  # Pydantic Literal validation
