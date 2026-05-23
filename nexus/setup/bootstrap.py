"""Skills-repo bootstrap orchestrator.

Either creates a new GitHub repo or attaches to an existing one, clones it to a
temp working dir, copies the bundled starter pack into `shared/`, commits, and
pushes. Idempotent on the seed step — if `shared/` already has files in the
target repo we don't overwrite.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from nexus.setup.github_api import create_repo

log = logging.getLogger(__name__)


def starter_pack_root() -> Path:
    """Absolute path to the bundled `starter/shared/` directory."""
    return Path(__file__).resolve().parent.parent / "skills" / "starter" / "shared"


class BootstrapError(RuntimeError):
    """Raised when the bootstrap flow cannot complete."""


@dataclass
class BootstrapResult:
    skills_repo_url: str
    files_seeded: int
    commit_sha: str | None
    created_repo: bool


async def bootstrap_skills_repo(
    *,
    mode: str,
    github_token: str | None = None,
    github_org: str | None = None,
    repo_name: str = "nexus-skills",
    existing_repo_url: str | None = None,
) -> BootstrapResult:
    """Run the skills_repo bootstrap.

    Args:
        mode: "create" to mint a new repo via GitHub API; "existing" to attach to
              a repo the user already owns.
        github_token: PAT with `repo` scope. Required for `mode="create"` and
                      to push to an existing private repo.
        github_org: When provided, the new repo is created under this org;
                    otherwise under the authenticated user.
        repo_name: Name of the repo to create (mode="create" only).
        existing_repo_url: Clone URL (HTTPS or SSH) of an existing repo
                           (mode="existing" only).
    """
    if mode not in {"create", "existing"}:
        raise BootstrapError(f"unknown mode: {mode!r}")

    created_repo = False
    if mode == "create":
        if not github_token:
            raise BootstrapError("github_token is required when mode='create'")
        repo_obj = await create_repo(
            token=github_token, name=repo_name, org=github_org
        )
        clone_url = _authenticated_clone_url(repo_obj["clone_url"], github_token)
        canonical_url = repo_obj["clone_url"]
        created_repo = True
    else:
        if not existing_repo_url:
            raise BootstrapError(
                "existing_repo_url is required when mode='existing'"
            )
        clone_url = _authenticated_clone_url(existing_repo_url, github_token)
        canonical_url = existing_repo_url

    starter = starter_pack_root()
    if not starter.is_dir():
        raise BootstrapError(f"starter pack not found at {starter}")

    with tempfile.TemporaryDirectory(prefix="nexus-bootstrap-") as tmp:
        workdir = Path(tmp) / "skills"
        try:
            repo = Repo.clone_from(clone_url, str(workdir))
        except Exception as e:
            raise BootstrapError(f"clone failed: {e}") from e

        shared_dir = workdir / "shared"
        files_seeded = _copy_starter_pack_into(starter, shared_dir)

        if files_seeded == 0:
            log.info("bootstrap: shared/ already populated, nothing to seed")
            return BootstrapResult(
                skills_repo_url=canonical_url,
                files_seeded=0,
                commit_sha=None,
                created_repo=created_repo,
            )

        repo.git.add(A=True)
        if not repo.is_dirty(untracked_files=True):
            return BootstrapResult(
                skills_repo_url=canonical_url,
                files_seeded=files_seeded,
                commit_sha=None,
                created_repo=created_repo,
            )
        commit = repo.index.commit(
            f"bootstrap: seed starter pack ({files_seeded} skills)"
        )
        try:
            repo.remotes.origin.push()
        except Exception as e:
            raise BootstrapError(f"push failed: {e}") from e

        return BootstrapResult(
            skills_repo_url=canonical_url,
            files_seeded=files_seeded,
            commit_sha=commit.hexsha,
            created_repo=created_repo,
        )


def _copy_starter_pack_into(starter: Path, dest_shared: Path) -> int:
    """Copy starter pack files that aren't already present. Returns count copied."""
    copied = 0
    for src in sorted(starter.rglob("*.skill.md")):
        rel = src.relative_to(starter)
        target = dest_shared / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied += 1
    return copied


def _authenticated_clone_url(url: str, token: str | None) -> str:
    """Inline a PAT into the HTTPS clone URL so push doesn't prompt for creds.

    SSH URLs and missing tokens are passed through untouched — caller is
    expected to have SSH keys set up in that case.
    """
    if not token:
        return url
    if not url.startswith("https://"):
        return url
    return url.replace("https://", f"https://x-access-token:{token}@", 1)
