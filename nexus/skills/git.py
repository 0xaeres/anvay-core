"""Git operations for the skills hierarchy.

Best-effort: if the hierarchy_root is not a git repo (e.g. seed directory), the
add/commit/push are no-ops and we log a warning. Real prod path uses an
ephemeral clone of the skills_repo configured in nexus.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from git import Repo  # type: ignore[import-not-found]
    from git.exc import GitError, InvalidGitRepositoryError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    Repo = None  # type: ignore[assignment]
    GitError = Exception  # type: ignore[misc,assignment]
    InvalidGitRepositoryError = Exception  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)


def commit_and_push(root: Path, message: str, *, push: bool = True) -> bool:
    """Stage everything in `root`, commit, and (optionally) push origin/main.

    Returns True if a commit was created. False on any reasonable skip
    (not a repo, nothing to commit, remote missing, etc.).
    """
    if Repo is None:
        log.warning("gitpython unavailable; skipping commit")
        return False
    try:
        repo = Repo(root, search_parent_directories=False)
    except InvalidGitRepositoryError:
        log.info("skills root %s is not a git repo; skipping commit", root)
        return False
    except GitError as e:  # pragma: no cover
        log.warning("git open failed: %s", e)
        return False

    repo.git.add(A=True)
    if not repo.is_dirty(untracked_files=True):
        log.info("no changes to commit")
        return False
    repo.index.commit(message)

    if push and repo.remotes:
        try:
            origin = repo.remotes.origin
            infos = origin.push()
            if any(info.flags & info.ERROR for info in infos):
                log.warning("git push failed: %s", "; ".join(info.summary for info in infos))
                return False
        except GitError as e:
            log.warning("git push failed (commit retained locally): %s", e)
            return False
    return True
