"""FastAPI dependency providers."""

from __future__ import annotations

import logging
from functools import lru_cache

from anvay.auth.store import AuthStore
from anvay.config import AnvayConfig, get_config
from anvay.council.queue import ProposalQueue
from anvay.registry import Registry
from anvay.setup import SetupKV
from anvay.skills.store import SkillStore

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_proposal_queue() -> ProposalQueue:
    config: AnvayConfig = get_config()
    return ProposalQueue(config.storage.proposal_queue)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    config: AnvayConfig = get_config()
    # Co-locate the registry alongside the proposal queue
    return Registry(config.storage.proposal_queue.parent / "registry.db")


@lru_cache(maxsize=1)
def get_setup_kv() -> SetupKV:
    config: AnvayConfig = get_config()
    return SetupKV(config.storage.proposal_queue.parent / "registry.db")


@lru_cache(maxsize=1)
def get_auth_store() -> AuthStore:
    config: AnvayConfig = get_config()
    return AuthStore(config.storage.proposal_queue.parent / "registry.db")


def resolve_skills_repo_url(
    config: AnvayConfig | None = None, kv: SetupKV | None = None
) -> str:
    """Return the active skills_repo URL or '' if setup is still required.

    Resolution order: runtime KV (set by /setup/skills-repo) > anvay.yaml.
    """
    cfg = config or get_config()
    store = kv or get_setup_kv()
    return store.get("skills_repo") or cfg.skills_repo or ""


@lru_cache(maxsize=1)
def get_skill_store() -> SkillStore:
    from pathlib import Path

    config: AnvayConfig = get_config()
    root = Path(config.hierarchy_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    return SkillStore(root)


def get_config_dep() -> AnvayConfig:
    return get_config()
