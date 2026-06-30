"""FastAPI dependency providers."""

from __future__ import annotations

import logging
from functools import lru_cache

from anvay.auth.store import AuthStore
from anvay.config import AnvayConfig, get_config
from anvay.council.queue import ProposalQueue
from anvay.registry import Registry
from anvay.skills.store import SkillStore

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_proposal_queue() -> ProposalQueue:
    config: AnvayConfig = get_config()
    return ProposalQueue(config.storage.proposal_queue)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    config: AnvayConfig = get_config()
    return Registry(config.storage.proposal_queue.parent / "registry.db")


@lru_cache(maxsize=1)
def get_auth_store() -> AuthStore:
    config: AnvayConfig = get_config()
    return AuthStore(config.storage.proposal_queue.parent / "registry.db")


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
