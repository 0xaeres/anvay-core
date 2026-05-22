"""Nexus configuration — loads nexus.yaml + env vars per ENGINEERING.md §15."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively substitute ${VAR} placeholders with environment values."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ----- nested models ---------------------------------------------------------


class ConnectorCfg(BaseModel):
    name: str
    type: str
    watch: bool = False
    # remaining fields are connector-specific; allow extras
    model_config = {"extra": "allow"}


class VectorCollectionsCfg(BaseModel):
    code: str = "nexus_code"
    text: str = "nexus_text"
    cache: str = "nexus_cache"


class VectorStoreCfg(BaseModel):
    url: str = "http://localhost:6333"
    collections: VectorCollectionsCfg = Field(default_factory=VectorCollectionsCfg)


class GraphCfg(BaseModel):
    url: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"


class ModelCfg(BaseModel):
    """Single LLM role config. Provider-specific extras allowed."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    url: str | None = None
    model_config = {"extra": "allow"}


class ModelsCfg(BaseModel):
    council_agents: ModelCfg
    synthesizer: ModelCfg
    adversary: ModelCfg
    pr_review: ModelCfg
    changelog: ModelCfg
    curator: ModelCfg
    light: ModelCfg
    embedding: ModelCfg
    reranker: ModelCfg
    # Optional dedicated tier for the Assistant agent loop. Falls back to
    # `council_agents` when unset, so existing nexus.yaml files keep working.
    assistant: ModelCfg | None = None


class LangfuseCfg(BaseModel):
    enabled: bool = True
    host: str = "http://localhost:3001"
    public_key: str = ""
    secret_key: str = ""


class ObservabilityCfg(BaseModel):
    langfuse: LangfuseCfg = Field(default_factory=LangfuseCfg)


class CacheCfg(BaseModel):
    semantic_threshold: float = 0.92
    ttl_hours: int = 24


class EnrichCfg(BaseModel):
    docs: bool = True
    code: bool = False


class ExtractRelationsCfg(BaseModel):
    docs: bool = True
    code: bool = False


class IngestionCfg(BaseModel):
    enrich_chunks: EnrichCfg = Field(default_factory=EnrichCfg)
    extract_relations: ExtractRelationsCfg = Field(default_factory=ExtractRelationsCfg)
    embed_batch_size: int = 32
    quality_gate_threshold: float = 0.3


class CircuitBreakerCfg(BaseModel):
    failure_threshold: int = 3
    recovery_timeout_s: int = 30


class RetrievalCfg(BaseModel):
    hyde_enabled: bool = True
    simple_query_threshold: float = 0.8
    circuit_breaker: CircuitBreakerCfg = Field(default_factory=CircuitBreakerCfg)


class AtlassianCfg(BaseModel):
    """Atlassian Rovo MCP Server connection + per-user OAuth — see
    docs/ASSISTANT-LAYER.md §5-6. Optional: when `enabled` is false the Assistant
    falls back to the stub connector."""

    enabled: bool = False
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8000/auth/atlassian/callback"
    mcp_url: str = "https://mcp.atlassian.com/v1/mcp"
    authorize_url: str = "https://auth.atlassian.com/authorize"
    token_url: str = "https://auth.atlassian.com/oauth/token"
    # UI page to return to after the OAuth callback completes.
    post_auth_redirect: str = "http://localhost:3000/settings/org"
    scopes: list[str] = Field(
        default_factory=lambda: [
            "read:jira-work",
            "write:jira-work",
            "read:confluence-content.all",
            "write:confluence-content",
            "offline_access",
        ]
    )


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    webhook_secret: str = ""


class StorageCfg(BaseModel):
    """Local SQLite paths. Defaults work for dev; mount /var/lib/nexus in prod."""

    proposal_queue: Path = Path("./data/proposals.db")
    council_checkpoint: Path = Path("./data/council.sqlite")


# ----- root config -----------------------------------------------------------


class NexusConfig(BaseSettings):
    """Root config. `NexusConfig.load(path)` reads YAML + env, returns instance."""

    model_config = SettingsConfigDict(extra="forbid")

    skills_repo: str
    org_skills_repo: str = ""
    hierarchy_root: Path = Path("./skills")
    org_library_root: Path = Path("./org-skills")

    connectors: list[ConnectorCfg] = Field(default_factory=list)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)
    graph: GraphCfg = Field(default_factory=GraphCfg)
    models: ModelsCfg
    observability: ObservabilityCfg = Field(default_factory=ObservabilityCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    ingestion: IngestionCfg = Field(default_factory=IngestionCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    atlassian: AtlassianCfg = Field(default_factory=AtlassianCfg)

    @classmethod
    def load(cls, path: str | Path = "nexus.yaml") -> NexusConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Config not found at {p}. Run `cp nexus.yaml.example nexus.yaml` and edit."
            )
        raw = yaml.safe_load(p.read_text())
        expanded = _expand_env(raw)
        return cls(**expanded)


@lru_cache(maxsize=1)
def get_config(path: str | Path = "nexus.yaml") -> NexusConfig:
    """Process-wide cached config accessor."""
    return NexusConfig.load(path)
