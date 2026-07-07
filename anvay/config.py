"""Anvay configuration — loads anvay.yaml + env vars per ENGINEERING.md §15."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
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
    code: str = "anvay_code"
    text: str = "anvay_text"


class VectorQuantizationCfg(BaseModel):
    """Qdrant dense-vector quantization.

    TurboQuant is available in Qdrant v1.18+. It is applied when collections are
    created; changing it for an existing collection requires a recreate/reindex
    or an explicit Qdrant collection update outside Anvay.
    """

    enabled: bool = True
    type: str = "turboquant"
    bits: str = "bits4"
    always_ram: bool = True


class VectorStoreCfg(BaseModel):
    url: str = "http://localhost:6333"
    timeout_s: int = 120
    upsert_batch_size: int = 16
    collections: VectorCollectionsCfg = Field(default_factory=VectorCollectionsCfg)
    quantization: VectorQuantizationCfg = Field(default_factory=VectorQuantizationCfg)


class GraphStoreCfg(BaseModel):
    """Required derived product-system graph store."""

    host: str = Field("localhost", min_length=1)
    port: int = Field(6379, gt=0, le=65535)
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    graph_prefix: str = Field("anvay", min_length=1)
    max_connections: int = Field(16, gt=0)
    timeout_ms: int = Field(5_000, gt=0)


class ModelCfg(BaseModel):
    """Single LLM role config. Provider-specific extras allowed."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    url: str | None = None
    dim: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    instruction_profile: str | None = None
    model_config = {"extra": "allow"}


class ModelsCfg(BaseModel):
    council: ModelCfg          # default for planner + evaluator + repair
    planner: ModelCfg | None = None
    evaluator: ModelCfg | None = None
    repair: ModelCfg | None = None
    synthesizer: ModelCfg | None = None
    chat_agent: ModelCfg | None = None
    light: ModelCfg            # enricher (HQE + doc context)
    embedding: ModelCfg
    reranker: ModelCfg

    @model_validator(mode="before")
    @classmethod
    def map_legacy_role_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        aliases = {
            "drafter": "planner",
            "critic": "evaluator",
            "reviser": "repair",
        }
        normalized = dict(data)
        for old, new in aliases.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized[old]
            normalized.pop(old, None)
        return normalized


class EnrichCfg(BaseModel):
    docs: bool = False  # Optional doc contextual retrieval; off by default for fast ingest
    code: bool = False  # HQE: optional, expensive code-question generation


class EnrichmentWorkerCfg(BaseModel):
    enabled: bool = False
    poll_interval_s: float = 5.0
    max_attempts: int = 3


class GraphIngestionCfg(BaseModel):
    mode: Literal["deterministic", "bounded_llm"] = "bounded_llm"
    max_resources_per_batch: int = Field(12, gt=0)
    max_facts_per_resource: int = Field(24, gt=0)
    concurrency: int = Field(2, gt=0)
    confidence_floor: float = Field(0.65, ge=0.0, le=1.0)


class OrphanSweepCfg(BaseModel):
    """Reverse reconciliation: delete Qdrant points no manifest row claims.

    Off by default, and dry-run by default when enabled. The sweep only ever
    considers raw resource chunks (artifact_type code/doc); skill chunks and
    synthetic summaries are always exempt."""

    enabled: bool = False
    dry_run: bool = True
    grace_minutes: int = 60


class IngestionCfg(BaseModel):
    enrich_chunks: EnrichCfg = Field(default_factory=EnrichCfg)
    graph: GraphIngestionCfg = Field(default_factory=GraphIngestionCfg)
    orphan_sweep: OrphanSweepCfg = Field(default_factory=OrphanSweepCfg)
    embed_batch_size: int = 32
    quality_gate_threshold: float = 0.0
    file_batch_size: int = 50
    read_concurrency: int = 10
    batch_concurrency: int = 2
    enricher_concurrency: int = 4       # cloud inference — rate-limited, not RAM-limited
    enrichment_worker: EnrichmentWorkerCfg = Field(default_factory=EnrichmentWorkerCfg)


class RetrievalCfg(BaseModel):
    """Evidence-retrieval knobs for interactive (MCP) callers.

    ``interactive_budget_ms`` is a soft deadline applied only to interactive
    evidence retrieval: the core fan-out always runs, but the optional long-tail
    enrichment (DRIFT-lite follow-ups, coverage repair) is skipped once spent, so
    the MCP path returns best-effort evidence instead of blocking. Council/eval
    callers leave it unset (unbounded). ``None`` disables the deadline.
    """

    interactive_budget_ms: float | None = 8000.0


class CouncilCfg(BaseModel):
    """Skill-generation council knobs.

    `faithfulness_gate` re-enables the bounded LLM entailment judge in the eval
    node. Off by default: the council runs deterministic-only eval (no judge,
    no LLM eval-repair), keeping skill generation to a single synthesis call.
    """

    faithfulness_gate: bool = False


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class StorageCfg(BaseModel):
    """Local SQLite paths. Defaults work for dev; mount /var/lib/anvay in prod."""

    proposal_queue: Path = Path("./data/proposals.db")
    council_checkpoint: Path = Path("./data/council.sqlite")


# ----- root config -----------------------------------------------------------


class AnvayConfig(BaseSettings):
    """Root config. `AnvayConfig.load(path)` reads YAML + env, returns instance."""

    model_config = SettingsConfigDict(extra="forbid")

    skills_repo: str = ""  # set via anvay.yaml OR runtime via /setup/skills-repo
    hierarchy_root: Path = Path("./skills")

    connectors: list[ConnectorCfg] = Field(default_factory=list)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)
    graph_store: GraphStoreCfg = Field(default_factory=GraphStoreCfg)
    models: ModelsCfg
    ingestion: IngestionCfg = Field(default_factory=IngestionCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    council: CouncilCfg = Field(default_factory=CouncilCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)

    @classmethod
    def load(cls, path: str | Path = "anvay.yaml") -> AnvayConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Config not found at {p}. Run `cp anvay.yaml.example anvay.yaml` and edit."
            )
        raw = yaml.safe_load(p.read_text())
        expanded = _expand_env(raw)
        return cls(**expanded)


@lru_cache(maxsize=1)
def get_config(path: str | Path = "anvay.yaml") -> AnvayConfig:
    """Process-wide cached config accessor."""
    return AnvayConfig.load(path)
