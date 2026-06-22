from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexus.config import GraphIngestionCfg, GraphStoreCfg


def test_graph_store_config_rejects_empty_host() -> None:
    with pytest.raises(ValidationError):
        GraphStoreCfg(host="")


def test_graph_store_config_rejects_bad_port() -> None:
    with pytest.raises(ValidationError):
        GraphStoreCfg(port=0)


def test_graph_ingestion_config_defaults_to_bounded_llm() -> None:
    cfg = GraphIngestionCfg()

    assert cfg.mode == "bounded_llm"
    assert cfg.max_facts_per_resource == 24


def test_graph_ingestion_config_rejects_bad_confidence_floor() -> None:
    with pytest.raises(ValidationError):
        GraphIngestionCfg(confidence_floor=1.5)
