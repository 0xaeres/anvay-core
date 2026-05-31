from __future__ import annotations

import pytest
from qdrant_client.http import models as qm

from nexus.ingest.indexer import Indexer, IndexerError


def test_indexer_builds_qdrant_turboquant_config() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_enabled=True,
        quantization_type="turboquant",
        quantization_bits="bits2",
        quantization_always_ram=False,
    )

    config = indexer._dense_quantization_config()

    assert isinstance(config, qm.TurboQuantization)
    assert config.turbo.bits == qm.TurboQuantBitSize.BITS2
    assert config.turbo.always_ram is False


def test_indexer_can_disable_quantization() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_enabled=False,
    )

    assert indexer._dense_quantization_config() is None


def test_indexer_rejects_unknown_quantization_bits() -> None:
    indexer = Indexer(
        url="http://127.0.0.1:6333",
        quantization_bits="bits3",
    )

    with pytest.raises(IndexerError, match=r"unsupported vector_store\.quantization\.bits"):
        indexer._dense_quantization_config()
