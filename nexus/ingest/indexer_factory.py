"""Qdrant indexer factory."""

from __future__ import annotations

from nexus.config import NexusConfig
from nexus.ingest.indexer import Indexer


def create_indexer(config: NexusConfig) -> Indexer:
    return Indexer(
        url=config.vector_store.url,
        code_collection=config.vector_store.collections.code,
        text_collection=config.vector_store.collections.text,
        vector_dim=config.models.embedding.dim or 2048,
        timeout_s=config.vector_store.timeout_s,
        upsert_batch_size=config.vector_store.upsert_batch_size,
        quantization_type=config.vector_store.quantization.type,
        quantization_enabled=config.vector_store.quantization.enabled,
        quantization_bits=config.vector_store.quantization.bits,
        quantization_always_ram=config.vector_store.quantization.always_ram,
    )


__all__ = ["create_indexer"]
