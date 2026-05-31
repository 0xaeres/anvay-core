"""Embedder client — OpenAI-compatible embeddings, local Jina or cloud.

Jina v4 separates **task** (retrieval / classification / clustering / …) and
**modality** (text / code). We use two modes, mapped to Qdrant named vectors:

| modality | task           | Qdrant vector |
|----------|----------------|---------------|
| code     | retrieval      | dense_code    |
| text     | retrieval      | dense_text    |

llama-server with `--embedding` does not natively switch LoRA adapters per
request, so we apply Jina's documented instruction prefix to the input string.
This is consistent with how transformers-jina v4 internally formats inputs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import httpx

from nexus.config import ModelCfg
from nexus.ingest.models import Chunk, EmbeddedChunk

log = logging.getLogger(__name__)

VectorName = Literal["dense_code", "dense_text"]
Modality = Literal["passage", "query"]

# Jina v4 instruction prefixes (per the model card). Keep in sync with the served GGUF.
_JINA_V4_PREFIXES: dict[tuple[VectorName, Modality], str] = {
    ("dense_code", "passage"): "Represent the code for retrieval: ",
    ("dense_code", "query"): "Represent the question for retrieving relevant code: ",
    ("dense_text", "passage"): "Represent the document for retrieval: ",
    ("dense_text", "query"): "Represent the question for retrieving relevant documents: ",
}
_QWEN3_QUERY_INSTRUCTIONS: dict[VectorName, str] = {
    "dense_code": "Given a developer search query, retrieve relevant code passages that answer the query",
    "dense_text": "Given a documentation search query, retrieve relevant passages that answer the query",
}


class EmbedderError(RuntimeError):
    pass


_RETRY_DELAYS = (1.0, 3.0, 8.0)  # seconds between attempts (3 total)


def _is_nonretryable_server_error(message: str) -> bool:
    """llama.cpp reports some request-shape errors as HTTP 500."""
    lower = message.lower()
    return (
        "input" in lower
        and "too large" in lower
        and ("physical batch size" in lower or "batch size" in lower)
    )


class EmbedderClient:
    """Thin async client. Construct once, reuse across the ingestion pipeline."""

    def __init__(
        self,
        base_url: str,
        *,
        provider: str = "jina-local",
        model: str = "jinaai/jina-embeddings-v4",
        api_key: str | None = None,
        instruction_profile: str | None = "jina-v4",
        timeout_s: float = 120.0,
        batch_size: int = 32,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.instruction_profile = (instruction_profile or "").lower()
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout_s)

    @classmethod
    def from_cfg(cls, cfg: ModelCfg, *, batch_size: int = 32) -> EmbedderClient:
        provider = cfg.provider.lower()
        base_url = cfg.base_url or cfg.url
        if not base_url:
            base_url = (
                "https://api.deepinfra.com/v1/openai"
                if provider == "deepinfra"
                else "http://localhost:8080"
            )
        return cls(
            base_url=base_url,
            provider=provider,
            model=cfg.model,
            api_key=cfg.api_key,
            instruction_profile=cfg.instruction_profile,
            batch_size=batch_size,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ core

    async def embed(
        self, texts: list[str], *, vector: VectorName, modality: Modality = "passage"
    ) -> list[list[float]]:
        """Return one vector per input string. Batched internally."""
        if not texts:
            return []
        prefix = self._prefix(vector, modality)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[i : i + self.batch_size]]
            out.extend(await self._call(batch))
        return out

    async def _call(self, inputs: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                resp = await self._client.post(
                    self._embeddings_path(),
                    json={"input": inputs, "model": self.model},
                )
            except httpx.HTTPError as e:
                last_exc = EmbedderError(f"embedder request failed: {e}")
            else:
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    ordered = sorted(data, key=lambda d: d.get("index", 0))
                    return [d["embedding"] for d in ordered]
                msg = f"embedder returned {resp.status_code}: {resp.text[:300]}"
                # 4xx = bad request (e.g. token limit) — don't retry
                if resp.status_code < 500 or _is_nonretryable_server_error(msg):
                    raise EmbedderError(msg)
                last_exc = EmbedderError(msg)

            if delay is not None:
                log.warning("embedder attempt %d failed; retrying in %.0fs", attempt + 1, delay)
                await asyncio.sleep(delay)

        raise last_exc or EmbedderError("embedder failed after retries")

    def _embeddings_path(self) -> str:
        if self.provider == "jina-local":
            return "/v1/embeddings"
        return "/embeddings"

    def _prefix(self, vector: VectorName, modality: Modality) -> str:
        if self.instruction_profile == "jina-v4":
            return _JINA_V4_PREFIXES[(vector, modality)]
        if self.instruction_profile == "qwen3" and modality == "query":
            return f"Instruct: {_QWEN3_QUERY_INSTRUCTIONS[vector]}\nQuery: "
        return ""

    # ------------------------------------------------------------------ helpers

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Compute the right named-vector for each chunk based on its kind."""
        code = [c for c in chunks if c.kind.value == "code"]
        docs = [c for c in chunks if c.kind.value == "doc"]

        code_vecs, doc_vecs = await asyncio.gather(
            self.embed([c.text_for_embedding() for c in code], vector="dense_code"),
            self.embed([c.text_for_embedding() for c in docs], vector="dense_text"),
        )
        result: list[EmbeddedChunk] = []
        for c, v in zip(code, code_vecs, strict=True):
            result.append(EmbeddedChunk(chunk=c, vector=v, vector_name="dense_code"))
        for c, v in zip(docs, doc_vecs, strict=True):
            result.append(EmbeddedChunk(chunk=c, vector=v, vector_name="dense_text"))
        return result

    async def embed_query(self, text: str, *, vector: VectorName) -> list[float]:
        vecs = await self.embed([text], vector=vector, modality="query")
        return vecs[0]

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False
