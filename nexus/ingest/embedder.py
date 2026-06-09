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
from openai import APIStatusError, AsyncOpenAI, OpenAIError

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
        """
        Initialize the EmbedderClient with connection, model, and runtime configuration.
        
        Parameters:
            base_url (str): Base URL of the embedding service (trailing '/' will be removed).
            provider (str): Provider name; normalized to lowercase. If "jina-local" and the URL does not already end with "/v1", "/v1" will be appended for SDK usage.
            model (str): Model identifier used for embedding requests.
            api_key (str | None): API key for the SDK client; if omitted, the SDK is constructed with the sentinel value "unused".
            instruction_profile (str | None): Instruction profile name normalized to lowercase (empty string if None).
            timeout_s (float): Request timeout in seconds for underlying HTTP clients.
            batch_size (int): Internal batching size used when sending embedding requests.
        
        Notes:
            - Sets instance attributes: `provider`, `model`, `base_url`, `batch_size`, and `instruction_profile`.
            - Constructs an httpx.AsyncClient stored as `_http_client`, an AsyncOpenAI SDK client stored as `_client` (with `max_retries=0` to disable SDK-level retries), and a separate httpx.AsyncClient `_health_client` for health checks.
        """
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.instruction_profile = (instruction_profile or "").lower()
        sdk_base_url = self.base_url
        if self.provider == "jina-local" and not sdk_base_url.rstrip("/").endswith("/v1"):
            sdk_base_url = f"{sdk_base_url}/v1"
        self._http_client = httpx.AsyncClient(timeout=timeout_s)
        self._client = AsyncOpenAI(
            api_key=api_key or "unused",
            base_url=sdk_base_url,
            timeout=timeout_s,
            max_retries=0,
            http_client=self._http_client,
        )
        self._health_client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    @classmethod
    def from_cfg(cls, cfg: ModelCfg, *, batch_size: int = 32) -> EmbedderClient:
        """
        Constructs an EmbedderClient configured from a ModelCfg.
        
        If the config does not provide a base URL, selects a sensible default: "https://api.deepinfra.com/v1/openai" when the provider is "deepinfra", otherwise "http://localhost:8080". The provider name is normalized to lowercase and the model, api_key, instruction_profile, and batch_size from the config are applied to the returned client.
        
        Parameters:
            cfg (ModelCfg): Source configuration containing provider, base_url or url, model, api_key, and instruction_profile.
            batch_size (int): Batch size to use for internal embedding requests.
        
        Returns:
            EmbedderClient: A client instance configured according to `cfg` and `batch_size`.
        """
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
        """
        Close the embedder client's underlying network clients and release associated resources.
        
        Closes the OpenAI-compatible SDK client used for embeddings and the separate HTTP client used for health checks.
        """
        await self._client.close()
        await self._health_client.aclose()

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
        """
        Request embeddings for the given input strings and return one embedding per input, preserving the original input order.
        
        Parameters:
            inputs (list[str]): Strings to embed.
        
        Returns:
            list[list[float]]: A list of embedding vectors corresponding to `inputs` in the same order.
        
        Raises:
            EmbedderError: If the embedder returns a non-retryable error (including client/request-shape errors) or if all retry attempts fail.
        """
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                resp = await self._client.embeddings.create(
                    input=inputs,
                    model=self.model,
                )
            except APIStatusError as e:
                msg = f"embedder returned {e.status_code}: {str(e)[:300]}"
                if e.status_code < 500 or _is_nonretryable_server_error(msg):
                    raise EmbedderError(msg) from e
                last_exc = EmbedderError(msg)
            except OpenAIError as e:
                last_exc = EmbedderError(f"embedder request failed: {e}")
            else:
                ordered = sorted(resp.data, key=lambda d: d.index)
                return [list(d.embedding) for d in ordered]

            if delay is not None:
                log.warning("embedder attempt %d failed; retrying in %.0fs", attempt + 1, delay)
                await asyncio.sleep(delay)

        raise last_exc or EmbedderError("embedder failed after retries")

    def _prefix(self, vector: VectorName, modality: Modality) -> str:
        """
        Selects the instruction prefix to prepend to inputs based on the configured instruction profile, vector, and modality.
        
        Parameters:
            vector (VectorName): Which embedding vector to target (e.g., "dense_code" or "dense_text").
            modality (Modality): The embedding modality ("passage" or "query").
        
        Returns:
            prefix (str): The instruction text to prepend to each input for the current instruction profile; empty string if no prefix applies.
        """
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
        """
        Checks whether the configured embedder service is healthy by querying its /health endpoint.
        
        Performs an HTTP GET to /health and returns success status observed from the response. Network or HTTP client errors are treated as an unhealthy result.
        
        Returns:
            `true` if the service responds with HTTP 200, `false` otherwise.
        """
        try:
            r = await self._health_client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False
