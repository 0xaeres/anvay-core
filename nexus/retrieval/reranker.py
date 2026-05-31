"""Reranker client for local Jina and DeepInfra rerankers.

llama-server `--reranking` exposes:
  POST /reranking  { "query": "...", "documents": ["...", ...] }
returning an `{"results": [{"index": i, "relevance_score": s}, ...]}` shape
similar to Cohere's rerank API. DeepInfra Qwen rerankers expose:
  POST /v1/inference/{model}  { "queries": ["..."], "documents": ["...", ...] }
returning `{"scores": [...]}` in input document order.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from nexus.config import ModelCfg


@dataclass(frozen=True)
class RerankResult:
    index: int  # index into the original `documents` list
    score: float


class RerankerError(RuntimeError):
    pass


class RerankerClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        *,
        provider: str = "jina-local",
        model: str = "jinaai/jina-reranker-v3",
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout_s)

    @classmethod
    def from_cfg(cls, cfg: ModelCfg) -> RerankerClient:
        provider = cfg.provider.lower()
        base_url = cfg.base_url or cfg.url
        if not base_url:
            base_url = (
                "https://api.deepinfra.com/v1/inference"
                if provider == "deepinfra"
                else "http://localhost:8081"
            )
        return cls(
            base_url=base_url,
            provider=provider,
            model=cfg.model,
            api_key=cfg.api_key,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def rerank(
        self, query: str, documents: list[str], *, top_k: int | None = None
    ) -> list[RerankResult]:
        """Score every document; return them ordered by score (desc)."""
        if not documents:
            return []
        body: dict[str, object] = self._rerank_body(query, documents)
        if top_k is not None:
            body["top_n"] = top_k
        try:
            resp = await self._client.post(self._rerank_path(), json=body)
        except httpx.HTTPError as e:
            raise RerankerError(f"reranker request failed: {e}") from e
        if resp.status_code != 200:
            raise RerankerError(
                f"reranker returned {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        if "scores" in payload:
            out = [
                RerankResult(index=i, score=float(score))
                for i, score in enumerate(payload.get("scores") or [])
            ]
            out.sort(key=lambda r: r.score, reverse=True)
            if top_k is not None:
                out = out[:top_k]
            return out
        results = payload.get("results", payload.get("data", []))
        out = [
            RerankResult(
                index=r.get("index", i),
                score=float(r.get("relevance_score", r.get("score", 0.0))),
            )
            for i, r in enumerate(results)
        ]
        out.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out

    def _rerank_path(self) -> str:
        if self.provider == "deepinfra":
            return f"/{self.model}"
        return "/reranking"

    def _rerank_body(self, query: str, documents: list[str]) -> dict[str, object]:
        if self.provider == "deepinfra":
            return {"queries": [query], "documents": documents}
        return {"query": query, "documents": documents}

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False
