"""Contextual chunk enrichment via hypothetical question generation (HQE) - ADR-010.

For code chunks, asks a light LLM to generate 3 hypothetical questions that the
chunk answers. Stored in context_summary and prepended before embedding via
text_for_embedding() — bridges the code→natural-language query gap.

Doc chunks: no LLM call. context_path (heading hierarchy) is prepended directly
by text_for_embedding() in models.py.

Uses OpenAI-compatible /v1/chat/completions (DeepInfra or any compatible endpoint).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import httpx

from nexus.ingest.models import Chunk, ChunkKind

_DEFAULT_PROMPT = (
    "You are a retrieval corpus annotator. For this code excerpt, write exactly 3 concise "
    "questions (each on its own line, prefixed with 'Q:') that a developer would type into "
    "a search engine to find this code. Focus on what the code DOES and what PROBLEM it "
    "SOLVES. Do not describe the location. Output only the 3 Q: lines, nothing else."
)


class EnricherError(RuntimeError):
    pass


class ContextualEnricher:
    """Calls an OpenAI-compatible LLM to generate hypothetical questions for code chunks."""

    def __init__(
        self,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        *,
        model: str = "google/gemma-3-4b-it",
        api_key: str | None = None,
        enrich_code: bool = True,
        enrich_docs: bool = False,
        concurrency: int = 4,
        timeout_s: float = 30.0,
    ):
        self.model = model
        self.enrich_code = enrich_code
        self.enrich_docs = enrich_docs
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _should_enrich(self, chunk: Chunk) -> bool:
        if chunk.kind is ChunkKind.CODE:
            return self.enrich_code
        return self.enrich_docs

    # ------------------------------------------------------------------ batch

    async def enrich(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        """Return chunks (possibly with `context_summary` populated)."""
        chunk_list = list(chunks)
        targets = [c for c in chunk_list if self._should_enrich(c)]
        if not targets:
            return chunk_list
        summaries = await asyncio.gather(*[self._summary_for(c) for c in targets])
        for c, summary in zip(targets, summaries, strict=True):
            if summary:
                c.context_summary = summary
        return chunk_list

    async def _summary_for(self, chunk: Chunk) -> str | None:
        async with self._sem:
            prompt = self._render_prompt(chunk)
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.2,
                    },
                )
            except httpx.HTTPError:
                return None
            if resp.status_code != 200:
                return None
            choices = resp.json().get("choices") or []
            if not choices:
                return None
            text = (choices[0].get("message") or {}).get("content", "").strip()
            return text or None

    @staticmethod
    def _render_prompt(chunk: Chunk) -> str:
        meta_lines = [
            f"FILE: {chunk.resource.uri}",
            f"LINES: {chunk.start_line}-{chunk.end_line}",
        ]
        if chunk.context_path:
            meta_lines.append(f"STRUCT: {chunk.context_path}")
        meta = "\n".join(meta_lines)
        snippet = chunk.content
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n…"
        return f"{_DEFAULT_PROMPT}\n\n{meta}\n\nEXCERPT:\n```\n{snippet}\n```\n\nQUESTIONS:"

    async def health(self) -> bool:
        try:
            r = await self._client.get("/models")
            return r.status_code == 200
        except httpx.HTTPError:
            return False
