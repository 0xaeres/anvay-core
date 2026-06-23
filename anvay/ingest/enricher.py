"""Per-chunk contextual enrichment.

Two strategies, dispatched on chunk kind:

* **Code → HQE (Hypothetical Question Embeddings)**. The LLM generates three
  natural-language questions a developer would type to find this code. Stored
  in `context_summary` and prepended at embed time. Closes the
  English↔identifier gap. See CoIR + Sourcegraph + practitioner evals.

* **Docs → Anthropic Contextual Retrieval** (anthropic.com/news/contextual-retrieval,
  Sep 2024). The LLM gets the *whole document* + the chunk and writes a 50-100
  token "situate this chunk within the document" blurb. Stored in
  `context_summary` and prepended at embed time. Anthropic's measured numbers:
  -35% top-20 failure rate vs. raw embeddings; -49% when combined with BM25;
  -67% when also reranked. The whole-doc prefix is naturally amenable to
  server-side prompt caching, which makes the cost negligible at scale.

Without `context_summary`, doc chunks still benefit from the heading-hierarchy
`context_path` prepend in `text_for_embedding()`. Contextual retrieval is a
strict upgrade over that fallback.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from anvay.ingest.models import Chunk, ChunkKind
from anvay.llm.client import ChatClient, LLMError

log = logging.getLogger(__name__)


_HQE_PROMPT = (
    "You are a retrieval corpus annotator. For this code excerpt, write exactly 3 concise "
    "questions (each on its own line, prefixed with 'Q:') that a developer would type into "
    "a search engine to find this code. Focus on what the code DOES and what PROBLEM it "
    "SOLVES. Do not describe the location. Output only the 3 Q: lines, nothing else."
)

# Anthropic's contextual retrieval cookbook prompt, lightly adapted.
_DOC_CONTEXT_PROMPT = (
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. Answer only "
    "with the succinct context and nothing else."
)

# Cap the whole-doc prefix so a giant runbook doesn't blow the request body.
# 30k chars ~= 7.5k tokens; plenty for any reasonable doc. Larger docs get
# truncated symmetrically around the chunk to preserve its surroundings.
_DOC_TRUNCATE_CHARS = 30_000


class EnricherError(RuntimeError):
    pass


class ContextualEnricher:
    """Calls an OpenAI-compatible LLM to enrich code (HQE) and doc (CR) chunks."""

    def __init__(
        self,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        *,
        model: str = "google/gemma-4-26B-A4B-it",
        api_key: str | None = None,
        enrich_code: bool = True,
        enrich_docs: bool = True,
        concurrency: int = 4,
        timeout_s: float = 30.0,
    ):
        """
        Initialize the ContextualEnricher with model selection, feature toggles, and an internal chat client.
        
        Parameters:
            base_url (str): Base URL for the OpenAI-compatible API endpoint used by the internal chat client.
            model (str): Model identifier to use for enrichment requests.
            api_key (str | None): Optional API key for authenticating to the backend; if None, the client will rely on other configured credentials.
            enrich_code (bool): Enable high-quality enrichment for code chunks when True.
            enrich_docs (bool): Enable contextual enrichment for document chunks when True.
            concurrency (int): Maximum number of concurrent chat requests; used to initialize an internal semaphore.
            timeout_s (float): Request timeout, in seconds, applied to the internal chat client.
        """
        self.model = model
        self.enrich_code = enrich_code
        self.enrich_docs = enrich_docs
        self._chat_client = ChatClient(
            provider="openai-compatible",
            model=model,
            base_url=base_url,
            api_key=api_key,
            role="enricher",
            timeout_s=timeout_s,
            temperature=0.2,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        """
        Close the internal ChatClient and release its resources.
        """
        await self._chat_client.aclose()

    # ------------------------------------------------------------------ batch

    async def enrich(
        self,
        chunks: Iterable[Chunk],
        *,
        doc_contents: dict[str, str] | None = None,
    ) -> list[Chunk]:
        """Populate `context_summary` on each enrichable chunk.

        `doc_contents` maps `resource.uri → full document text`. Required for
        contextual retrieval on doc chunks; when absent (or when a doc's text
        is missing from the dict), doc chunks fall back to no LLM summary —
        `text_for_embedding()` will still prepend `context_path`.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return chunk_list

        docs = doc_contents or {}

        async def _summarize(chunk: Chunk) -> tuple[Chunk, str | None]:
            if chunk.kind is ChunkKind.CODE:
                if not self.enrich_code:
                    return chunk, None
                return chunk, await self._hqe_for_code(chunk)
            # doc
            if not self.enrich_docs:
                return chunk, None
            doc_text = docs.get(chunk.resource.uri)
            return chunk, await self._context_for_doc(chunk, doc_text)

        results = await asyncio.gather(*[_summarize(c) for c in chunk_list])
        for chunk, summary in results:
            if summary:
                chunk.context_summary = summary
        return chunk_list

    # ------------------------------------------------------------ code (HQE)

    async def _hqe_for_code(self, chunk: Chunk) -> str | None:
        prompt = self._render_hqe_prompt(chunk)
        return await self._chat(prompt, max_tokens=150)

    @staticmethod
    def _render_hqe_prompt(chunk: Chunk) -> str:
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
        return f"{_HQE_PROMPT}\n\n{meta}\n\nEXCERPT:\n```\n{snippet}\n```\n\nQUESTIONS:"

    # ----------------------------------------------- docs (contextual retrieval)

    async def _context_for_doc(self, chunk: Chunk, full_doc: str | None) -> str | None:
        if not full_doc:
            return None
        doc_excerpt = _truncate_doc(full_doc, around_chunk=chunk.content)
        prompt = (
            f"<document>\n{doc_excerpt}\n</document>\n\n"
            f"Here is the chunk we want to situate within the whole document:\n"
            f"<chunk>\n{chunk.content}\n</chunk>\n\n"
            f"{_DOC_CONTEXT_PROMPT}"
        )
        # 100 tokens is what Anthropic recommends for the situating context.
        return await self._chat(prompt, max_tokens=120)

    # ---------------------------------------------------------------- transport

    async def _chat(self, prompt: str, *, max_tokens: int) -> str | None:
        """
        Send a prompt to the configured chat client under the enricher's concurrency limit and return the trimmed model response.
        
        Parameters:
            prompt (str): The user-facing prompt to send to the chat model.
            max_tokens (int): Maximum number of tokens the model is allowed to generate for the response.
        
        Returns:
            str: The model's response with surrounding whitespace removed, or
            None: if the response is empty after trimming or an LLMError occurred while calling the chat client.
        """
        async with self._sem:
            try:
                resp = await self._chat_client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
            except LLMError as e:
                log.debug("enricher: chat error: %s", e)
                return None
            text = resp.content.strip()
            return text or None

    async def health(self) -> bool:
        """Return whether the configured chat provider is reachable."""
        return await self._chat_client.health()


def _truncate_doc(full_doc: str, *, around_chunk: str) -> str:
    """If the doc is too big, keep a window centred on the chunk.

    For docs under the cap we just return as-is — server-side prompt caching
    will dedupe the prefix across multiple chunks of the same doc.
    """
    if len(full_doc) <= _DOC_TRUNCATE_CHARS:
        return full_doc

    pos = full_doc.find(around_chunk[:200])  # cheap locator
    if pos < 0:
        # Chunk text not found verbatim (e.g. trimmed by the chunker) — fall
        # back to the head of the doc.
        return full_doc[:_DOC_TRUNCATE_CHARS] + "\n…[truncated]"

    half = _DOC_TRUNCATE_CHARS // 2
    start = max(0, pos - half)
    end = min(len(full_doc), pos + half)
    prefix = "…[truncated]\n" if start > 0 else ""
    suffix = "\n…[truncated]" if end < len(full_doc) else ""
    return prefix + full_doc[start:end] + suffix
