"""Ingestion data model.

A `ResourceRef` identifies one file/page/document inside a source.
A `Chunk` is a span of that resource carrying a `file:line` anchor.
An `EmbeddedChunk` is a chunk plus its dense vector (named in Qdrant).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, computed_field

_ANVAY_NS = uuid.UUID("8c4f4d7e-2c1b-4a6a-9a3e-1f5b8d2c9e10")

# Hard cap on the characters fed to the embedder. Stored `content` is never
# truncated — this only bounds the embedding representation so oversized
# skeletons/doc spills stay inside local llama.cpp physical batch limits
# (--ubatch-size 512 ≈ ~2000 chars incl. context header).
EMBED_CHAR_CAP = 1600


def symbol_id_for(product_id: str, resource_uri: str, context_path: str) -> str:
    """Deterministic id linking all chunks of one symbol (declaration chunk,
    doc-comment spill chunk, oversized-split sub-chunks). Keyed on the
    qualified name — NOT line spans — so it survives line shifts."""
    key = f"{product_id}|{resource_uri}|{context_path}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ChunkKind(StrEnum):
    CODE = "code"
    DOC = "doc"


class ResourceRef(BaseModel):
    """Pointer to a single addressable resource within a source."""

    source_id: str  # e.g. "github:myorg/repo" or "local:/abs/path"
    uri: str  # canonical URI for the resource (path or URL)
    mime: str
    size_bytes: int | None = None
    last_modified: str | None = None  # ISO-8601 when known

    @computed_field
    @property
    def kind(self) -> ChunkKind:
        if _is_code_mime(self.mime) or _is_code_path(self.uri):
            return ChunkKind.CODE
        return ChunkKind.DOC


class Chunk(BaseModel):
    """A retrievable span of a resource."""

    product_id: str
    resource: ResourceRef
    content: str
    start_line: int  # 1-indexed (first line of content)
    end_line: int  # 1-indexed, inclusive
    kind: ChunkKind
    # Structural context discovered by the chunker (function/class/heading name)
    context_path: str | None = None
    # Filled by the contextual enricher (ADR-010); falsy = no enrichment
    context_summary: str | None = None
    # Links declaration / doc-spill / split sub-chunks of the same symbol.
    symbol_id: str | None = None
    # First declaration line — breadcrumb for embed text of split sub-chunks.
    signature: str | None = None

    @computed_field
    @property
    def id(self) -> str:
        """Deterministic content-addressable UUID (valid Qdrant point ID)."""
        key = f"{self.product_id}|{self.resource.uri}|{self.start_line}-{self.end_line}"
        return str(uuid.uuid5(_ANVAY_NS, key))

    @property
    def anchor(self) -> str:
        """The `file:line` anchor used in citations."""
        return f"{self.resource.uri}:{self.start_line}"

    def text_for_embedding(self) -> str:
        """The string actually fed to the embedder.

        Header (context path + signature + enricher summary) is never
        truncated; content is head-truncated so header + content stays under
        EMBED_CHAR_CAP. Stored `content` is unaffected."""
        parts: list[str] = []
        if self.context_path:
            parts.append(self.context_path)
        if self.signature and self.signature.strip() != (self.context_path or "").strip():
            parts.append(self.signature)
        if self.context_summary:
            parts.append(self.context_summary)
        header = "\n".join(parts)
        if not header:
            return self.content[:EMBED_CHAR_CAP]
        if len(header) >= EMBED_CHAR_CAP:
            return header[:EMBED_CHAR_CAP]
        budget = EMBED_CHAR_CAP - len(header) - 2
        if budget <= 0:
            return header[:EMBED_CHAR_CAP]
        return f"{header}\n\n{self.content[:budget]}"[:EMBED_CHAR_CAP]

    def sparse_text_for_embedding(self) -> str:
        """BM25 passage text: embed text plus split camelCase/snake_case
        identifier parts so exact-word queries match code tokens."""
        base = self.text_for_embedding()
        decoration = _identifier_decoration(base)
        if not decoration:
            return base
        return f"{base}\n{decoration}"


class EmbeddedChunk(BaseModel):
    """A chunk with its computed vector for the appropriate Qdrant named vector."""

    chunk: Chunk
    vector: list[float]
    vector_name: Literal["dense_code", "dense_text"]


# ---------------------------------------------------------------- helpers


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
_MAX_DECORATION_TOKENS = 80


def _identifier_decoration(text: str) -> str:
    """Split compound identifiers into word parts (`getUserById` →
    `get user by id`). Doc-side only; appended, never replacing the original
    tokens. Deterministic — no model calls."""
    seen: set[str] = set()
    out: list[str] = []
    for ident in _IDENTIFIER_RE.findall(text):
        if "_" not in ident and ident.lower() == ident:
            continue  # single lowercase word — already a BM25 token
        parts = [p.lower() for p in _CAMEL_SPLIT_RE.findall(ident) if len(p) > 1]
        if len(parts) < 2:
            continue
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
                if len(out) >= _MAX_DECORATION_TOKENS:
                    return " ".join(out)
    return " ".join(out)


_CODE_EXTS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".swift",
    ".rb",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".cs",
    ".sol",
    ".kts",
}

_CODE_MIMES = {
    "text/x-python",
    "text/x-typescript",
    "text/x-javascript",
    "application/javascript",
    "text/x-rust",
    "text/x-go",
}


def _is_code_path(uri: str) -> bool:
    return any(uri.endswith(ext) for ext in _CODE_EXTS)


def _is_code_mime(mime: str) -> bool:
    return mime in _CODE_MIMES or mime.startswith("text/x-")


def guess_mime(path: str) -> str:
    """Lightweight mime guess based on extension; used by sources that don't carry mimes."""
    lower = path.lower()
    if lower.endswith(".py"):
        return "text/x-python"
    if lower.endswith((".ts", ".tsx")):
        return "text/x-typescript"
    if lower.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "application/javascript"
    if lower.endswith(".rs"):
        return "text/x-rust"
    if lower.endswith(".go"):
        return "text/x-go"
    if lower.endswith(".java"):
        return "text/x-java"
    if lower.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h")):
        return "text/x-c++"
    if lower.endswith((".kt", ".kts")):
        return "text/x-kotlin"
    if lower.endswith(".sol"):
        return "text/x-solidity"
    if lower.endswith((".md", ".mdx")):
        return "text/markdown"
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith((".txt", ".rst")):
        return "text/plain"
    return "application/octet-stream"
