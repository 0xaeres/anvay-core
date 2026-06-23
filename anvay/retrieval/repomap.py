"""Repo map — a compact symbol outline injected into council system prompts.

Inspired by aider's repo map (aider.chat/docs/repomap.html). For each source
file under the product root we extract top-level definitions (functions,
classes, methods, structs, traits, etc.) via tree-sitter and persist the
collected symbol list to disk. At council time the map is loaded, ranked
against the session topic via lexical overlap + a small structural weight
(classes/types > functions > methods), and rendered into a token-bounded
block of `file:\n  symbol(...)` lines.

We deliberately skip aider's personalized-PageRank step for v1: with
fewer than ~5k files, lexical + structural ranking is within striking
distance of PR-based ranking and avoids networkx as a dependency. Add PR
back if an eval set proves the gap matters.

The map is persisted as JSON at `<state_dir>/repomaps/<product_id>.json` so
the council can load it without rebuilding on every session.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tree_sitter import Node

# We re-use the chunker's tree-sitter language registry so we don't drift on
# which extensions / boundary nodes are supported.
from anvay.ingest.chunker import _LANGS, _identifier_of, _lang_for

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- model


@dataclass(frozen=True)
class Symbol:
    kind: str       # 'function' | 'class' | 'method' | 'struct' | 'trait' | 'interface' | 'type'
    name: str
    file: str       # relative to the scanned root (POSIX-style)
    line: int       # 1-indexed start line
    signature: str  # short header line, e.g. 'def login(email, password)'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepoMap:
    """Persistable bag of symbols for one product."""

    symbols: list[Symbol] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.symbols

    @property
    def file_count(self) -> int:
        return len({s.file for s in self.symbols})

    @classmethod
    def from_dict(cls, d: dict) -> RepoMap:
        return cls(symbols=[Symbol(**s) for s in d.get("symbols", [])])

    def to_dict(self) -> dict:
        return {"symbols": [s.to_dict() for s in self.symbols]}

    def render(
        self,
        *,
        bias_terms: list[str] | None = None,
        token_budget: int = 600,
    ) -> str:
        """Return a compact outline ranked + truncated to roughly token_budget tokens.

        Tokens are estimated at ~4 chars/token (a safe over-estimate for code).
        Empty map → empty string; caller is expected to guard.
        """
        if not self.symbols:
            return ""
        char_budget = max(token_budget * 4, 200)
        scored = _rank(self.symbols, bias_terms=bias_terms or [])
        return _render(scored, char_budget=char_budget)


# ---------------------------------------------------------------- extraction


# A few node types per language carry signature-like headers; for each we
# capture the kind we want to surface to the LLM.
_KIND_BY_NODE: dict[str, str] = {
    # Python
    "function_definition": "function",
    "class_definition": "class",
    # TS / TSX / JS
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "variable_declarator": "function",
    # Rust
    "function_item": "function",
    "struct_item": "struct",
    "trait_item": "trait",
    "enum_item": "enum",
    "impl_item": "impl",
    # Go
    "method_declaration": "method",
    "type_declaration": "type",
    # Java / Kotlin
    "enum_declaration": "enum",
    "record_declaration": "type",
    "constructor_declaration": "method",
    "object_declaration": "object",
    "property_declaration": "property",
    "type_alias": "type",
    # C++
    "namespace_definition": "namespace",
    "class_specifier": "class",
    "struct_specifier": "struct",
    "union_specifier": "type",
    "enum_specifier": "enum",
    "template_declaration": "template",
    "declaration": "declaration",
    # Solidity
    "contract_declaration": "contract",
    "library_declaration": "library",
    "modifier_definition": "modifier",
    "struct_declaration": "struct",
    "event_definition": "event",
    "error_declaration": "error",
}

# File extensions / directories to ignore even if tree-sitter could parse them.
_IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
    "target", "out", ".next", ".turbo", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "coverage", "vendor", "third_party",
}
_MAX_FILE_BYTES = 250_000  # skip very large files (generated, vendored)


def extract_repo_map(root: Path) -> RepoMap:
    """Walk `root`, parse code files with tree-sitter, return collected symbols.

    Failures on a single file are logged and skipped — never raised.
    """
    root = Path(root)
    if not root.is_dir():
        log.warning("repomap: root %s is not a directory; returning empty map", root)
        return RepoMap()

    symbols: list[Symbol] = []
    for path in _walk_source_files(root):
        try:
            content = path.read_bytes()
        except OSError as e:
            log.debug("repomap: read failed %s: %s", path, e)
            continue
        if len(content) > _MAX_FILE_BYTES:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        lang = _lang_for(path.name)
        if not lang:
            continue
        cfg = _LANGS.get(lang)
        if cfg is None:
            continue
        try:
            file_symbols = list(_extract_file(path, text, cfg, root))
        except Exception as e:  # tree-sitter parse errors etc.
            log.debug("repomap: extract failed %s: %s", path, e)
            continue
        symbols.extend(file_symbols)

    log.info(
        "repomap: scanned %s — %d symbols across %d files",
        root,
        len(symbols),
        len({s.file for s in symbols}),
    )
    return RepoMap(symbols=symbols)


def _walk_source_files(root: Path) -> Iterator[Path]:
    """Yield code files under root, skipping ignored directories."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        if path.is_symlink():
            continue
        yield path


def _extract_file(path: Path, text: str, cfg, root: Path) -> Iterator[Symbol]:
    """Parse one file and yield top-level + nested definition symbols."""
    from tree_sitter import Parser

    parser = Parser(cfg.language)
    tree = parser.parse(text.encode("utf-8"))
    rel = path.relative_to(root).as_posix()

    # Walk all nodes (DFS); emit any node whose type maps to a kind. We don't
    # restrict to top-level — methods inside classes are useful navigation hooks.
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        kind = _KIND_BY_NODE.get(node.type)
        if kind is not None:
            name = _name_of(node, text)
            if name:
                signature = _signature_of(node, text)
                yield Symbol(
                    kind=kind,
                    name=name,
                    file=rel,
                    line=node.start_point[0] + 1,
                    signature=signature,
                )
        # Continue walking children (named only — skips punctuation tokens).
        stack.extend(reversed(node.named_children))


def _name_of(node: Node, text: str) -> str | None:
    """Best-effort name extraction. Most languages expose a `name` field."""
    raw = _identifier_of(node)
    return raw[:80] if raw else None


def _signature_of(node: Node, text: str) -> str:
    """Return the first non-empty line of the node — the header / signature."""
    start = node.start_byte
    end = node.end_byte
    snippet = text[start:end]
    for line in snippet.splitlines():
        s = line.strip()
        if s:
            # Trim trailing brace/colon for readability.
            return s.rstrip("{").rstrip(":").strip()[:120]
    return ""


# ---------------------------------------------------------------- ranking + render


_BASE_WEIGHT = {
    "class": 2.0,
    "struct": 2.0,
    "trait": 2.0,
    "interface": 2.0,
    "contract": 2.0,
    "library": 1.8,
    "enum": 1.8,
    "type": 1.5,
    "template": 1.5,
    "namespace": 1.4,
    "function": 1.2,
    "method": 1.0,
    "impl": 1.0,
    "modifier": 1.0,
}


def _rank(symbols: list[Symbol], *, bias_terms: list[str]) -> list[tuple[float, Symbol]]:
    """Score each symbol by (lexical overlap with bias) + (kind weight). Higher wins.

    Lexical match is computed against the symbol name + file path so a topic
    like "auth token rotation" pulls in `AuthHandler`, `verify_token`, and
    files under `auth/`.
    """
    bias = {t.lower() for t in bias_terms if len(t) >= 3}
    scored: list[tuple[float, Symbol]] = []
    for s in symbols:
        weight = _BASE_WEIGHT.get(s.kind, 1.0)
        if bias:
            haystack = f"{s.name} {s.file}".lower()
            hits = sum(1 for t in bias if t in haystack)
            weight += 3.0 * hits
        scored.append((weight, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _render(scored: list[tuple[float, Symbol]], *, char_budget: int) -> str:
    """Group symbols by file in score order, render under a char budget.

    Files appear in the order their highest-ranked symbol was first seen, so
    a topic-relevant file lands at the top. Within a file, symbols are sorted
    by line number so the rendering reads top-down like the source.
    """
    by_file: dict[str, list[Symbol]] = {}
    for _, s in scored:
        by_file.setdefault(s.file, []).append(s)

    out_lines: list[str] = ["## Codebase map"]
    used_chars = len(out_lines[0]) + 1

    for file_path, syms in by_file.items():
        syms_sorted = sorted(syms, key=lambda s: s.line)
        header = f"\n{file_path}"
        block: list[str] = [header]
        for s in syms_sorted:
            block.append(f"  {s.signature}  [L{s.line}]")
        block_text = "\n".join(block)
        if used_chars + len(block_text) + 1 > char_budget:
            # Skip this file entirely if it would blow the budget — keeps the
            # rendering coherent (no half-rendered file).
            continue
        out_lines.append(block_text)
        used_chars += len(block_text) + 1

    if len(out_lines) == 1:
        return ""  # header only — caller treats as empty
    return "\n".join(out_lines)


# ---------------------------------------------------------------- persistence


def save_repo_map(rm: RepoMap, path: Path) -> None:
    """Atomic-ish write: dump JSON then rename onto target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rm.to_dict()), encoding="utf-8")
    tmp.replace(path)


def load_repo_map(path: Path) -> RepoMap | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("repomap: load failed %s: %s", path, e)
        return None
    return RepoMap.from_dict(data)


def repomap_path_for(state_dir: Path, product_id: str) -> Path:
    """Canonical location: `<state_dir>/repomaps/<product_id>.json`."""
    return Path(state_dir) / "repomaps" / f"{product_id}.json"


def load_repo_map_for_product(config, product_id: str) -> RepoMap:
    """Load the persisted map for a product. Empty RepoMap if none was built yet."""
    state_dir = Path(config.storage.proposal_queue).parent
    return load_repo_map(repomap_path_for(state_dir, product_id)) or RepoMap()


def topic_bias_terms(topic: str) -> list[str]:
    """Tokenize a council topic into bias terms for ranking. Drops <3-char tokens."""
    out: list[str] = []
    cur: list[str] = []
    for ch in topic.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if len(tok) >= 3:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if len(tok) >= 3:
            out.append(tok)
    return out
