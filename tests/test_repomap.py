"""Tests for the tree-sitter repo map."""

from __future__ import annotations

from pathlib import Path

from nexus.retrieval.repomap import (
    RepoMap,
    Symbol,
    extract_repo_map,
    load_repo_map,
    repomap_path_for,
    save_repo_map,
    topic_bias_terms,
)

# ---------- extract_repo_map ------------------------------------------------


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_extract_finds_python_classes_and_functions(tmp_path: Path) -> None:
    _write(tmp_path, "auth.py", '''
class AuthHandler:
    def verify_token(self, token: str) -> bool:
        return True

    def refresh(self):
        pass


def login_user(email: str, password: str):
    return AuthHandler()
'''.strip())
    rm = extract_repo_map(tmp_path)
    names = {(s.kind, s.name) for s in rm.symbols}
    assert ("class", "AuthHandler") in names
    assert ("method", "verify_token") in names or ("function", "verify_token") in names
    assert ("function", "login_user") in names


def test_extract_finds_typescript_definitions(tmp_path: Path) -> None:
    _write(tmp_path, "src/auth.ts", '''
export interface AuthConfig {
  secret: string
}

export type Token = string

export class TokenService {
  rotate() {}
}

export function verifyToken(t: string): boolean { return true }
'''.strip())
    rm = extract_repo_map(tmp_path)
    kinds = {(s.kind, s.name) for s in rm.symbols}
    assert ("interface", "AuthConfig") in kinds
    assert ("type", "Token") in kinds
    assert ("class", "TokenService") in kinds
    assert ("function", "verifyToken") in kinds


def test_extract_finds_required_polyglot_symbols(tmp_path: Path) -> None:
    _write(tmp_path, "src/Greeter.java", """
class Greeter {
  public String greet(String name) {
    return "hi " + name;
  }
}
""".strip())
    _write(tmp_path, "src/greeter.cpp", """
namespace demo {
class Greeter {
public:
  const char* greet() { return "hi"; }
};
int add(int a, int b) { return a + b; }
}
""".strip())
    _write(tmp_path, "src/Greeter.kt", """
class Greeter(val prefix: String) {
  fun greet(name: String): String {
    return prefix + name
  }
}
""".strip())
    _write(tmp_path, "src/Vault.sol", """
contract Vault {
  struct Account { uint bal; }
  function deposit() public payable {
  }
}
""".strip())

    rm = extract_repo_map(tmp_path)
    by_file = {(s.file, s.name) for s in rm.symbols}

    assert ("src/Greeter.java", "Greeter") in by_file
    assert ("src/greeter.cpp", "Greeter") in by_file
    assert ("src/Greeter.kt", "Greeter") in by_file
    assert ("src/Vault.sol", "Vault") in by_file


def test_extract_skips_ignored_dirs(tmp_path: Path) -> None:
    _write(tmp_path, "node_modules/lib.js", "function junk() {}")
    _write(tmp_path, "src/real.py", "def real(): pass")
    rm = extract_repo_map(tmp_path)
    files = {s.file for s in rm.symbols}
    assert "src/real.py" in files
    assert not any("node_modules" in f for f in files)


def test_extract_skips_unknown_extensions(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Hello")
    _write(tmp_path, "data.json", '{"x": 1}')
    rm = extract_repo_map(tmp_path)
    assert rm.is_empty


def test_extract_handles_unparseable_file_gracefully(tmp_path: Path) -> None:
    # Syntactically broken Python — extract_repo_map must not raise even when
    # tree-sitter recovery yields nothing useful.
    _write(tmp_path, "broken.py", "def foo(:\n    nope nope\nclass Bar:\n  pass\n")
    _write(tmp_path, "ok.py", "def ok(): pass\n")
    rm = extract_repo_map(tmp_path)
    # The healthy file should still be extracted.
    assert "ok" in {s.name for s in rm.symbols}


def test_extract_skips_oversized_files(tmp_path: Path) -> None:
    # Generated/vendored files often blow the budget — skip them entirely.
    huge = "def f{}():\n    pass\n\n"
    _write(tmp_path, "vendored.py", "".join(huge.format(i) for i in range(40_000)))
    rm = extract_repo_map(tmp_path)
    assert "vendored.py" not in {s.file for s in rm.symbols}


# ---------- ranking + render ------------------------------------------------


def _sym(kind: str, name: str, file: str = "f.py", line: int = 1, sig: str = "") -> Symbol:
    return Symbol(kind=kind, name=name, file=file, line=line, signature=sig or name)


def test_render_empty_map_returns_empty_string() -> None:
    assert RepoMap(symbols=[]).render() == ""


def test_render_includes_header_and_file_block() -> None:
    rm = RepoMap(symbols=[_sym("class", "Auth", "auth.py", 10, "class Auth:")])
    out = rm.render(token_budget=200)
    assert "## Codebase map" in out
    assert "auth.py" in out
    assert "class Auth:" in out
    assert "[L10]" in out


def test_render_groups_symbols_under_their_file_in_source_order() -> None:
    rm = RepoMap(symbols=[
        _sym("function", "b", "x.py", 20, "def b():"),
        _sym("function", "a", "x.py", 10, "def a():"),
    ])
    out = rm.render(token_budget=400)
    a_pos = out.index("def a():")
    b_pos = out.index("def b():")
    assert a_pos < b_pos  # within a file, line-order


def test_render_ranks_bias_matches_above_others() -> None:
    # `auth_match` should outrank `unrelated`; the auth file lands first.
    rm = RepoMap(symbols=[
        _sym("function", "unrelated", "utils.py", 5, "def unrelated():"),
        _sym("function", "auth_match", "auth/login.py", 1, "def auth_match():"),
    ])
    out = rm.render(bias_terms=["auth"], token_budget=400)
    assert out.index("auth/login.py") < out.index("utils.py")


def test_render_respects_token_budget() -> None:
    # 100 chars per symbol x 30 symbols = 3000 chars; budget of 200 tokens
    # (~800 chars) must drop most of them.
    syms = [
        _sym("function", f"sym_{i}", f"f{i}.py", 1, f"def sym_{i}(): " + "x" * 80)
        for i in range(30)
    ]
    out = RepoMap(symbols=syms).render(token_budget=200)
    # Output should be well below the safety-cap on chars.
    assert len(out) <= 200 * 4 + 50
    # And should still include at least one rendered file block.
    assert "## Codebase map" in out


def test_render_skips_files_that_would_blow_budget() -> None:
    # The big file appears first by rank but should be skipped because its
    # block alone exceeds the budget; the small file still renders.
    big = [_sym("function", f"big_{i}", "big.py", i, "x" * 200) for i in range(20)]
    small = [_sym("function", "small", "small.py", 1, "def small():")]
    rm = RepoMap(symbols=big + small)
    out = rm.render(bias_terms=[], token_budget=80)  # tiny budget
    assert "small.py" in out
    assert "big.py" not in out


# ---------- persistence -----------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    rm = RepoMap(symbols=[_sym("class", "X", "a.py", 1, "class X:")])
    p = repomap_path_for(tmp_path, "demo")
    save_repo_map(rm, p)
    loaded = load_repo_map(p)
    assert loaded is not None
    assert loaded.symbols == rm.symbols


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_repo_map(tmp_path / "absent.json") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert load_repo_map(bad) is None


# ---------- topic_bias_terms ------------------------------------------------


def test_topic_bias_terms_splits_and_drops_short() -> None:
    assert topic_bias_terms("Auth token rotation") == ["auth", "token", "rotation"]
    assert topic_bias_terms("a-b-c") == []   # all 1-char tokens
    assert topic_bias_terms("PDA seed_validation") == ["pda", "seed_validation"]
