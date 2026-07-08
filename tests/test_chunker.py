from anvay.ingest.chunker import chunk_resource
from anvay.ingest.models import ChunkKind, ResourceRef


def _res(uri: str, mime: str) -> ResourceRef:
    return ResourceRef(source_id="local:test", uri=uri, mime=mime)


def test_python_chunks_at_function_and_class_boundaries() -> None:
    code = (
        "import os\n"
        "\n"
        "def hello(name: str) -> str:\n"
        '    """Greet someone."""\n'
        '    return f"Hello, {name}!"\n'
        "\n"
        "class Greeter:\n"
        "    def __init__(self, prefix: str):\n"
        "        self.prefix = prefix\n"
        "\n"
        "    def greet(self, name: str) -> str:\n"
        '        return f"{self.prefix} {name}!"\n'
    )
    chunks = chunk_resource("forge", _res("a.py", "text/x-python"), code)
    paths = sorted(c.context_path for c in chunks if c.context_path)
    assert "hello" in paths
    assert "Greeter" in paths
    assert "Greeter.__init__" in paths
    assert "Greeter.greet" in paths
    # All code chunks
    assert all(c.kind is ChunkKind.CODE for c in chunks)
    # Each anchor points at a real line
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_markdown_chunks_carry_heading_path() -> None:
    md = (
        "# Project\n"
        "\nIntro text long enough.\n"
        "\n## Setup\n"
        "\nStep 1: install dependencies.\n"
        "Step 2: run the thing.\n"
        "\n## Usage\n"
        "\nRun with --help to see options.\n"
        "\n### Advanced\n"
        "\nPower-user features go here.\n"
    )
    chunks = chunk_resource("forge", _res("README.md", "text/markdown"), md)
    assert chunks, "markdown should produce chunks"
    paths = [c.context_path for c in chunks]
    assert any(p and "Setup" in p for p in paths)
    assert any(p and "Usage" in p for p in paths)
    assert any(p and "Advanced" in p and "Usage" in p for p in paths)
    assert all(c.kind is ChunkKind.DOC for c in chunks)


def test_chunk_id_is_deterministic_uuid() -> None:
    code = "def foo():\n    pass\n    return 1\n    return 2\n"
    a = chunk_resource("p", _res("x.py", "text/x-python"), code)
    b = chunk_resource("p", _res("x.py", "text/x-python"), code)
    assert [c.id for c in a] == [c.id for c in b]
    for c in a:
        # UUID format: 8-4-4-4-12
        parts = c.id.split("-")
        assert len(parts) == 5


def test_chunk_anchor_matches_start_line() -> None:
    code = (
        "x = 1\n"
        "\n"
        "def bar(value: int) -> int:\n"
        '    """Double a number with a long-enough body to clear min size."""\n'
        "    if value < 0:\n"
        "        return 0\n"
        "    return value * 2\n"
    )
    chunks = chunk_resource("p", _res("y.py", "text/x-python"), code)
    bar = next(c for c in chunks if c.context_path == "bar")
    assert bar.anchor.endswith(f":{bar.start_line}")


def test_text_for_embedding_prepends_context_summary() -> None:
    code = (
        "def foo(seed: int) -> list[int]:\n"
        '    """Produce a small list — bodied enough to be a chunk."""\n'
        "    result = []\n"
        "    for i in range(seed):\n"
        "        result.append(i * 2)\n"
        "    return result\n"
    )
    chunks = chunk_resource("p", _res("z.py", "text/x-python"), code)
    assert chunks, "expected at least one chunk for a real function body"
    c = chunks[0]
    # context_path is prepended even without a context_summary
    assert c.context_path is not None
    assert c.text_for_embedding().startswith(c.context_path)
    assert c.content in c.text_for_embedding()
    # context_summary joins the header alongside context_path + signature
    c.context_summary = "Q: How to produce a list from seed?\nQ: What does foo return?"
    embed_text = c.text_for_embedding()
    assert "Q: How to produce" in embed_text
    header = embed_text.split("\n\n", 1)[0]
    assert c.context_path in header
    assert "Q: How to produce" in header
    assert c.content in embed_text


def test_required_code_languages_produce_code_chunks_with_context_paths() -> None:
    cases = {
        "Greeter.java": (
            "text/x-java",
            "class Greeter {\n"
            "  private String prefix;\n"
            "  Greeter(String p) { this.prefix = p; }\n"
            "  public String greet(String name) {\n"
            "    return prefix + name;\n"
            "  }\n"
            "}\n",
            "Greeter.greet",
        ),
        "widget.tsx": (
            "text/x-typescript",
            "type Props = { title: string };\n"
            "export const Widget = (props: Props) => {\n"
            "  return props.title ? <div>{props.title}</div> : <span>missing</span>;\n"
            "};\n",
            "Widget",
        ),
        "lib.rs": (
            "text/x-rust",
            "struct Greeter { prefix: String }\n"
            "impl Greeter {\n"
            "  fn greet(&self, name: &str) -> String {\n"
            "    format!(\"{} {}\", self.prefix, name)\n"
            "  }\n"
            "}\n",
            "Greeter",
        ),
        "greeter.cpp": (
            "text/x-c++",
            "namespace demo {\n"
            "class Greeter {\n"
            "public:\n"
            "  const char* greet() { return \"hello\"; }\n"
            "};\n"
            "int add(int a, int b) {\n"
            "  int sum = a + b;\n"
            "  return sum;\n"
            "}\n"
            "}\n",
            "demo.Greeter",
        ),
        "Greeter.kt": (
            "text/x-kotlin",
            "class Greeter(val prefix: String) {\n"
            "  fun greet(name: String): String {\n"
            "    return prefix + name\n"
            "  }\n"
            "}\n",
            "Greeter.greet",
        ),
        "Vault.sol": (
            "text/x-solidity",
            "contract Vault {\n"
            "  event Deposit(address indexed user);\n"
            "  error NoFunds();\n"
            "  struct Account { uint bal; }\n"
            "  modifier onlyOwner() { _; }\n"
            "  function deposit() public payable {\n"
            "    emit Deposit(msg.sender);\n"
            "  }\n"
            "}\n",
            "Vault.deposit",
        ),
    }

    for uri, (mime, code, expected_path) in cases.items():
        chunks = chunk_resource("p", _res(uri, mime), code)
        assert chunks, f"{uri} should produce chunks"
        assert all(c.kind is ChunkKind.CODE for c in chunks)
        assert any(c.context_path == expected_path for c in chunks)
        assert all(c.start_line >= 1 and c.end_line >= c.start_line for c in chunks)


def test_javascript_arrow_function_gets_symbol_context() -> None:
    code = (
        "export const Widget = (props) => {\n"
        "  const title = props.title || 'missing';\n"
        "  return title.toUpperCase();\n"
        "};\n"
    )

    chunks = chunk_resource("p", _res("Widget.js", "application/javascript"), code)

    assert any(c.kind is ChunkKind.CODE and c.context_path == "Widget" for c in chunks)


def test_java_javadoc_attached_to_class_chunk() -> None:
    """Javadoc block_comment must land in the class chunk, not as a stray <module> chunk."""
    code = (
        "package com.example;\n"
        "\n"
        "/**\n"
        " * Immutable ordered collection of elements.\n"
        " * Use this when you need fast indexed access.\n"
        " */\n"
        "public class ImmutableList<E> {\n"
        "    private final Object[] elements;\n"
        "    public ImmutableList(Object[] elems) {\n"
        "        this.elements = elems.clone();\n"
        "    }\n"
        "    public E get(int index) {\n"
        "        return (E) elements[index];\n"
        "    }\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("ImmutableList.java", "text/x-java"), code)
    class_chunks = [c for c in chunks if c.context_path == "ImmutableList"]
    module_chunks = [c for c in chunks if c.context_path == "<module>"]
    assert class_chunks, "ImmutableList chunk must exist"
    # Javadoc text is in the class chunk, not in a <module> chunk
    assert any("Immutable ordered collection" in c.content for c in class_chunks), (
        "javadoc must be attached to the ImmutableList class chunk"
    )
    assert not any("Immutable ordered collection" in c.content for c in module_chunks), (
        "javadoc must NOT appear as an orphaned <module> chunk"
    )


def test_java_line_comment_attached_to_method_chunk() -> None:
    """Leading line_comment must attach to the following method chunk."""
    code = (
        "class Util {\n"
        "    // Returns the sum of two integers.\n"
        "    // This is a simple utility method.\n"
        "    public int add(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("Util.java", "text/x-java"), code)
    method_chunks = [c for c in chunks if c.context_path and "add" in c.context_path]
    assert method_chunks, "add method chunk must exist"
    assert any("Returns the sum" in c.content for c in method_chunks), (
        "line comment must be attached to the add method chunk"
    )
    assert not any(
        "Returns the sum" in c.content
        for c in chunks
        if c.context_path == "<module>"
    ), "comment must NOT be an orphaned <module> chunk"


def test_unknown_code_extension_uses_code_fallback_not_doc_chunks() -> None:
    code = (
        "class Greeter\n"
        "  def greet(name)\n"
        "    message = \"hello #{name}\"\n"
        "    message.upcase\n"
        "  end\n"
        "end\n"
    )

    chunks = chunk_resource("p", _res("greeter.rb", "text/x-ruby"), code)

    assert chunks
    assert all(c.kind is ChunkKind.CODE for c in chunks)
    assert all(c.context_path == "<module>" for c in chunks)


# ---------------------------------------------------------------- skeleton / B1


_BIG_PY_CLASS = (
    '"""Module docstring."""\n'
    "\n"
    "class Widget:\n"
    '    """A widget that does widget things."""\n'
    "\n"
    "    def alpha(self) -> int:\n"
    '        """Compute alpha using the alpha-specific widget algorithm."""\n'
    "        total_alpha_value = 0\n"
    "        for step_index in range(100):\n"
    "            total_alpha_value += step_index * 3\n"
    "        return total_alpha_value\n"
    "\n"
    "    def beta(self) -> str:\n"
    '        """Render beta as a long descriptive string for testing."""\n'
    "        rendered_beta_output = 'beta:' + '-'.join(str(n) for n in range(50))\n"
    "        return rendered_beta_output.upper().strip().replace('-', '_')\n"
)


def test_container_body_text_lives_in_exactly_one_chunk() -> None:
    """B1 regression: member bodies must not be duplicated in the class chunk."""
    chunks = chunk_resource("p", _res("w.py", "text/x-python"), _BIG_PY_CLASS)
    needle = "total_alpha_value += step_index * 3"
    holders = [c for c in chunks if needle in c.content]
    assert len(holders) == 1
    assert holders[0].context_path == "Widget.alpha"


def test_container_emits_skeleton_with_member_signatures() -> None:
    chunks = chunk_resource("p", _res("w.py", "text/x-python"), _BIG_PY_CLASS)
    class_chunks = [c for c in chunks if c.context_path == "Widget"]
    assert len(class_chunks) == 1
    skeleton = class_chunks[0]
    assert "class Widget:" in skeleton.content
    assert "A widget that does widget things." in skeleton.content
    assert "def alpha(self) -> int:" in skeleton.content
    assert "def beta(self) -> str:" in skeleton.content
    # Bodies elided
    assert "total_alpha_value += step_index * 3" not in skeleton.content
    # Anchor covers the full container span for citations
    assert skeleton.start_line == 3
    assert skeleton.end_line == len(_BIG_PY_CLASS.splitlines())
    assert skeleton.signature == "class Widget:"


def test_skeleton_and_members_share_no_symbol_but_have_own_symbol_ids() -> None:
    chunks = chunk_resource("p", _res("w.py", "text/x-python"), _BIG_PY_CLASS)
    by_ctx = {c.context_path: c for c in chunks if c.kind is ChunkKind.CODE}
    assert by_ctx["Widget"].symbol_id != by_ctx["Widget.alpha"].symbol_id
    assert all(c.symbol_id for c in chunks if c.kind is ChunkKind.CODE)


def test_nested_container_members_not_duplicated() -> None:
    code = (
        "class Outer:\n"
        '    """Outer container."""\n'
        "\n"
        "    class Inner:\n"
        '        """Inner container with its own method."""\n'
        "\n"
        "        def inner_method(self) -> int:\n"
        "            accumulated_inner_total = sum(k * 7 for k in range(40))\n"
        "            return accumulated_inner_total + 11\n"
        "\n"
        "    def outer_method(self) -> int:\n"
        "        outer_result_value = max(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)\n"
        "        return outer_result_value * 13\n"
    )
    chunks = chunk_resource("p", _res("n.py", "text/x-python"), code)
    needle = "accumulated_inner_total = sum(k * 7 for k in range(40))"
    holders = [c for c in chunks if needle in c.content]
    assert len(holders) == 1
    assert holders[0].context_path == "Outer.Inner.inner_method"


def test_container_skeletons_across_languages() -> None:
    java = (
        "/** Greets people politely and with enthusiasm. */\n"
        "public class Greeter {\n"
        "    /** Greet by name with a decorated salutation string. */\n"
        "    public String greet(String name) {\n"
        "        String decorated = \"Hello, \" + name + \"! Welcome to the widget factory.\";\n"
        "        return decorated.trim().toUpperCase();\n"
        "    }\n"
        "    public int count(String input) {\n"
        "        int total = input.length() * 3 + 42;\n"
        "        return Math.max(total, input.hashCode());\n"
        "    }\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("Greeter.java", "text/x-java"), java)
    class_chunks = [c for c in chunks if c.context_path == "Greeter"]
    assert len(class_chunks) == 1
    body_needle = 'String decorated = "Hello, "'
    assert body_needle not in class_chunks[0].content
    assert len([c for c in chunks if body_needle in c.content]) == 1

    rust = (
        "/// Frobnicator implementation block for the primary widget type.\n"
        "impl Frobnicator {\n"
        "    /// Frobnicate the widget using the standard coefficient table.\n"
        "    pub fn frobnicate(&self, factor: u32) -> u32 {\n"
        "        let intermediate_frobnication_value = factor * 31 + 7;\n"
        "        intermediate_frobnication_value.wrapping_mul(3)\n"
        "    }\n"
        "    pub fn defrobnicate(&self, value: u32) -> u32 {\n"
        "        let restored_original_widget_value = value / 3;\n"
        "        restored_original_widget_value.saturating_sub(7) / 31\n"
        "    }\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("f.rs", "text/x-rust"), rust)
    impl_chunks = [c for c in chunks if c.context_path == "Frobnicator"]
    assert len(impl_chunks) == 1
    needle = "let intermediate_frobnication_value = factor * 31 + 7;"
    assert needle not in impl_chunks[0].content
    assert len([c for c in chunks if needle in c.content]) == 1


# ---------------------------------------------------------------- doc-comment split


def test_long_doc_comment_spills_into_linked_doc_chunk() -> None:
    doc_lines = "\n".join(
        f"/// Detail paragraph line {i} explaining edge case behaviour in depth."
        for i in range(12)
    )
    rust = (
        "/// Summary line for the widget frobnication entry point.\n"
        "///\n"
        f"{doc_lines}\n"
        "pub fn frobnicate_widget(factor: u32) -> u32 {\n"
        "    let computed_widget_output = factor * 31 + 7;\n"
        "    computed_widget_output.wrapping_mul(3)\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("d.rs", "text/x-rust"), rust)
    code_chunks = [c for c in chunks if c.kind is ChunkKind.CODE and c.context_path == "frobnicate_widget"]
    doc_chunks = [c for c in chunks if c.kind is ChunkKind.DOC]
    assert len(code_chunks) >= 1
    assert len(doc_chunks) == 1
    decl = code_chunks[0]
    spill = doc_chunks[0]
    # Summary stays with the declaration; detail moved to the spill chunk.
    assert "Summary line for the widget" in decl.content
    assert "Detail paragraph line 3" not in decl.content
    assert "Detail paragraph line 3" in spill.content
    # Linked by symbol_id, same qualified context.
    assert spill.symbol_id == decl.symbol_id
    assert spill.context_path == "frobnicate_widget"
    assert spill.signature == decl.signature


def test_short_doc_comment_stays_attached() -> None:
    rust = (
        "/// Short summary that fits fine.\n"
        "pub fn tiny_helper(factor: u32) -> u32 {\n"
        "    let helper_result_value = factor * 31 + 7;\n"
        "    helper_result_value.wrapping_mul(3)\n"
        "}\n"
    )
    chunks = chunk_resource("p", _res("s.rs", "text/x-rust"), rust)
    assert all(c.kind is ChunkKind.CODE for c in chunks)
    assert any("Short summary that fits fine." in c.content for c in chunks)


# ---------------------------------------------------------------- embed text / sparse


def test_split_subchunks_carry_signature_breadcrumb() -> None:
    body = "\n".join(
        f"    processed_row_accumulator_{i} = transform_input_row(raw_input_rows[{i}])"
        for i in range(40)
    )
    code = f"def process_all_rows(raw_input_rows):\n{body}\n"
    chunks = chunk_resource("p", _res("big.py", "text/x-python"), code)
    subs = [c for c in chunks if c.context_path == "process_all_rows"]
    assert len(subs) > 1, "expected the oversized function to split"
    for sub in subs:
        assert sub.signature == "def process_all_rows(raw_input_rows):"
        assert sub.symbol_id == subs[0].symbol_id
        assert sub.signature in sub.text_for_embedding()


def test_embed_text_capped_but_content_untruncated() -> None:
    from anvay.ingest.models import EMBED_CHAR_CAP, Chunk, ResourceRef

    long_content = "x" * 5000
    c = Chunk(
        product_id="p",
        resource=ResourceRef(source_id="local:test", uri="a.py", mime="text/x-python"),
        content=long_content,
        start_line=1,
        end_line=1,
        kind=ChunkKind.CODE,
        context_path="ctx",
        signature="def f():",
    )
    assert len(c.content) == 5000
    assert len(c.text_for_embedding()) <= EMBED_CHAR_CAP


def test_embed_text_caps_oversized_header() -> None:
    from anvay.ingest.models import EMBED_CHAR_CAP, Chunk, ResourceRef

    c = Chunk(
        product_id="p",
        resource=ResourceRef(source_id="local:test", uri="a.py", mime="text/x-python"),
        content="body",
        start_line=1,
        end_line=1,
        kind=ChunkKind.CODE,
        context_path="ctx",
        signature="def f():",
        context_summary="s" * (EMBED_CHAR_CAP + 500),
    )
    assert len(c.text_for_embedding()) == EMBED_CHAR_CAP


def test_decorated_python_signature_uses_definition_header() -> None:
    chunks = chunk_resource(
        "p",
        _res("decorated.py", "text/x-python"),
        "@route('/items')\ndef list_items():\n"
        "    values = [str(i) for i in range(50)]\n"
        "    return ','.join(values)\n",
    )
    fn = next(c for c in chunks if c.context_path == "list_items")
    assert fn.signature == "def list_items():"


def test_sparse_text_decorates_identifiers() -> None:
    from anvay.ingest.models import Chunk, ResourceRef

    c = Chunk(
        product_id="p",
        resource=ResourceRef(source_id="local:test", uri="a.py", mime="text/x-python"),
        content="def getUserById(user_id):\n    return fetch_user_record(user_id)",
        start_line=1,
        end_line=2,
        kind=ChunkKind.CODE,
    )
    sparse = c.sparse_text_for_embedding()
    assert c.content in sparse
    tail = sparse[len(c.text_for_embedding()):]
    for word in ("get", "user", "by", "id", "fetch", "record"):
        assert word in tail.split()
