# core/retrieval/chunking.py

import ast
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

LARGE_FUNCTION_LINE_THRESHOLD = 40


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    name: str
    kind: str  # "function", "class", "method", "module", "summary"
    start_line: int
    end_line: int
    content: str
    signature: str | None = None
    docstring: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Converts chunk metadata to a flat dict suitable for ChromaDB."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "name": self.name,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature or "",
            "docstring": self.docstring or "",
        }

    @classmethod
    def from_metadata(cls, content: str, meta: Mapping[str, Any]) -> "CodeChunk":
        """Recreates a CodeChunk from ChromaDB document and metadata."""
        return cls(
            chunk_id=meta.get("chunk_id", ""),
            file_path=meta.get("file_path", ""),
            name=meta.get("name", ""),
            kind=meta.get("kind", ""),
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
            content=content,
            signature=meta.get("signature") or None,
            docstring=meta.get("docstring") or None,
        )


def _generate_chunk_id(file_path: str, name: str, start_line: int, end_line: int, kind: str) -> str:
    seed = f"{file_path}:{name}:{kind}:{start_line}-{end_line}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"{file_path}:{name}:{start_line}_{digest}"


# =====================================================================
# AST Parser (Built-in Python fallback + primary AST extractor)
# =====================================================================

def _chunk_python_ast(code: str, file_path: str) -> list[CodeChunk]:
    """
    Parses Python code using Python's built-in `ast` module.
    Extracts classes, functions, methods, and separate summary chunks
    for large functions as specified in Section 3.2.
    """
    chunks: list[CodeChunk] = []
    lines = code.splitlines(keepends=True)
    total_lines = len(lines)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If file has syntax errors, fall back to whole-file chunk
        return [
            CodeChunk(
                chunk_id=_generate_chunk_id(file_path, "module", 1, total_lines, "module"),
                file_path=file_path,
                name="module",
                kind="module",
                start_line=1,
                end_line=max(1, total_lines),
                content=code,
            )
        ]

    module_doc = ast.get_docstring(tree)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_extract_function_chunks(node, lines, file_path))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_extract_class_chunks(node, lines, file_path))

    # If no functions or classes were found (e.g. script/config), chunk whole file or sections
    if not chunks and code.strip():
        chunks.append(
            CodeChunk(
                chunk_id=_generate_chunk_id(file_path, "module", 1, total_lines, "module"),
                file_path=file_path,
                name="module",
                kind="module",
                start_line=1,
                end_line=max(1, total_lines),
                content=code,
                docstring=module_doc,
            )
        )

    return chunks


def _get_node_source(node: ast.AST, lines: list[str]) -> str:
    start_line = getattr(node, "lineno", 1) - 1
    end_line = getattr(node, "end_lineno", len(lines))
    return "".join(lines[start_line:end_line])


def _extract_function_chunks(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    file_path: str,
    parent_class: str | None = None,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line)
    func_content = _get_node_source(node, lines)

    docstring = ast.get_docstring(node)
    func_name = f"{parent_class}.{node.name}" if parent_class else node.name
    kind = "method" if parent_class else "function"

    # Extract signature (first line up to colon)
    sig_line = lines[start_line - 1].strip() if start_line <= len(lines) else f"def {node.name}(...):"

    # Primary chunk (full body)
    chunks.append(
        CodeChunk(
            chunk_id=_generate_chunk_id(file_path, func_name, start_line, end_line, kind),
            file_path=file_path,
            name=func_name,
            kind=kind,
            start_line=start_line,
            end_line=end_line,
            content=func_content,
            signature=sig_line,
            docstring=docstring,
        )
    )

    # For large functions: generate a separate summary chunk
    line_count = end_line - start_line + 1
    if line_count > LARGE_FUNCTION_LINE_THRESHOLD:
        summary_lines = [sig_line]
        if docstring:
            summary_lines.append(f'    """{docstring}"""')
        # Include first 5 body lines for context
        body_sample = lines[start_line : min(start_line + 5, end_line)]
        summary_lines.extend([line.rstrip() for line in body_sample if line.strip()])
        summary_content = "\n".join(summary_lines)

        chunks.append(
            CodeChunk(
                chunk_id=_generate_chunk_id(file_path, func_name, start_line, end_line, "summary"),
                file_path=file_path,
                name=f"{func_name}:summary",
                kind="summary",
                start_line=start_line,
                end_line=end_line,
                content=summary_content,
                signature=sig_line,
                docstring=docstring,
            )
        )

    return chunks


def _extract_class_chunks(node: ast.ClassDef, lines: list[str], file_path: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    start_line = node.lineno
    end_line = getattr(node, "end_lineno", start_line)
    class_content = _get_node_source(node, lines)
    docstring = ast.get_docstring(node)

    class_sig = lines[start_line - 1].strip() if start_line <= len(lines) else f"class {node.name}:"

    # Class chunk
    chunks.append(
        CodeChunk(
            chunk_id=_generate_chunk_id(file_path, node.name, start_line, end_line, "class"),
            file_path=file_path,
            name=node.name,
            kind="class",
            start_line=start_line,
            end_line=end_line,
            content=class_content,
            signature=class_sig,
            docstring=docstring,
        )
    )

    # Extract individual methods
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_extract_function_chunks(item, lines, file_path, parent_class=node.name))

    return chunks


# =====================================================================
# Tree-Sitter Integration (with graceful fallback to AST)
# =====================================================================

def _chunk_with_tree_sitter(code: str, file_path: str, language: str) -> list[CodeChunk] | None:
    """Attempts to chunk using tree-sitter-language-pack if available."""
    try:
        import tree_sitter_language_pack as tslp

        parser = tslp.get_parser(language)
        tree = parser.parse(code.encode("utf-8"))
        root = tree.root_node

        chunks: list[CodeChunk] = []
        lines = code.splitlines(keepends=True)

        for child in root.children:
            if child.type in ("function_definition", "class_definition", "method_definition"):
                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1
                snippet = "".join(lines[start_line - 1 : end_line])

                name = "unknown"
                for sub in child.children:
                    if sub.type == "identifier":
                        name = code.encode("utf-8")[sub.start_byte : sub.end_byte].decode("utf-8", errors="ignore")
                        break

                kind = "class" if "class" in child.type else "function"
                chunks.append(
                    CodeChunk(
                        chunk_id=_generate_chunk_id(file_path, name, start_line, end_line, kind),
                        file_path=file_path,
                        name=name,
                        kind=kind,
                        start_line=start_line,
                        end_line=end_line,
                        content=snippet,
                    )
                )

        return chunks if chunks else None

    except Exception:
        # Return None so callers fall back to Python AST or generic chunking
        return None


def chunk_file_content(code: str, file_path: str) -> list[CodeChunk]:
    """
    Main chunking entry point for a single file's content.
    Uses tree-sitter or AST for Python, and tree-sitter for other languages.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        # Python built-in AST provides exact symbols, docstrings, method nesting, and summary chunks
        ast_chunks = _chunk_python_ast(code, file_path)
        if ast_chunks:
            return ast_chunks
        ts_chunks = _chunk_with_tree_sitter(code, file_path, "python")
        if ts_chunks:
            return ts_chunks

    # For JavaScript / TypeScript / Go / Rust
    lang_map = {
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }
    lang = lang_map.get(ext)
    if lang:
        ts_chunks = _chunk_with_tree_sitter(code, file_path, lang)
        if ts_chunks:
            return ts_chunks

    # Fallback: whole file chunk
    lines = code.splitlines()
    total_lines = max(1, len(lines))
    return [
        CodeChunk(
            chunk_id=_generate_chunk_id(file_path, "file", 1, total_lines, "file"),
            file_path=file_path,
            name=Path(file_path).name,
            kind="file",
            start_line=1,
            end_line=total_lines,
            content=code,
        )
    ]


def chunk_file(file_path: Path | str, repo_root: Path | str | None = None) -> list[CodeChunk]:
    """Reads and chunks a file from disk, relativizing file_path to repo_root if given."""
    path = Path(file_path)
    if not path.is_file():
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(path.relative_to(repo_root)) if repo_root else str(path)
    return chunk_file_content(content, rel_path)
