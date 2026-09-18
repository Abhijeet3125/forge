# core/tools/read_file.py

from dataclasses import dataclass
from pathlib import Path
from langchain_core.tools import tool

MAX_FILE_SIZE_BYTES = 200_000  # ~200KB cap; large enough for almost any source file


@dataclass
class ReadFileResult:
    success: bool
    content: str | None = None
    error: str | None = None
    truncated: bool = False


def _read_file_impl(path: str, repo_root: str, with_line_numbers: bool = True) -> ReadFileResult:
    """Core logic, kept separate from the tool wrapper so it's also unit-testable directly."""
    try:
        root = Path(repo_root).resolve()
        p = Path(path.strip())

        # If absolute path already inside root, use it; otherwise treat as relative to root
        if p.is_absolute() and p.resolve().is_relative_to(root):
            target = p.resolve()
        else:
            target = (root / path.strip().lstrip("/\\")).resolve()

        # If target does not exist, check if stripping leading path segments
        # (e.g. repo name prefix like "mock_repo/src/cart.py" -> "src/cart.py") resolves to an existing file in root
        if not target.exists():
            rel_parts = Path(path.strip().lstrip("/\\")).parts
            for i in range(1, len(rel_parts)):
                candidate = (root / Path(*rel_parts[i:])).resolve()
                if candidate.exists() and candidate.is_file() and candidate.is_relative_to(root):
                    target = candidate
                    break

        if not target.is_relative_to(root):
            return ReadFileResult(success=False, error=f"Path '{path}' is outside the repo root.")

        if not target.exists():
            return ReadFileResult(success=False, error=f"File not found: {path}")

        if not target.is_file():
            return ReadFileResult(success=False, error=f"Not a file: {path}")

        size = target.stat().st_size
        truncated = size > MAX_FILE_SIZE_BYTES

        with open(target, "rb") as f:
            raw_bytes = f.read(MAX_FILE_SIZE_BYTES) if truncated else f.read()

        # Binary file check
        if b"\x00" in raw_bytes:
            return ReadFileResult(success=False, error=f"File appears to be binary: {path}")

        raw = raw_bytes.decode("utf-8")

        if not raw:
            content = "(empty file)"
        elif with_line_numbers:
            lines = raw.splitlines()
            content = "\n".join(f"{i+1:>5}\t{line}" for i, line in enumerate(lines))
        else:
            content = raw

        if truncated:
            content += f"\n\n[... truncated, file exceeds {MAX_FILE_SIZE_BYTES} bytes ...]"

        return ReadFileResult(success=True, content=content, truncated=truncated)

    except UnicodeDecodeError:
        return ReadFileResult(success=False, error=f"File is not valid UTF-8 text (likely binary): {path}")
    except PermissionError:
        return ReadFileResult(success=False, error=f"Permission denied: {path}")
    except Exception as e:
        return ReadFileResult(success=False, error=f"Unexpected error reading '{path}': {e}")


def make_read_file_tool(repo_root: str):
    """
    Factory that binds `repo_root` into the tool, since LangGraph tools
    are called by the LLM with only the arguments it decides to pass —
    we don't want the agent supplying its own repo_root.
    """

    @tool
    def read_file(path: str) -> str:
        """
        Read the contents of a file within the repository, with line numbers.
        Use this to inspect existing code before writing a fix.

        Args:
            path: file path relative to the repository root, e.g. "src/cart.py"
        """
        result = _read_file_impl(path, repo_root=repo_root)
        if result.success:
            return result.content or ""
        return f"ERROR: {result.error}"

    return read_file