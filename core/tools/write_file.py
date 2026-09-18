# core/tools/write_file.py

from dataclasses import dataclass
import difflib
from pathlib import Path
from langchain_core.tools import tool


@dataclass
class WriteFileResult:
    success: bool
    path: str = ""
    bytes_written: int = 0
    is_new_file: bool = False
    diff: str | None = None
    error: str | None = None


def _write_file_impl(path: str, content: str, repo_root: str) -> WriteFileResult:
    """Core file writing logic, safe against traversal and protected directories."""
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
            return WriteFileResult(success=False, path=path, error=f"Path '{path}' is outside the repo root.")

        # Prevent touching internal git directory
        try:
            rel = target.relative_to(root)
            if rel.parts and rel.parts[0] == ".git":
                return WriteFileResult(success=False, path=path, error="Modifying .git directory is forbidden.")
        except ValueError:
            return WriteFileResult(success=False, path=path, error=f"Invalid path relative to repo root: {path}")

        if target.exists() and target.is_dir():
            return WriteFileResult(success=False, path=path, error=f"Target path is an existing directory: {path}")

        is_new = not target.exists()
        old_content = ""
        if not is_new:
            try:
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except Exception:
                old_content = ""

        # Ensure parent directories exist
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write the new content as UTF-8
        with open(target, "w", encoding="utf-8") as f:
            bytes_written = f.write(content)

        # Compute unified diff
        rel_str = str(rel)
        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{rel_str}",
                tofile=f"b/{rel_str}",
            )
        )
        diff = "".join(diff_lines) if diff_lines else None

        return WriteFileResult(
            success=True,
            path=rel_str,
            bytes_written=bytes_written,
            is_new_file=is_new,
            diff=diff,
        )

    except PermissionError:
        return WriteFileResult(success=False, path=path, error=f"Permission denied: {path}")
    except Exception as e:
        return WriteFileResult(success=False, path=path, error=f"Unexpected error writing to '{path}': {e}")


def make_write_file_tool(repo_root: str):
    """
    Factory that binds `repo_root` into the tool.
    The LLM provides only the relative file path and new file content.
    """

    @tool
    def write_file(path: str, content: str) -> str:
        """
        Write or overwrite a file within the repository.
        Parent directories are automatically created if they do not exist.

        Args:
            path: file path relative to the repository root, e.g. "src/cart.py"
            content: complete text content to write to the file
        """
        result = _write_file_impl(path, content=content, repo_root=repo_root)
        if result.success:
            action = "Created" if result.is_new_file else "Updated"
            msg = f"{action} '{result.path}' ({result.bytes_written} characters written)."
            if result.diff:
                msg += f"\n\n--- Diff ---\n{result.diff}"
            return msg
        return f"ERROR: {result.error}"

    return write_file


def make_sandboxed_write_file_tool(session, runner=None):
    """
    Factory creating a write_file tool that writes strictly inside
    an isolated sandbox session (real repo remains untouched).
    """

    @tool
    def write_file(path: str, content: str) -> str:
        """
        Write or overwrite a file inside the isolated sandbox copy.
        The real repository remains untouched.

        Args:
            path: file path relative to the repository root, e.g. "src/cart.py"
            content: complete text content to write to the file
        """
        if runner:
            result = runner.write_file(session=session, path=path, content=content)
        else:
            result = _write_file_impl(path=path, content=content, repo_root=session.sandbox_path)

        if result.success:
            action = "Created" if result.is_new_file else "Updated"
            msg = f"{action} '{result.path}' in sandbox ({result.bytes_written} chars)."
            if result.diff:
                msg += f"\n\n--- Diff ---\n{result.diff}"
            return msg
        return f"ERROR: {result.error}"

    return write_file

