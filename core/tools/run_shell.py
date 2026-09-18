# core/tools/run_shell.py

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from langchain_core.tools import tool

MAX_OUTPUT_BYTES = 100_000  # 100KB output limit
DEFAULT_TIMEOUT_SECONDS = 60

# Allowlisted base commands and prefixes for sandboxed inspection/testing.
# Commands not starting with one of these will be rejected.
DEFAULT_ALLOWLIST = [
    # Testing & Python
    "pytest",
    "python",
    "python3",
    "uv run",
    "coverage",
    # JavaScript / TypeScript
    "npm",
    "npx",
    "yarn",
    "pnpm",
    # Rust & Go
    "cargo",
    "go",
    # Basic inspection & file tools
    "git status",
    "git diff",
    "git log",
    "ls",
    "find",
    "cat",
    "head",
    "tail",
    "grep",
    "echo",
    "which",
    "make",
]


@dataclass
class ShellResult:
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    truncated: bool = False
    error: str | None = None


def _is_command_allowed(cmd: str, allowlist: list[str]) -> bool:
    """Checks whether the given command string starts with an allowlisted command."""
    stripped = cmd.strip()
    return any(stripped == allowed or stripped.startswith(allowed + " ") for allowed in allowlist)


def _run_shell_impl(
    cmd: str,
    repo_root: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allowlist: list[str] | None = None,
) -> ShellResult:
    """
    Executes a shell command within repo_root if permitted by the allowlist.
    Limits execution time and output size to protect against runaway processes.
    """
    allowed = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
    cleaned_cmd = cmd.strip()

    if not _is_command_allowed(cleaned_cmd, allowed):
        return ShellResult(
            success=False,
            error=(
                f"Command '{cleaned_cmd.split()[0] if cleaned_cmd else ''}' is not permitted. "
                f"Allowed command prefixes: {', '.join(allowed[:8])}..."
            ),
        )

    root = Path(repo_root).resolve()
    if not root.exists() or not root.is_dir():
        return ShellResult(success=False, error=f"Working directory does not exist: {repo_root}")

    try:
        proc = subprocess.run(
            cleaned_cmd,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + (("\n" + stderr) if stderr else "")

        truncated = len(combined) > MAX_OUTPUT_BYTES
        if truncated:
            combined = combined[:MAX_OUTPUT_BYTES] + "\n\n[... output truncated ...]"

        return ShellResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=stdout[:MAX_OUTPUT_BYTES],
            stderr=stderr[:MAX_OUTPUT_BYTES],
            output=combined,
            truncated=truncated,
        )

    except subprocess.TimeoutExpired:
        return ShellResult(
            success=False,
            exit_code=-1,
            error=f"Command timed out after {timeout} seconds: {cleaned_cmd}",
        )
    except PermissionError:
        return ShellResult(
            success=False,
            error=f"Permission denied executing command: {cleaned_cmd}",
        )
    except Exception as e:
        return ShellResult(
            success=False,
            error=f"Unexpected error executing command: {e}",
        )


def make_run_shell_tool(
    repo_root: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allowlist: list[str] | None = None,
):
    """
    Factory that binds `repo_root` and security parameters into the tool.
    Intended for the Tester agent only.
    """

    @tool
    def run_shell(cmd: str) -> str:
        """
        Execute an allowlisted shell command within the repository.
        Available only for testing, inspecting files, and running build scripts.

        Args:
            cmd: the command line to run, e.g. "pytest tests/test_cart.py"
        """
        result = _run_shell_impl(cmd, repo_root=repo_root, timeout=timeout, allowlist=allowlist)
        if result.error:
            return f"ERROR: {result.error}"

        status_line = f"Exit code: {result.exit_code}"
        if not result.output.strip():
            return f"{status_line} (no output)"
        return f"{status_line}\n\n{result.output}"

    return run_shell


def make_sandboxed_run_shell_tool(
    session,
    runner=None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    allowlist: list[str] | None = None,
):
    """
    Factory creating a run_shell tool that executes strictly inside
    an isolated Docker container or sandbox copy.
    """

    @tool
    def run_shell(cmd: str) -> str:
        """
        Execute an allowlisted shell command within the sandbox environment.
        Restricted to testing and inspection commands.

        Args:
            cmd: the command line to run, e.g. "pytest tests/test_cart.py"
        """
        if runner:
            result = runner.run_command(session=session, cmd=cmd, timeout=timeout, allowlist=allowlist)
        else:
            result = _run_shell_impl(cmd, repo_root=session.sandbox_path, timeout=timeout, allowlist=allowlist)

        if result.error:
            return f"ERROR: {result.error}"

        status_line = f"Exit code: {result.exit_code}"
        if not result.output.strip():
            return f"{status_line} (no output)"
        return f"{status_line}\n\n{result.output}"

    return run_shell

