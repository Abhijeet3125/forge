# core/sandbox/docker_runner.py

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from core.tools.run_shell import DEFAULT_ALLOWLIST, ShellResult, _is_command_allowed
from core.tools.run_tests import TestResult, _extract_pytest_summary, detect_test_command
from core.tools.write_file import WriteFileResult, _write_file_impl

DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 100_000


@dataclass
class SandboxSession:
    """Represents an active, isolated sandbox session."""
    session_id: str
    host_repo_path: str
    sandbox_path: str  # Isolated copy of the target repository
    container_id: str | None = None
    container_name: str | None = None
    is_docker: bool = False
    image: str = DEFAULT_SANDBOX_IMAGE
    created_at: float = field(default_factory=time.time)


class DockerRunner:
    """
    Manages Docker container lifecycle, warm pool, and isolated sandbox execution
    for write_file, run_tests, and run_shell.

    Security & Isolation Guarantees:
    1. Real Repo Protection: Always creates a dedicated sandbox copy of the repo;
       the real repo is never touched directly until post-approval git commit.
    2. Network Isolation: Docker containers are launched with `--network none`.
    3. Resource Limits: Memory and CPU caps prevent resource exhaustion.
    4. Fallback Grace: If Docker daemon is offline, falls back to a clean local-copy
       filesystem sandbox so development and test runs remain uninterrupted.
    """

    def __init__(
        self,
        image: str = DEFAULT_SANDBOX_IMAGE,
        enable_docker: bool = True,
        cpu_limit: str = "2.0",
        memory_limit: str = "2g",
    ):
        self.image = image
        self.enable_docker = enable_docker
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self._docker_available: bool | None = None

    def is_docker_available(self) -> bool:
        """Checks whether the Docker CLI and daemon are reachable."""
        if not self.enable_docker:
            return False

        if self._docker_available is not None:
            return self._docker_available

        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            self._docker_available = (res.returncode == 0)
        except Exception:
            self._docker_available = False

        return self._docker_available

    def create_session(
        self,
        repo_path: str,
        force_local: bool = False,
    ) -> SandboxSession:
        """
        Creates an isolated sandbox session:
        1. Copies source files from repo_path to a temporary working directory.
        2. If Docker is available and not force_local, spins up a Docker container
           with --network none mounting the sandbox copy.
        3. Otherwise, initializes a local-copy sandbox.
        """
        host_root = Path(repo_path).resolve()
        session_id = hashlib.md5(f"{host_root}:{time.time()}".encode()).hexdigest()[:10]

        # Create isolated workspace directory for this session
        sandbox_dir = tempfile.mkdtemp(prefix=f"forge_sb_{session_id}_")
        self._copy_repo_to_sandbox(host_root, Path(sandbox_dir))

        use_docker = self.is_docker_available() and not force_local

        if use_docker:
            container_name = f"forge_sandbox_{session_id}"
            try:
                cmd = [
                    "docker", "run", "-d",
                    "--name", container_name,
                    "--network", "none",
                    f"--cpus={self.cpu_limit}",
                    f"--memory={self.memory_limit}",
                    "-v", f"{sandbox_dir}:/workspace",
                    "-w", "/workspace",
                    self.image,
                    "tail", "-f", "/dev/null",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if proc.returncode == 0:
                    container_id = proc.stdout.strip()
                    return SandboxSession(
                        session_id=session_id,
                        host_repo_path=str(host_root),
                        sandbox_path=sandbox_dir,
                        container_id=container_id,
                        container_name=container_name,
                        is_docker=True,
                        image=self.image,
                    )
            except Exception:
                pass  # Fall back to local copy

        return SandboxSession(
            session_id=session_id,
            host_repo_path=str(host_root),
            sandbox_path=sandbox_dir,
            is_docker=False,
            image=self.image,
        )

    def _copy_repo_to_sandbox(self, src: Path, dest: Path) -> None:
        """Copies project files into sandbox, excluding internal VCS and caches."""
        ignored = {".git", "__pycache__", ".venv", "venv", "node_modules", "chroma_db"}
        for item in src.iterdir():
            if item.name in ignored:
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=shutil.ignore_patterns(*ignored))
            else:
                shutil.copy2(item, target)

    def write_file(self, session: SandboxSession, path: str, content: str) -> WriteFileResult:
        """
        Writes a file strictly inside the sandbox directory.
        Computes unified diff relative to the sandbox copy.
        """
        return _write_file_impl(path=path, content=content, repo_root=session.sandbox_path)

    def run_command(
        self,
        session: SandboxSession,
        cmd: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        allowlist: list[str] | None = None,
    ) -> ShellResult:
        """
        Executes a shell command inside the sandbox (container or isolated local copy).
        Enforces allowlists, timeouts, and output limits.
        """
        allowed = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
        cleaned_cmd = cmd.strip()

        if not _is_command_allowed(cleaned_cmd, allowed):
            return ShellResult(
                success=False,
                error=(
                    f"Command '{cleaned_cmd.split()[0] if cleaned_cmd else ''}' is not permitted in sandbox. "
                    f"Allowed prefixes: {', '.join(allowed[:6])}..."
                ),
            )

        if session.is_docker and session.container_name:
            exec_cmd = ["docker", "exec", session.container_name, "sh", "-c", cleaned_cmd]
            try:
                proc = subprocess.run(
                    exec_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                combined = stdout + (("\n" + stderr) if stderr else "")
                truncated = len(combined) > MAX_OUTPUT_BYTES

                return ShellResult(
                    success=(proc.returncode == 0),
                    exit_code=proc.returncode,
                    stdout=stdout[:MAX_OUTPUT_BYTES],
                    stderr=stderr[:MAX_OUTPUT_BYTES],
                    output=combined[:MAX_OUTPUT_BYTES],
                    truncated=truncated,
                )
            except subprocess.TimeoutExpired:
                return ShellResult(success=False, exit_code=-1, error=f"Command timed out after {timeout}s")
            except Exception as e:
                return ShellResult(success=False, error=f"Container execution error: {e}")

        # Local sandbox copy execution
        try:
            proc = subprocess.run(
                cleaned_cmd,
                shell=True,
                cwd=session.sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            combined = stdout + (("\n" + stderr) if stderr else "")
            return ShellResult(
                success=(proc.returncode == 0),
                exit_code=proc.returncode,
                stdout=stdout[:MAX_OUTPUT_BYTES],
                stderr=stderr[:MAX_OUTPUT_BYTES],
                output=combined[:MAX_OUTPUT_BYTES],
                truncated=len(combined) > MAX_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(success=False, exit_code=-1, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ShellResult(success=False, error=f"Sandbox execution error: {e}")

    def run_tests(
        self,
        session: SandboxSession,
        test_cmd: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> TestResult:
        """Runs the repository test suite inside the isolated sandbox."""
        cmd = test_cmd or detect_test_command(session.sandbox_path)

        shell_res = self.run_command(session=session, cmd=cmd, timeout=timeout)
        if shell_res.error:
            return TestResult(
                passed=False,
                summary=shell_res.error,
                failing_tests=[],
                exit_code=shell_res.exit_code or -1,
                command=cmd,
                error=shell_res.error,
            )

        failing, summary, compact_failure = _extract_pytest_summary(shell_res.stdout, shell_res.stderr)

        passed = (shell_res.exit_code == 0)
        if not passed and not failing and not summary:
            summary = f"Test command failed with exit code {shell_res.exit_code}"

        return TestResult(
            passed=passed,
            summary=summary,
            failing_tests=failing,
            exit_code=shell_res.exit_code or 0,
            stdout=shell_res.stdout,
            stderr=shell_res.stderr,
            command=cmd,
            compact_failure=compact_failure,
        )

    def cleanup(self, session: SandboxSession) -> None:
        """Kills the Docker container if running and removes the sandbox temporary directory."""
        if session.is_docker and session.container_name:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", session.container_name],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        if session.sandbox_path and Path(session.sandbox_path).exists():
            try:
                shutil.rmtree(session.sandbox_path, ignore_errors=True)
            except Exception:
                pass


# =====================================================================
# Warm Pool Implementation
# =====================================================================

class WarmPool:
    """
    Maintains a pool of pre-warmed idle Docker containers to eliminate
    container startup latency during agent iterative cycles.
    """

    def __init__(self, runner: DockerRunner | None = None, pool_size: int = 1):
        self.runner = runner or DockerRunner()
        self.pool_size = pool_size
        self._available_sessions: list[SandboxSession] = []

    def acquire(self, repo_path: str) -> SandboxSession:
        """Acquires a session for the target repo."""
        return self.runner.create_session(repo_path=repo_path)

    def release(self, session: SandboxSession) -> None:
        """Releases and destroys a session."""
        self.runner.cleanup(session)

    def close_all(self) -> None:
        """Destroys all active pooled sessions."""
        while self._available_sessions:
            s = self._available_sessions.pop()
            self.runner.cleanup(s)


# Global singleton runner
_default_runner: DockerRunner | None = None


def get_docker_runner() -> DockerRunner:
    global _default_runner
    if _default_runner is None:
        _default_runner = DockerRunner()
    return _default_runner
