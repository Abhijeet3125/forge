# core/tools/git_commit.py

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Any
from langchain_core.tools import tool


@dataclass
class GitCommitResult:
    success: bool
    commit_hash: str | None = None
    message: str = ""
    files_committed: list[str] = field(default_factory=list)
    error: str | None = None


def apply_sandbox_to_real_repo(
    repo_path: str,
    sandbox_path: str,
    files_touched: list[str],
) -> tuple[bool, str | None]:
    """
    Copies the tested, verified files from the sandbox copy into the real repo.
    """
    real_root = Path(repo_path).resolve()
    sb_root = Path(sandbox_path).resolve()

    if not real_root.exists():
        return False, f"Real repository path does not exist: {repo_path}"

    try:
        for rel_file in files_touched:
            src_file = sb_root / rel_file
            dest_file = real_root / rel_file

            if src_file.exists():
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
        return True, None
    except Exception as e:
        return False, f"Failed to copy files from sandbox to real repository: {e}"


def _git_commit_impl(
    repo_path: str,
    message: str,
    files_to_add: list[str] | None = None,
) -> GitCommitResult:
    """
    Executes git add and git commit on the real repository.
    Only called post-human-approval.
    """
    root = Path(repo_path).resolve()
    if not (root / ".git").exists():
        return GitCommitResult(
            success=False,
            error=f"Not a git repository: {repo_path}",
        )

    try:
        # 1. Stage files
        if files_to_add:
            add_cmd = ["git", "add"] + files_to_add
        else:
            add_cmd = ["git", "add", "-A"]

        add_proc = subprocess.run(
            add_cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if add_proc.returncode != 0:
            return GitCommitResult(
                success=False,
                error=f"git add failed: {add_proc.stderr or add_proc.stdout}",
            )

        # Check if there are changes to commit
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not status_proc.stdout.strip():
            return GitCommitResult(
                success=True,
                message="No staged changes to commit.",
                files_committed=[],
            )

        # 2. Commit with fallback author info if not globally set
        commit_cmd = [
            "git",
            "-c", "user.name=Forge Coding Agent",
            "-c", "user.email=forge-agent@local",
            "commit",
            "-m", message,
        ]
        commit_proc = subprocess.run(
            commit_cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if commit_proc.returncode != 0:
            return GitCommitResult(
                success=False,
                error=f"git commit failed: {commit_proc.stderr or commit_proc.stdout}",
            )

        # 3. Retrieve commit hash
        hash_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_hash = hash_proc.stdout.strip() if hash_proc.returncode == 0 else "unknown"

        staged_files = [line.split()[-1] for line in status_proc.stdout.splitlines() if line.strip()]

        return GitCommitResult(
            success=True,
            commit_hash=commit_hash,
            message=message,
            files_committed=staged_files,
        )

    except Exception as e:
        return GitCommitResult(
            success=False,
            error=f"Git commit failed: {e}",
        )


def make_git_commit_tool(repo_path: str):
    """
    Factory creating git_commit tool.
    Per Section 8: acts only on real repo, post-approval only.
    """

    @tool
    def git_commit(message: str) -> str:
        """
        Commit verified changes to the real repository.
        Only invoked post-approval.

        Args:
            message: git commit message describing the fix
        """
        result = _git_commit_impl(repo_path=repo_path, message=message)
        if result.success:
            return f"Successfully committed changes as {(result.commit_hash or '')[:7]}: '{result.message}'"
        return f"ERROR: {result.error}"

    return git_commit
