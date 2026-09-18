# core/tools/run_tests.py

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from langchain_core.tools import tool

DEFAULT_TEST_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 100_000


@dataclass
class TestResult:
    passed: bool
    summary: str
    failing_tests: list[str]
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    compact_failure: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Converts to TaskState test_result schema."""
        return {
            "passed": self.passed,
            "summary": self.summary,
            "failing_tests": self.failing_tests,
        }


def detect_test_command(repo_root: str) -> str:
    """
    Detects the appropriate test command from the repository configuration
    (pyproject.toml, uv.lock, package.json, Cargo.toml, go.mod, Makefile, etc.).
    """
    root = Path(repo_root).resolve()

    # 1. Python projects
    if (root / "uv.lock").exists():
        return "uv run pytest"

    if (
        (root / "pyproject.toml").exists()
        or (root / "pytest.ini").exists()
        or (root / "setup.cfg").exists()
        or (root / "tox.ini").exists()
    ):
        import shutil
        import sys
        if shutil.which("pytest"):
            return "pytest"
        if shutil.which("uv"):
            return "uv run pytest"
        return f"{sys.executable} -m pytest"

    # 2. Node.js / JavaScript / TypeScript projects
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "test" in data.get("scripts", {}):
                    return "npm test"
        except Exception:
            return "npm test"

    # 3. Rust projects
    if (root / "Cargo.toml").exists():
        return "cargo test"

    # 4. Go projects
    if (root / "go.mod").exists():
        return "go test ./..."

    # 5. Makefile with a test target
    makefile = root / "Makefile"
    if makefile.exists():
        try:
            with open(makefile, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if re.search(r"^test\s*:", content, re.MULTILINE):
                    return "make test"
        except Exception:
            pass

    # 6. Fallback if test folders exist
    if (root / "tests").is_dir() or (root / "test").is_dir():
        return "pytest"

    return "pytest"


def _extract_pytest_summary(
    stdout: str,
    stderr: str,
    target_files: list[str] | None = None,
) -> tuple[list[str], str, str | None]:
    """
    Deterministically parses test output to extract:
    - failing_tests: list of failed test identifiers
    - summary_line: concise status (e.g. '1 failed, 3 passed in 0.2s')
    - compact_failure: key assertion and line for small-model context compaction
    """
    combined = stdout + "\n" + stderr
    failing_tests: list[str] = []
    summary_line = ""
    compact_failure = None

    # Extract failing test names from pytest (e.g. FAILED tests/test_cart.py::test_negative)
    # Pytest lines formatted as: FAILED tests/test_cart.py::test_name - ... or FAILED tests/test_cart.py::test_name
    for match in re.finditer(r"^FAILED\s+(\S+?)(?:\s+-\s+|\s*$)", combined, re.MULTILINE):
        test_id = match.group(1).strip()
        if test_id and test_id not in failing_tests:
            failing_tests.append(test_id)

    # Standard unittest patterns (FAIL: test_discount (__main__.TestCart.test_discount))
    for match in re.finditer(r"^(?:FAIL|ERROR):\s+([^\s\(\n]+)", combined, re.MULTILINE):
        test_id = match.group(1).strip()
        if test_id and test_id not in failing_tests:
            failing_tests.append(test_id)

    # Extract summary line
    # 1. Pytest summary: e.g. "=== 1 failed, 4 passed in 0.12s ==="
    summary_matches = list(re.finditer(r"=+\s+([0-9]+\s+(?:failed|passed|error)[^=]+)=+", combined))
    if summary_matches:
        summary_line = summary_matches[-1].group(1).strip()
    # 2. Unittest summary: e.g. "FAILED (failures=1)" or "OK"
    elif "FAILED (" in combined:
        m = re.search(r"FAILED\s*\((.*?)\)", combined)
        summary_line = f"FAILED ({m.group(1)})" if m else "Tests failed"
    elif failing_tests:
        summary_line = f"{len(failing_tests)} test(s) failed"
    elif "PASSED" in combined or "OK" in combined:
        summary_line = "All tests passed"
    else:
        summary_line = "Tests executed"

    # Extract compacted failure (assertion + relevant file:line)
    # Pytest failure blocks: e.g. ________________ test_foo ________________
    failure_blocks = list(re.finditer(r"_{3,}\s+(.+?)\s+_{3,}(.+?)(?=(?:_{3,}|\Z|=+))", combined, re.DOTALL))
    if failure_blocks:
        selected_block = failure_blocks[0]
        # Prioritize failure block that matches target_files or files being modified
        if target_files:
            target_stems = {
                Path(f).stem.lower().replace("test_", "")
                for f in target_files
                if f and Path(f).stem.lower().replace("test_", "")
            }
            for fb in failure_blocks:
                header = fb.group(1).lower()
                body = fb.group(2).lower()
                if any(stem in header or stem in body for stem in target_stems):
                    selected_block = fb
                    break

        title = selected_block.group(1).strip()
        snippet = selected_block.group(2).strip()
        lines = [
            line.strip()
            for line in snippet.splitlines()
            if line.strip().startswith(">")
            or line.strip().startswith("E ")
            or line.strip().startswith("E\t")
            or "Error" in line
            or "Failed" in line
            or re.search(r":\d+:\s+", line)
        ]
        if lines:
            compact_failure = f"{title}: " + " | ".join(lines[:3])
        else:
            compact_failure = f"{title}: " + " | ".join(snippet.splitlines()[-3:])
    # Unittest failure block
    elif "AssertionError" in combined or "Error:" in combined:
        match_err = re.search(r"(FAIL:\s+[^\n]+.*?AssertionError:[^\n]+)", combined, re.DOTALL)
        if match_err:
            compact_failure = match_err.group(1).strip()
        else:
            matches = [
                line.strip()
                for line in combined.splitlines()
                if "AssertionError" in line or "Error:" in line or line.startswith("FAIL:")
            ]
            if matches:
                compact_failure = "\n".join(matches[:4])

    return failing_tests, summary_line, compact_failure


def _run_tests_impl(
    repo_root: str,
    test_cmd: str | None = None,
    timeout: int = DEFAULT_TEST_TIMEOUT_SECONDS,
    target_files: list[str] | None = None,
) -> TestResult:
    """Core test execution logic, detecting the command and parsing the results."""
    root = Path(repo_root).resolve()
    if not root.exists() or not root.is_dir():
        return TestResult(
            passed=False,
            summary="Invalid repository directory",
            failing_tests=[],
            error=f"Directory does not exist: {repo_root}",
        )

    cmd = test_cmd or detect_test_command(str(root))

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # If command exited with 127 (command not found) and no test_cmd override was given,
        # try fallback runners (e.g. uv run pytest or sys.executable -m pytest)
        if proc.returncode == 127 and not test_cmd and "pytest" in cmd:
            import sys
            for fallback in ["uv run pytest", f"{sys.executable} -m pytest"]:
                if fallback != cmd:
                    try:
                        f_proc = subprocess.run(
                            fallback,
                            shell=True,
                            cwd=str(root),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        if f_proc.returncode != 127:
                            proc = f_proc
                            cmd = fallback
                            break
                    except Exception:
                        pass

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        passed = (proc.returncode == 0)

        failing_tests, summary, compact_failure = _extract_pytest_summary(
            stdout, stderr, target_files=target_files
        )

        # If returncode is non-zero but summary reported nothing, ensure failure is indicated
        if not passed and not failing_tests and not summary:
            summary = f"Test command failed with exit code {proc.returncode}"

        return TestResult(
            passed=passed,
            summary=summary,
            failing_tests=failing_tests,
            exit_code=proc.returncode,
            stdout=stdout[:MAX_OUTPUT_BYTES],
            stderr=stderr[:MAX_OUTPUT_BYTES],
            command=cmd,
            compact_failure=compact_failure,
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            summary=f"Tests timed out after {timeout} seconds",
            failing_tests=[],
            exit_code=-1,
            command=cmd,
            error=f"Timeout expired running: {cmd}",
        )
    except Exception as e:
        return TestResult(
            passed=False,
            summary="Execution error",
            failing_tests=[],
            exit_code=-1,
            command=cmd,
            error=f"Failed to execute tests: {e}",
        )


def make_run_tests_tool(
    repo_root: str,
    test_cmd: str | None = None,
    timeout: int = DEFAULT_TEST_TIMEOUT_SECONDS,
):
    """
    Factory that creates the run_tests tool for the Tester agent.
    Binds repo_root and optional test command override.
    """

    @tool
    def run_tests() -> str:
        """
        Run the test suite for the current repository and return a structured report.
        Automatically detects the repository's test runner (e.g. pytest, npm test).
        """
        result = _run_tests_impl(repo_root=repo_root, test_cmd=test_cmd, timeout=timeout)

        if result.error:
            return f"ERROR running tests with command '{result.command}': {result.error}"

        status_str = "PASSED" if result.passed else "FAILED"
        report = [
            f"Test Suite: {status_str}",
            f"Command: {result.command}",
            f"Summary: {result.summary}",
        ]

        if result.failing_tests:
            report.append("Failing tests:")
            for test in result.failing_tests:
                report.append(f"  - {test}")

        if result.compact_failure:
            report.append(f"\nCompacted Failure Details:\n{result.compact_failure}")

        return "\n".join(report)

    return run_tests


def make_sandboxed_run_tests_tool(
    session,
    runner=None,
    test_cmd: str | None = None,
    timeout: int = DEFAULT_TEST_TIMEOUT_SECONDS,
):
    """
    Factory creating a run_tests tool that executes tests strictly inside
    an isolated Docker container or sandbox copy.
    """

    @tool
    def run_tests() -> str:
        """
        Run the test suite inside the isolated sandbox environment.
        Automatically detects test runner or executes specified test command.
        """
        if runner:
            result = runner.run_tests(session=session, test_cmd=test_cmd, timeout=timeout)
        else:
            result = _run_tests_impl(repo_root=session.sandbox_path, test_cmd=test_cmd, timeout=timeout)

        if result.error:
            return f"ERROR running tests in sandbox with '{result.command}': {result.error}"

        status_str = "PASSED" if result.passed else "FAILED"
        report = [
            f"Sandbox Test Suite: {status_str}",
            f"Command: {result.command}",
            f"Summary: {result.summary}",
        ]

        if result.failing_tests:
            report.append("Failing tests:")
            for test in result.failing_tests:
                report.append(f"  - {test}")

        if result.compact_failure:
            report.append(f"\nCompacted Failure Details:\n{result.compact_failure}")

        return "\n".join(report)

    return run_tests

