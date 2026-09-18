# core/agents/tester.py

from typing import Any

from core.llm.provider import LLMMessage, LLMProvider
from core.state import LogEntry, TaskState, TestResultDict, make_log_entry
from core.tools.run_tests import _run_tests_impl

COMPACT_FAILURE_PROMPT = """You are a software testing assistant.
Summarize the following test failure in 1-3 lines focusing strictly on:
1. The failing test name
2. The specific assertion error or exception that occurred
3. The file and line number if visible

Do NOT include stack traces or full logs. Keep it compact.
Test Output:
"""


class TesterAgent:
    """
    Tester Agent (Section 3.4):
    Runs the repo's test suite and evaluates the results.
    Deterministic execution and failure compaction; optionally uses LLM
    to compact complex tracebacks when deterministic extraction is insufficient.
    """

    def __init__(self, llm: LLMProvider | None = None, test_cmd: str | None = None):
        self.llm = llm
        self.test_cmd = test_cmd

    def _compact_failure_with_llm(self, stdout: str, stderr: str) -> str:
        """Uses LLM to compact lengthy test tracebacks into a concise 1-2 line summary."""
        if not self.llm:
            return ""
        try:
            combined = (stdout + "\n" + stderr)[:4000]
            messages = [
                LLMMessage(role="system", content="You are a test output compaction assistant."),
                LLMMessage(role="user", content=COMPACT_FAILURE_PROMPT + combined),
            ]
            response = self.llm.generate(messages, temperature=0.0, max_tokens=150)
            return response.content.strip()
        except Exception:
            return ""

    def run(self, state: TaskState) -> dict[str, Any]:
        """
        Executes the Tester node:
        1. Runs the test suite in repo_path.
        2. Parses results into TestResultDict.
        3. Compacts failure log if tests failed.
        4. Determines next state (awaiting_approval, retrying, or failed).
        """
        repo_path = state.get("sandbox_path") or state["repo_path"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        files_touched = state.get("files_touched", [])
        # Run test suite
        res = _run_tests_impl(repo_root=repo_path, test_cmd=self.test_cmd, target_files=files_touched)

        # Prepare summary & compaction
        summary = res.summary
        if not res.passed:
            if res.compact_failure:
                summary = f"{res.summary} - {res.compact_failure}"
            elif self.llm:
                llm_compact = self._compact_failure_with_llm(res.stdout, res.stderr)
                if llm_compact:
                    summary = f"{res.summary} - {llm_compact}"

        test_result_dict: TestResultDict = {
            "passed": res.passed,
            "summary": summary,
            "failing_tests": res.failing_tests,
        }

        # Check plan progress
        plan = state.get("plan") or []
        current_step = state.get("current_step_index", 0)
        has_more_steps = len(plan) > 0 and (current_step + 1 < len(plan))

        user_req = (state.get("user_request") or "").lower()
        is_bug_fix = any(
            kw in user_req
            for kw in ("fix", "bug", "error", "fail", "issue", "pass", "broken", "exception", "test_")
        )

        if res.passed:
            # For bug fixes or once all plan steps are done, pass immediately to approval gate
            if is_bug_fix or not has_more_steps:
                next_status = "awaiting_approval"
                log_entry = make_log_entry(
                    node="tester",
                    message=f"All tests PASSED: {summary}. Task goal achieved! Awaiting approval.",
                    level="success",
                )
                return {
                    "test_result": test_result_dict,
                    "status": next_status,
                    "logs": [log_entry],
                }
            else:
                next_status = "retrieving"
                log_entry = make_log_entry(
                    node="tester",
                    message=f"Step {current_step + 1} tests PASSED: {summary}. Advancing to next step.",
                    level="success",
                )
                return {
                    "test_result": test_result_dict,
                    "current_step_index": current_step + 1,
                    "retry_count": 0,
                    "status": next_status,
                    "logs": [log_entry],
                }

        # If tests failed:
        if retry_count < max_retries:
            next_status = "retrying"
            new_retry_count = retry_count + 1
            log_entry = make_log_entry(
                node="tester",
                message=(
                    f"Tests FAILED: {summary}. "
                    f"Routing back to Coder (Attempt {new_retry_count}/{max_retries})."
                ),
                level="warning",
            )
            return {
                "test_result": test_result_dict,
                "retry_count": new_retry_count,
                "status": next_status,
                "logs": [log_entry],
            }
        else:
            next_status = "failed"
            log_entry = make_log_entry(
                node="tester",
                message=f"Tests FAILED and retries exhausted ({max_retries}/{max_retries}). Stopping task.",
                level="error",
            )
            return {
                "test_result": test_result_dict,
                "status": next_status,
                "logs": [log_entry],
            }


def make_tester_node(llm: LLMProvider | None = None, test_cmd: str | None = None):
    """Factory creating a LangGraph node function for the Tester agent."""
    agent = TesterAgent(llm=llm, test_cmd=test_cmd)

    def tester_node(state: TaskState) -> dict[str, Any]:
        return agent.run(state)

    return tester_node
