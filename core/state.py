# core/state.py

from datetime import datetime, timezone
import operator
from typing import Annotated, Literal, TypedDict

from core.retrieval.chunking import CodeChunk

TaskStatus = Literal[
    "planning",
    "retrieving",
    "coding",
    "testing",
    "retrying",
    "awaiting_approval",
    "done",
    "failed",
]


class LogEntry(TypedDict):
    timestamp: str
    node: str
    message: str
    level: Literal["info", "warning", "error", "success"]


class TestResultDict(TypedDict):
    passed: bool
    summary: str
    failing_tests: list[str]


class TaskState(TypedDict):
    # input
    user_request: str
    repo_path: str
    sandbox_path: str | None  # isolated working copy (Docker mount or local clone)

    # planning
    plan: list[str]  # ordered list of steps
    current_step_index: int

    # retrieval
    retrieved_context: list[CodeChunk]  # top-k reranked chunks for current step

    # coding
    proposed_diff: str | None  # patch-style output from Coder
    files_touched: list[str]

    # testing
    test_result: TestResultDict | None  # {passed: bool, summary: str, failing_tests: list[str]}
    retry_count: int
    max_retries: int  # e.g. 3

    # human-in-the-loop approval gate
    user_approved: bool | None

    # control & streaming
    status: TaskStatus
    logs: Annotated[list[LogEntry], operator.add]  # automatically appends new entries in LangGraph


def make_log_entry(
    node: str,
    message: str,
    level: Literal["info", "warning", "error", "success"] = "info",
) -> LogEntry:
    """Helper to create a timestamped LogEntry."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "message": message,
        "level": level,
    }


def create_initial_state(
    user_request: str,
    repo_path: str,
    max_retries: int = 3,
) -> TaskState:
    """Helper to create a fresh, valid initial TaskState."""
    return {
        "user_request": user_request,
        "repo_path": repo_path,
        "sandbox_path": None,
        "plan": [],
        "current_step_index": 0,
        "retrieved_context": [],
        "proposed_diff": None,
        "files_touched": [],
        "test_result": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "user_approved": None,
        "status": "planning",
        "logs": [
            make_log_entry(
                node="orchestrator",
                message=f"Initialized task for repo '{repo_path}'.",
                level="info",
            )
        ],
    }