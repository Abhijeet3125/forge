# core/orchestrator.py

from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast
from langgraph.graph import END, StateGraph

from core.agents.coder import CoderAgent, make_coder_node
from core.agents.planner import PlannerAgent, make_planner_node
from core.agents.tester import TesterAgent, make_tester_node
from core.llm.provider import LLMProvider
from core.retrieval.reranker import Reranker, get_reranker
from core.retrieval.vectorstore import VectorStore
from core.sandbox.docker_runner import DockerRunner, get_docker_runner
from core.state import LogEntry, TaskState, make_log_entry
from core.tools.git_commit import _git_commit_impl, apply_sandbox_to_real_repo
from core.tools.search_codebase import retrieve_context


def make_retriever_node(db_path: str = "chroma_db", reranker: Reranker | None = None):
    """Factory creating the Retriever tool node for the LangGraph orchestrator."""
    vs = VectorStore(db_path=db_path)
    rr = reranker or get_reranker()

    def retriever_node(state: TaskState) -> dict[str, Any]:
        plan = state.get("plan") or []
        idx = state.get("current_step_index", 0)
        query = plan[idx] if (plan and idx < len(plan)) else state.get("user_request", "")

        chunks = retrieve_context(query=query, vector_store=vs, reranker=rr, top_k_initial=20, top_k_final=3)
        log_entry = make_log_entry(
            node="retriever",
            message=f"Retrieved {len(chunks)} relevant code chunk(s) for step {idx + 1}.",
            level="info",
        )
        return {
            "retrieved_context": chunks,
            "status": "coding",
            "logs": [log_entry],
        }

    return retriever_node


def make_approval_gate_node(require_approval: bool = True):
    """
    Approval Gate node (Section 4):
    Pauses execution before committing changes to the real repository.
    Can be bypassed via config (require_approval=False) for autonomous test runs.
    """
    def approval_gate_node(state: TaskState | Mapping[str, Any]) -> dict[str, Any]:
        user_approved = state.get("user_approved")

        if not require_approval or user_approved is True:
            return {
                "status": "awaiting_approval",
                "logs": [make_log_entry("approval", "Approval granted. Proceeding to commit.", level="info")],
            }
        elif user_approved is False:
            return {
                "status": "done",
                "logs": [make_log_entry("approval", "Changes rejected by user. Diff left uncommitted.", level="warning")],
            }
        else:
            return {
                "status": "awaiting_approval",
                "logs": [make_log_entry("approval", "Awaiting user approval before git commit.", level="info")],
            }

    return approval_gate_node


def make_commit_node():
    """
    Commit Executor node (Section 4 & 8):
    Only invoked post-approval.
    Applies tested changes from the sandbox copy to the real repository and runs git commit.
    """
    def commit_node(state: TaskState) -> dict[str, Any]:
        repo_path = state["repo_path"]
        sandbox_path = state.get("sandbox_path")
        files_touched = state.get("files_touched", [])
        user_approved = state.get("user_approved")

        if user_approved is False:
            return {
                "status": "done",
                "logs": [make_log_entry("commit", "Commit skipped: approval rejected.", level="info")],
            }

        # 1. Apply changes from sandbox to real repository
        if sandbox_path and Path(sandbox_path).exists() and files_touched:
            ok, err = apply_sandbox_to_real_repo(
                repo_path=repo_path,
                sandbox_path=sandbox_path,
                files_touched=files_touched,
            )
            if not ok:
                return {
                    "status": "failed",
                    "logs": [make_log_entry("commit", f"Failed applying sandbox changes to real repo: {err}", level="error")],
                }

        # 2. Run git commit on the real repository
        plan = state.get("plan") or []
        msg = f"Fix: {state['user_request']}"
        if plan:
            msg += "\n\nPlan completed:\n" + "\n".join(f"- {s}" for s in plan)

        res = _git_commit_impl(repo_path=repo_path, message=msg, files_to_add=files_touched or None)
        if res.success:
            log_msg = f"Committed changes as {res.commit_hash[:7]}: '{res.message}'" if res.commit_hash else res.message
            return {
                "status": "done",
                "logs": [make_log_entry("commit", log_msg, level="success")],
            }
        else:
            return {
                "status": "done",
                "logs": [make_log_entry("commit", f"Commit notice: {res.error}", level="warning")],
            }

    return commit_node


def route_after_tester(state: TaskState | Mapping[str, Any]) -> Literal["coder", "retriever", "approval_gate", "failed"]:
    """
    Conditional routing edge after Tester node:
    - If passed and more plan steps remain: route to 'retriever' for next step.
    - If passed and all steps done: route to 'approval_gate'.
    - If failed and retries remain: route to 'coder' with failure summary.
    - If failed and retries exhausted: route to 'failed'.
    """
    status = state.get("status")

    if status in ("awaiting_approval", "done"):
        return "approval_gate"
    elif status == "retrieving":
        return "retriever"
    elif status == "retrying":
        return "coder"
    else:
        return "failed"


def route_after_approval(state: TaskState | Mapping[str, Any]) -> Literal["commit", "done"]:
    """Conditional routing edge after Approval Gate node."""
    status = state.get("status")
    if status == "awaiting_approval":
        return "done"
    elif state.get("user_approved") is True:
        return "commit"
    else:
        return "done"


# =====================================================================
# Full Core Loop (Planner + Retriever + Coder + Tester + Approval Gate)
# =====================================================================

def build_full_graph(
    llm: LLMProvider,
    db_path: str = "chroma_db",
    test_cmd: str | None = None,
    require_approval: bool = True,
    interrupt_before_commit: bool = True,
    checkpointer: Any = None,
) -> Any:
    """
    Builds the complete Phase 1 LangGraph state machine with Approval Gate:
    Planner -> Retriever -> Coder -> Tester -> (loop) -> Approval Gate -> Commit
    """
    planner_node = make_planner_node(llm=llm)
    retriever_node = make_retriever_node(db_path=db_path)
    coder_node = make_coder_node(llm=llm)
    tester_node = make_tester_node(llm=llm, test_cmd=test_cmd)
    approval_node = make_approval_gate_node(require_approval=require_approval)
    commit_node = make_commit_node()

    graph = StateGraph(cast(Any, TaskState))

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)
    graph.add_node("approval_gate", approval_node)
    graph.add_node("commit_node", commit_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "coder")
    graph.add_edge("coder", "tester")

    graph.add_conditional_edges(
        "tester",
        route_after_tester,
        {
            "coder": "coder",
            "retriever": "retriever",
            "approval_gate": "approval_gate",
            "failed": END,
        },
    )

    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "commit": "commit_node",
            "done": END,
        },
    )
    graph.add_edge("commit_node", END)

    interrupt_nodes = ["commit_node"] if (require_approval and interrupt_before_commit) else None

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )


def resume_task_with_approval(
    state: TaskState | Mapping[str, Any],
    approved: bool,
) -> TaskState:
    """
    Resumes an awaiting_approval task with human decision (y/n).
    - If approved: copies changes from sandbox to real repo and executes git commit.
    - If rejected: leaves diff uncommitted and marks task as done.
    """
    updated = dict(state)
    updated["user_approved"] = approved

    commit_node_fn = make_commit_node()
    res = commit_node_fn(updated)  # type: ignore
    updated.update(res)
    if "logs" in res:
        updated["logs"] = list(cast(list[LogEntry], updated.get("logs") or [])) + list(res.get("logs") or [])

    return updated  # type: ignore


def compute_clean_diff(repo_path: str, sandbox_path: str | None, files_touched: list[str]) -> str | None:
    """
    Computes a clean unified diff between the real repository and sandbox copy for all touched files.
    Avoids string concatenation of intermediate step diffs.
    """
    import difflib

    if not sandbox_path or not files_touched:
        return None

    real_root = Path(repo_path).resolve()
    sb_root = Path(sandbox_path).resolve()

    diff_parts = []
    for rel_file in files_touched:
        real_file = real_root / rel_file
        sb_file = sb_root / rel_file

        old_lines = real_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if real_file.exists() else []
        new_lines = sb_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if sb_file.exists() else []

        file_diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel_file}",
                tofile=f"b/{rel_file}",
            )
        )
        if file_diff:
            diff_parts.append("".join(file_diff))

    return "\n".join(diff_parts) if diff_parts else None


def _merge_state_updates(state: dict[str, Any], updates: dict[str, Any]) -> None:
    """Applies node updates to state, ensuring logs, files_touched accumulate and diff is cleanly computed."""
    new_logs = updates.get("logs")
    new_files = updates.get("files_touched")
    new_diff = updates.get("proposed_diff")

    updates_no_special = {
        k: v for k, v in updates.items()
        if k not in ("logs", "files_touched", "proposed_diff")
    }
    state.update(updates_no_special)

    if new_logs:
        state["logs"] = list(cast(list[LogEntry], state.get("logs") or [])) + list(new_logs)

    if new_files:
        existing_files = list(state.get("files_touched") or [])
        for f in new_files:
            if f and f not in existing_files:
                existing_files.append(f)
        state["files_touched"] = existing_files

    # Prefer clean net diff computed between original repo and sandbox copy
    repo_path = state.get("repo_path")
    sandbox_path = state.get("sandbox_path")
    files_touched = state.get("files_touched") or []

    if repo_path and sandbox_path and files_touched:
        clean_diff = compute_clean_diff(repo_path=repo_path, sandbox_path=sandbox_path, files_touched=files_touched)
        if clean_diff:
            state["proposed_diff"] = clean_diff
    elif new_diff:
        existing_diff = state.get("proposed_diff")
        if existing_diff and new_diff not in existing_diff:
            state["proposed_diff"] = existing_diff + "\n" + new_diff
        elif not existing_diff:
            state["proposed_diff"] = new_diff


def run_full_loop(
    state: TaskState,
    llm: LLMProvider,
    db_path: str = "chroma_db",
    test_cmd: str | None = None,
    use_sandbox: bool = True,
    auto_approve: bool = False,
    max_steps: int = 20,
    callback: Callable[[TaskState], None] | None = None,
) -> TaskState:
    """
    Direct runner for the full Planner + Retriever + Coder + Tester + Approval cycle.
    Executes within an isolated sandbox.
    If auto_approve is True: commits automatically upon passing tests.
    If auto_approve is False: stops at awaiting_approval so caller can prompt user.
    """
    runner = get_docker_runner()
    session = runner.create_session(repo_path=state["repo_path"]) if use_sandbox else None

    current_state = dict(state)
    if session:
        current_state["sandbox_path"] = session.sandbox_path
        mode = "Docker container" if session.is_docker else "isolated sandbox copy"
        current_state["logs"] = list(cast(list[LogEntry], current_state.get("logs") or [])) + [
            make_log_entry("orchestrator", f"Initialized {mode} for execution.", level="info")
        ]

        # Execute pre-flight baseline test run to capture any existing failure
        try:
            preflight_res = runner.run_tests(session, test_cmd=test_cmd)
            current_state["test_result"] = preflight_res.to_dict()
            if not preflight_res.passed:
                failing_str = ", ".join(preflight_res.failing_tests[:2]) if preflight_res.failing_tests else preflight_res.summary
                current_state["logs"] = list(current_state.get("logs") or []) + [
                    make_log_entry("orchestrator", f"Baseline tests FAILED: {failing_str}.", level="warning")
                ]
            else:
                current_state["logs"] = list(current_state.get("logs") or []) + [
                    make_log_entry("orchestrator", f"Baseline tests PASSED: {preflight_res.summary}.", level="info")
                ]
        except Exception:
            pass

    planner = PlannerAgent(llm=llm)
    retriever_node = make_retriever_node(db_path=db_path)
    coder = CoderAgent(llm=llm)
    tester = TesterAgent(llm=llm, test_cmd=test_cmd)

    try:
        # 1. Planner
        planner_updates = planner.run(current_state)  # type: ignore
        _merge_state_updates(current_state, planner_updates)
        if callback:
            callback(current_state)  # type: ignore

        steps = 0
        next_node = "retriever"

        while steps < max_steps and next_node not in ("approval_gate", "done", "failed"):
            steps += 1

            if next_node == "retriever":
                retriever_updates = retriever_node(current_state)  # type: ignore
                _merge_state_updates(current_state, retriever_updates)
                if callback:
                    callback(current_state)  # type: ignore
                next_node = "coder"

            elif next_node == "coder":
                coder_updates = coder.run(current_state)  # type: ignore
                _merge_state_updates(current_state, coder_updates)
                if callback:
                    callback(current_state)  # type: ignore

                # Tester follows coder
                tester_updates = tester.run(current_state)  # type: ignore
                _merge_state_updates(current_state, tester_updates)
                if callback:
                    callback(current_state)  # type: ignore

                next_node = route_after_tester(current_state)  # type: ignore

        # Handle approval gate
        if next_node == "approval_gate" or current_state.get("status") == "awaiting_approval":
            if auto_approve:
                current_state = resume_task_with_approval(cast(TaskState, current_state), approved=True)
            else:
                current_state["status"] = "awaiting_approval"

    finally:
        # If not auto-approved, we retain sandbox copy until approval or rejection
        if session and (auto_approve or current_state.get("status") in ("done", "failed")):
            runner.cleanup(session)

    return current_state  # type: ignore


# =====================================================================
# Milestone A (Coder + Tester only) for backward compatibility
# =====================================================================

def build_milestone_a_graph(
    llm: LLMProvider,
    test_cmd: str | None = None,
) -> Any:
    """Builds the LangGraph state machine for Milestone A (Coder + Tester only)."""
    coder_node = make_coder_node(llm=llm)
    tester_node = make_tester_node(llm=llm, test_cmd=test_cmd)

    graph = StateGraph(cast(Any, TaskState))

    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)

    graph.set_entry_point("coder")
    graph.add_edge("coder", "tester")

    graph.add_conditional_edges(
        "tester",
        route_after_tester,
        {
            "coder": "coder",
            "retriever": "coder",
            "approval_gate": END,
            "failed": END,
        },
    )

    return graph.compile()


def run_milestone_a_loop(
    state: TaskState,
    llm: LLMProvider,
    test_cmd: str | None = None,
    max_steps: int = 10,
    callback: Callable[[TaskState], None] | None = None,
) -> TaskState:
    """Direct runner for Milestone A without relying on Planner/Retriever nodes."""
    coder = CoderAgent(llm=llm)
    tester = TesterAgent(llm=llm, test_cmd=test_cmd)

    current_state = dict(state)
    steps = 0

    while steps < max_steps:
        steps += 1

        coder_updates = coder.run(current_state)  # type: ignore
        _merge_state_updates(current_state, coder_updates)
        if callback:
            callback(current_state)  # type: ignore

        tester_updates = tester.run(current_state)  # type: ignore
        _merge_state_updates(current_state, tester_updates)
        if callback:
            callback(current_state)  # type: ignore

        route = route_after_tester(current_state)  # type: ignore
        if route in ("approval_gate", "done", "failed"):
            break

    return current_state  # type: ignore
