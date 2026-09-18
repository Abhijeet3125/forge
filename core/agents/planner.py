# core/agents/planner.py

from pathlib import Path
import re
from typing import Any

from core.llm.provider import LLMMessage, LLMProvider
from core.state import LogEntry, TaskState, make_log_entry
from core.tools.search_codebase import retrieve_context

PLANNER_SYSTEM_PROMPT = """You are an expert software planning agent.
Your task is to analyze a user request for a repository and break it down into an ordered list of concrete, actionable, independently-verifiable steps.

CRITICAL GUIDELINES:
1. Minimal Steps (DO NOT OVER-PLAN):
   - For a bug fix, error resolution, or test-passing task, produce EXACTLY 1 step. Never artificially split a single bug fix across multiple steps (e.g. do not create separate steps for quantity vs price, or validation vs calculation).
   - Only produce 2-3 steps if the user explicitly requested multiple independent features or tasks.
2. PROHIBITED PROCEDURAL STEPS (VERY IMPORTANT):
   - Do NOT create procedural, diagnostic, or verification steps such as:
     * "Run test_*.py and check/identify error"
     * "Locate the corresponding function or file"
     * "Inspect the error message"
     * "Verify the tests pass"
   - The execution harness runs tests and searches files automatically. Every step you produce is sent to a Coder agent to MODIFY code files.
   - Every step MUST be a concrete code modification specifying the target file, function, and implementation change.
3. Scope Discipline: Focus strictly on the exact bug or feature requested.
   - Do NOT plan steps to update README.md, documentation, comments, or unrelated files.
   - Do NOT plan refactoring or rewriting existing code.
4. Do NOT modify or add test files:
   - Do NOT plan steps to create, add, or edit test files (e.g. tests/, test_*.py) unless the user request explicitly asks to write or update tests. Only plan changes to application source files.
5. Concrete & Self-Contained: Each step must specify what to modify, which file/function, and the expected behavior so the Coder agent can act on it independently without needing the full user prompt.
   Good: "In src/cart.py, add validation checks in add_item() that raise ValueError if quantity or price is negative"
   Bad: "Fix the bug"
   Bad: "Run pytest tests/test_cart.py to identify the error"
6. No Fluff: Output ONLY the ordered steps formatted as:
1. <step 1>
2. <step 2>
"""


def get_repo_orientation(repo_path: str, max_readme_chars: int = 2500) -> str:
    """
    Builds lightweight repository orientation for the Planner (Section 3.1 & 7):
    - Contents of AGENTS.md (if present)
    - Contents of README.md (if present)
    - Top-level directory structure (depth <= 2)
    """
    root = Path(repo_path).resolve()
    parts: list[str] = []

    if not root.exists() or not root.is_dir():
        return "Repository directory not found."

    # 1. AGENTS.md (conventions, instructions)
    agents_file = root / "AGENTS.md"
    if agents_file.is_file():
        try:
            content = agents_file.read_text(encoding="utf-8", errors="replace")
            parts.append(f"### AGENTS.md Conventions:\n{content.strip()}")
        except Exception:
            pass

    # 2. README.md
    readme_file = root / "README.md"
    if readme_file.is_file():
        try:
            content = readme_file.read_text(encoding="utf-8", errors="replace")[:max_readme_chars]
            parts.append(f"### Repository README:\n{content.strip()}")
        except Exception:
            pass

    # 3. Top-level tree
    tree_lines: list[str] = ["### Top-Level Structure:"]
    ignored = {".git", "__pycache__", ".venv", "venv", "node_modules", "chroma_db"}
    try:
        entries = sorted([p for p in root.iterdir() if p.name not in ignored])
        for entry in entries[:25]:
            if entry.is_dir():
                tree_lines.append(f"- {entry.name}/")
                try:
                    children = sorted([c for c in entry.iterdir() if c.name not in ignored])[:8]
                    for child in children:
                        tree_lines.append(f"  - {child.name}")
                except Exception:
                    pass
            else:
                tree_lines.append(f"- {entry.name}")
    except Exception:
        pass

    parts.append("\n".join(tree_lines))
    return "\n\n".join(parts)


def _is_procedural_step(step: str) -> bool:
    """Detects meta/procedural steps that don't specify code modifications."""
    s = step.lower().strip()
    # Matches steps that are just instructions to run tests, find files, or verify
    procedural_patterns = [
        r"^(?:run|execute)\s+(?:the\s+)?(?:unit\s+)?(?:files?\s+|tests?\s+|pytest|python)",
        r"^(?:locate|find|identify)\s+(?:the\s+)?(?:corresponding\s+)?(?:function|code|file|error|bug)",
        r"^(?:verify|check)\s+(?:that\s+)?(?:the\s+)?(?:unit\s+)?(?:tests?|fix|changes?)\s*(?:pass|work)?",
        r"^(?:inspect|observe)\s+(?:the\s+)?(?:test\s+)?(?:output|error|failure|traceback)",
    ]
    for pat in procedural_patterns:
        if re.search(pat, s) and not any(kw in s for kw in ("add ", "implement ", "update ", "modify ", "change ", "raise ", "return ")):
            return True
    return False


def parse_plan_output(response: str, default_request: str, is_bug_fix: bool = False) -> list[str]:
    """
    Parses ordered steps from the Planner's response.
    Filters out non-code procedural steps and clamps to 1 step for bug fixes.
    """
    steps: list[str] = []

    # Pattern: 1. <step> or 1) <step>
    matches = re.findall(r"^(?:[0-9]+[\.\)]|\-|\*)\s+(.+)$", response, re.MULTILINE)
    for m in matches:
        cleaned = m.strip()
        if cleaned and len(cleaned) > 5 and not _is_procedural_step(cleaned):
            steps.append(cleaned)

    # Fallback if no valid code steps found after filtering
    if not steps:
        lines = [
            line.strip() for line in response.splitlines()
            if line.strip() and not line.strip().startswith("#") and not _is_procedural_step(line)
        ]
        if lines:
            steps = lines[:4]
        else:
            steps = [default_request.strip()]

    # If it is a bug fix, enforce strictly 1 step
    if is_bug_fix:
        return steps[:1]

    # Limit to maximum 4 steps per spec
    return steps[:4]


class PlannerAgent:
    """
    Planner Agent (Section 3.1):
    Turns a user request and repo orientation into 1-4 concrete, ordered plan steps.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, state: TaskState) -> dict[str, Any]:
        """
        Executes the Planner node:
        1. Loads repo orientation and baseline test failure if present.
        2. Calls LLM with user_request.
        3. Parses steps into state['plan'].
        4. Initializes current_step_index = 0, status = 'retrieving'.
        """
        user_request = state["user_request"]
        repo_path = state["repo_path"]

        orientation = get_repo_orientation(repo_path)

        req_lower = user_request.lower()
        test_result = state.get("test_result")
        is_bug_fix = any(
            kw in req_lower
            for kw in ("fix", "bug", "error", "fail", "issue", "pass", "broken", "exception", "test_")
        ) or (test_result is not None and not test_result.get("passed", True))

        test_context = ""
        if test_result and not test_result.get("passed"):
            failing = ", ".join(test_result.get("failing_tests", [])) or "None specified"
            summary = test_result.get("summary", "")
            test_context = (
                f"\n### Baseline Test Failure (Tests are already failing):\n"
                f"- Failing Tests: {failing}\n"
                f"- Summary: {summary}\n"
            )

        if is_bug_fix:
            instruction = (
                "This is a bug fix task. Output EXACTLY 1 concrete implementation step "
                "specifying which file/function to modify and the exact code change required. "
                "Do NOT include procedural steps like 'run test' or 'locate function'."
            )
        else:
            instruction = (
                "Produce an ordered list of 1 to 3 concrete, actionable implementation steps. "
                "Each step must be a concrete code modification."
            )

        user_content = (
            f"User Task Request:\n{user_request}\n\n"
            f"Repository Context:\n{orientation}{test_context}\n\n"
            f"{instruction}"
        )

        messages = [
            LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        response = self.llm.generate(messages, temperature=0.1)
        plan_steps = parse_plan_output(response.content, default_request=user_request, is_bug_fix=is_bug_fix)

        log_entry = make_log_entry(
            node="planner",
            message=f"Planner formulated {len(plan_steps)} step(s): {'; '.join(plan_steps[:2])}",
            level="info",
        )

        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "status": "retrieving",
            "logs": [log_entry],
        }


def make_planner_node(llm: LLMProvider):
    """Factory creating a LangGraph node function for the Planner agent."""
    agent = PlannerAgent(llm=llm)

    def planner_node(state: TaskState) -> dict[str, Any]:
        return agent.run(state)

    return planner_node
