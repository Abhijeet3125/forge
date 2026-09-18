# core/agents/coder.py

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from core.llm.provider import LLMMessage, LLMProvider
from core.state import LogEntry, TaskState, make_log_entry
from core.tools.read_file import _read_file_impl
from core.tools.search_codebase import format_chunks_for_llm
from core.tools.write_file import _write_file_impl

CODER_SYSTEM_PROMPT = """You are an expert autonomous software engineer.
Your task is to implement the given step by modifying or creating source files.

CRITICAL RULES:
1. Fix the underlying implementation to make tests pass.
2. Do NOT weaken, skip, or delete existing tests or test assertions.
3. SURGICAL PRESERVATION (MANDATORY):
   - Keep changes minimal and focused directly on the required fix.
   - ALWAYS preserve all existing functions, helper functions, docstrings, imports, classes, default parameter values, and return statements in the file.
   - NEVER delete, simplify, reformat, or rewrite functions or docstrings not mentioned in the step.
   - Do NOT prune or shorten the file. If you touch a 100-line file, the updated file should still be ~100 lines, retaining every existing function and docstring.
4. Do NOT create, modify, or add test files (e.g. tests/, test_*.py, *_test.py) unless the user explicitly requested writing or modifying tests. All changes must be made strictly in the application source code.
5. Do NOT create or modify README.md, documentation, or markdown files unless explicitly requested.
6. For each file you write or modify, output the target path and complete updated content in this exact format:

*** FILE: path/to/file.py ***
```python
<complete updated file content>
```
"""


@dataclass
class ParsedCodeFile:
    path: str
    content: str


def parse_coder_output(
    response: str,
    default_file_path: str | None = None,
) -> list[ParsedCodeFile]:
    """
    Parses file changes from Coder's response.
    Supports:
    1. *** FILE: path/to/file.py ***
    2. File: path/to/file.py / ### path/to/file.py
    3. ```python:path/to/file.py
    4. Fallback to default_file_path if only a single code block is present.
    """
    files: list[ParsedCodeFile] = []

    # Pattern 1: *** FILE: path ***
    pattern_star = r"\*{3}\s*FILE:\s*([^\*\n\r]+?)\s*\*{3}\s*```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)```"
    matches = list(re.finditer(pattern_star, response, re.DOTALL))
    if matches:
        for m in matches:
            path = m.group(1).strip().strip("'\"`")
            content = m.group(2)
            files.append(ParsedCodeFile(path=path, content=content))
        return files

    # Pattern 2: (### / File:) path followed by fenced code block
    pattern_header = r"(?:###\s*|File:\s*)([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9_]+)\s*\n\s*```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)```"
    matches = list(re.finditer(pattern_header, response, re.DOTALL))
    if matches:
        for m in matches:
            path = m.group(1).strip().strip("'\"`")
            content = m.group(2)
            files.append(ParsedCodeFile(path=path, content=content))
        return files

    # Pattern 3: ```python:path/to/file.py
    pattern_lang = r"```(?:python|py)?:([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9_]+)\n(.*?)```"
    matches = list(re.finditer(pattern_lang, response, re.DOTALL))
    if matches:
        for m in matches:
            path = m.group(1).strip().strip("'\"`")
            content = m.group(2)
            files.append(ParsedCodeFile(path=path, content=content))
        return files

    # Fallback: single code block with known default_file_path
    if default_file_path:
        code_block = re.search(r"```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)```", response, re.DOTALL)
        if code_block:
            files.append(ParsedCodeFile(path=default_file_path, content=code_block.group(1)))

    return files


def build_coder_prompt(state: TaskState) -> list[LLMMessage]:
    """
    Constructs the compact input for Coder per Section 6 rules:
    - Current step text only (no full plan).
    - Top-3 retrieved chunks with metadata (no full repo tree).
    - On retry: one compacted failure summary (assertion + file:line), no full traceback.
    """
    plan = state.get("plan") or []
    step_index = state.get("current_step_index", 0)

    # Use current step if available, or fall back to user_request for single-step runs
    step_text = plan[step_index] if (plan and step_index < len(plan)) else state.get("user_request", "")

    retrieved_chunks = state.get("retrieved_context") or []
    context_str = format_chunks_for_llm(retrieved_chunks)

    prompt_parts = [
        f"Target Step to Implement:\n{step_text.strip()}",
        f"\nRelevant Codebase Context:\n{context_str}",
    ]

    # Include existing file content for files found in retrieved context
    repo_path = state.get("sandbox_path") or state.get("repo_path")
    if repo_path and retrieved_chunks:
        distinct_files: list[str] = []
        for c in retrieved_chunks:
            if c.file_path and c.file_path not in distinct_files:
                distinct_files.append(c.file_path)

        for fp in distinct_files[:2]:
            read_res = _read_file_impl(path=fp, repo_root=repo_path, with_line_numbers=False)
            if read_res.success and read_res.content:
                snippet = read_res.content[:6000]
                prompt_parts.append(
                    f"\nExisting File Content for '{fp}':\n```python\n{snippet}\n```"
                )

    # Check for baseline pre-flight failure or retry failure
    retry_count = state.get("retry_count", 0)
    test_result = state.get("test_result")
    if test_result and not test_result.get("passed"):
        prefix = "PREVIOUS ATTEMPT FAILED THE TEST SUITE" if retry_count > 0 else "BASELINE TEST SUITE IS FAILING"
        summary = test_result.get("summary", "Tests failed")
        failing = ", ".join(test_result.get("failing_tests", [])) or "None specified"
        prompt_parts.append(
            f"\n\n[ATTENTION - {prefix}]\n"
            f"Failure Summary: {summary}\n"
            f"Failing Test(s): {failing}\n"
            f"Note: Do not weaken or remove the tests. Fix the implementation so tests pass."
        )

    user_content = "\n".join(prompt_parts)

    return [
        LLMMessage(role="system", content=CODER_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


class CoderAgent:
    """
    Coder Agent (Section 3.3):
    Produces code changes (patch-style diffs) addressing the current plan step.
    Writes changes to files and captures the unified diff.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, state: TaskState) -> dict[str, Any]:
        """
        Executes the Coder node:
        1. Formulates compacted prompt.
        2. Calls LLM.
        3. Parses code changes and writes to disk.
        4. Updates proposed_diff, files_touched, and status.
        """
        repo_path = state.get("sandbox_path") or state["repo_path"]
        messages = build_coder_prompt(state)

        # Default fallback file path if single chunk in context
        chunks = state.get("retrieved_context") or []
        default_path = chunks[0].file_path if len(chunks) == 1 else None

        # Call LLM
        response = self.llm.generate(messages, temperature=0.1)
        parsed_files = parse_coder_output(response.content, default_file_path=default_path)

        if not parsed_files:
            log_entry = make_log_entry(
                node="coder",
                message="Coder did not produce structured file changes.",
                level="warning",
            )
            return {
                "status": "testing",
                "proposed_diff": None,
                "files_touched": [],
                "logs": [log_entry],
            }

        diffs: list[str] = []
        files_touched: list[str] = []

        user_req = (state.get("user_request") or "").lower()
        user_explicitly_wanted_tests = any(
            kw in user_req
            for kw in ("write test", "add test", "create test", "update test", "modify test", "fix test", "new test")
        )

        for pf in parsed_files:
            pf_p = Path(pf.path)
            is_test_file = (
                any(part in ("tests", "test") for part in pf_p.parts)
                or pf_p.stem.startswith("test_")
                or pf_p.stem.endswith("_test")
            )
            if is_test_file and not user_explicitly_wanted_tests:
                continue

            is_doc_file = (
                pf_p.name.lower() in ("readme.md", "readme", "contributing.md", "docs.md")
                or any(part in ("docs", "documentation") for part in pf_p.parts)
            )
            if is_doc_file and not any(kw in user_req for kw in ("readme", "doc", "documentation")):
                continue

            write_res = _write_file_impl(path=pf.path, content=pf.content, repo_root=repo_path)
            if write_res.success:
                files_touched.append(write_res.path)
                if write_res.diff:
                    diffs.append(write_res.diff)
            else:
                log_entry = make_log_entry(
                    node="coder",
                    message=f"Failed to write file '{pf.path}': {write_res.error}",
                    level="error",
                )
                return {
                    "status": "testing",
                    "proposed_diff": None,
                    "files_touched": files_touched,
                    "logs": [log_entry],
                }

        combined_diff = "\n".join(diffs) if diffs else None
        touched_str = ", ".join(files_touched) if files_touched else "no files"

        log_entry = make_log_entry(
            node="coder",
            message=f"Coder generated updates for: {touched_str}.",
            level="info",
        )

        return {
            "proposed_diff": combined_diff,
            "files_touched": files_touched,
            "status": "testing",
            "logs": [log_entry],
        }


def make_coder_node(llm: LLMProvider):
    """Factory creating a LangGraph node function for the Coder agent."""
    agent = CoderAgent(llm=llm)

    def coder_node(state: TaskState) -> dict[str, Any]:
        return agent.run(state)

    return coder_node
