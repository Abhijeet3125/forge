# tests/test_orchestrator.py

from collections.abc import Iterator, Sequence
from pathlib import Path
import sys
import tempfile
from typing import Any, cast
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.coder import CoderAgent
from core.agents.tester import TesterAgent
from core.llm.provider import LLMMessage, LLMProvider, LLMResponse
from core.orchestrator import (
    build_milestone_a_graph,
    compute_clean_diff,
    route_after_tester,
    run_milestone_a_loop,
)
from core.retrieval.chunking import CodeChunk
from core.state import TaskState, create_initial_state


class ScriptedLLMProvider(LLMProvider):
    """Mock LLMProvider that returns pre-scripted responses for deterministic loop testing."""

    def __init__(self, responses: list[str]):
        super().__init__(model="scripted-test-model")
        self.responses = responses
        self.call_count = 0
        self.recorded_messages: list[list[Any]] = []

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.recorded_messages.append(list(messages))
        content = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return LLMResponse(content=content, model=self.model)

    def generate_stream(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.generate(messages).content


class TestMilestoneALoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # 1. Create source file with a deliberate bug (ignoring discount)
        self.cart_py = self.repo / "cart.py"
        self.cart_py.write_text(
            "def calculate_total(prices, discount=0.0):\n"
            "    # Bug: discount is not subtracted\n"
            "    return sum(prices)\n",
            encoding="utf-8",
        )

        # 2. Create test file that asserts discount is applied
        self.test_cart_py = self.repo / "test_cart.py"
        self.test_cart_py.write_text(
            "import unittest\n"
            "from cart import calculate_total\n\n"
            "class TestCart(unittest.TestCase):\n"
            "    def test_discount(self):\n"
            "        self.assertEqual(calculate_total([100.0, 50.0], discount=20.0), 130.0)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        self.test_cmd = f"python3 {self.test_cart_py.name}"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_test_retry_pass_cycle(self):
        """
        Tests the full Milestone A cycle:
        Attempt 1: Coder produces an incorrect fix -> tests fail -> retry count increments.
        Attempt 2: Coder receives test failure -> produces correct fix -> tests pass -> status awaiting_approval.
        """
        response_attempt_1 = (
            "Here is the change:\n"
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    # Incorrect fix (still fails)\n"
            "    return sum(prices) - 5.0\n"
            "```"
        )

        response_attempt_2 = (
            "Fixed the calculation:\n"
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    return sum(prices) - discount\n"
            "```"
        )

        mock_llm = ScriptedLLMProvider([response_attempt_1, response_attempt_2])

        initial_state = create_initial_state(
            user_request="Fix calculate_total in cart.py to properly deduct discount",
            repo_path=str(self.repo),
            max_retries=3,
        )
        initial_state["plan"] = ["Fix calculate_total in cart.py to deduct discount"]
        initial_state["retrieved_context"] = [
            CodeChunk(
                chunk_id="cart_chunk",
                file_path="cart.py",
                name="calculate_total",
                kind="function",
                start_line=1,
                end_line=3,
                content=self.cart_py.read_text(encoding="utf-8"),
                signature="def calculate_total(prices, discount=0.0):",
            )
        ]

        # Execute loop
        final_state = run_milestone_a_loop(
            state=initial_state,
            llm=mock_llm,
            test_cmd=self.test_cmd,
            max_steps=10,
        )

        # Assertions
        self.assertEqual(final_state["status"], "awaiting_approval")
        self.assertEqual(final_state["retry_count"], 1)
        self.assertIsNotNone(final_state["test_result"])
        assert final_state["test_result"] is not None
        self.assertTrue(final_state["test_result"]["passed"])
        self.assertIn("cart.py", final_state["files_touched"])
        self.assertIsNotNone(final_state["proposed_diff"])
        self.assertIn("- discount", self.cart_py.read_text(encoding="utf-8"))

        # Verify retry compaction was included in the second prompt
        second_prompt = mock_llm.recorded_messages[1]
        user_msg = next(m.content for m in second_prompt if m.role == "user")
        self.assertIn("PREVIOUS ATTEMPT FAILED", user_msg)
        self.assertIn("130.0", user_msg)

    def test_retry_exhaustion_stops_as_failed(self):
        """
        Tests that when all retries fail, status transitions to 'failed'.
        """
        always_failing_response = (
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    return 0.0\n"
            "```"
        )

        mock_llm = ScriptedLLMProvider([always_failing_response])

        initial_state = create_initial_state(
            user_request="Fix calculate_total",
            repo_path=str(self.repo),
            max_retries=2,
        )
        initial_state["plan"] = ["Fix calculate_total"]

        final_state = run_milestone_a_loop(
            state=initial_state,
            llm=mock_llm,
            test_cmd=self.test_cmd,
            max_steps=10,
        )

        self.assertEqual(final_state["status"], "failed")
        self.assertEqual(final_state["retry_count"], 2)
        self.assertIsNotNone(final_state["test_result"])
        assert final_state["test_result"] is not None
        self.assertFalse(final_state["test_result"]["passed"])

    def test_route_after_tester(self):
        self.assertEqual(route_after_tester({"status": "awaiting_approval"}), "approval_gate")
        self.assertEqual(route_after_tester({"status": "done"}), "approval_gate")
        self.assertEqual(route_after_tester({"status": "retrying"}), "coder")
        self.assertEqual(route_after_tester({"status": "failed"}), "failed")

    def test_langgraph_compiled_graph(self):
        """Tests that the LangGraph StateGraph executes Coder and Tester nodes properly."""
        response_fix = (
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    return sum(prices) - discount\n"
            "```"
        )
        mock_llm = ScriptedLLMProvider([response_fix])
        graph = build_milestone_a_graph(llm=mock_llm, test_cmd=self.test_cmd)

        initial_state = create_initial_state(
            user_request="Fix calculate_total",
            repo_path=str(self.repo),
        )
        initial_state["plan"] = ["Fix calculate_total"]
        initial_state["retrieved_context"] = [
            CodeChunk(
                chunk_id="cart_chunk",
                file_path="cart.py",
                name="calculate_total",
                kind="function",
                start_line=1,
                end_line=3,
                content=self.cart_py.read_text(encoding="utf-8"),
            )
        ]

        result = graph.invoke(initial_state)
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertTrue(result["test_result"]["passed"])
        self.assertIn("- discount", self.cart_py.read_text(encoding="utf-8"))

    def test_full_planner_coder_tester_loop(self):
        """Tests the complete autonomous pipeline starting from only a user request."""
        planner_response = (
            "1. Fix calculate_total in cart.py to subtract discount parameter"
        )
        coder_response = (
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    return sum(prices) - discount\n"
            "```"
        )

        mock_llm = ScriptedLLMProvider([planner_response, coder_response])

        # Initial state has NO manual plan and NO manual chunks
        initial_state = create_initial_state(
            user_request="Fix calculate_total in cart.py so discount is deducted",
            repo_path=str(self.repo),
            max_retries=3,
        )

        # Execute full loop
        from core.orchestrator import run_full_loop
        final_state = run_full_loop(
            state=initial_state,
            llm=mock_llm,
            test_cmd=self.test_cmd,
            max_steps=10,
        )

        self.assertEqual(final_state["status"], "awaiting_approval")
        self.assertTrue(len(final_state["plan"]) >= 1)
        self.assertEqual(final_state["plan"][0], "Fix calculate_total in cart.py to subtract discount parameter")
        self.assertIsNotNone(final_state["test_result"])
        assert final_state["test_result"] is not None
        self.assertTrue(final_state["test_result"]["passed"])
        self.assertIsNotNone(final_state["proposed_diff"])
        self.assertIn("- discount", final_state["proposed_diff"])

        # Confirm host repo was protected and not modified directly
        self.assertIn("return sum(prices)", self.cart_py.read_text(encoding="utf-8"))

    def test_early_exit_on_bug_fix_when_tests_pass(self):
        """Tests that when a bug fix passes tests on step 1, subsequent planned steps are not executed."""
        fix_response = (
            "*** FILE: cart.py ***\n"
            "```python\n"
            "def calculate_total(prices, discount=0.0):\n"
            "    return sum(prices) - discount\n"
            "```"
        )
        should_never_run_response = (
            "*** FILE: cart.py ***\n"
            "```python\n"
            "# SHOULD NEVER RUN\n"
            "```"
        )
        mock_llm = ScriptedLLMProvider([fix_response, should_never_run_response])

        state = create_initial_state(
            user_request="Fix calculate_total in cart.py",
            repo_path=str(self.repo),
        )
        state["plan"] = [
            "Fix calculate_total in cart.py",
            "Redundant Step 2 that should be skipped",
            "Redundant Step 3 that should be skipped",
        ]
        state["current_step_index"] = 0

        from core.agents.tester import TesterAgent
        coder = CoderAgent(llm=mock_llm)
        tester = TesterAgent(llm=mock_llm, test_cmd=self.test_cmd)

        # Step 1 coder executes fix
        coder_updates = coder.run(state)
        state.update(cast(Any, coder_updates))

        # Step 1 tester runs and observes tests pass
        tester_updates = tester.run(state)
        self.assertEqual(tester_updates["status"], "awaiting_approval")
        self.assertTrue(tester_updates["test_result"]["passed"])
        self.assertIn("Task goal achieved!", tester_updates["logs"][0]["message"])
        # Verified: Coder was only called once, step 2 and 3 were not invoked!
        self.assertEqual(mock_llm.call_count, 1)

    def test_compute_clean_diff_no_duplicate_concatenation(self):
        """Tests that compute_clean_diff produces a single clean unified diff comparing original repo to sandbox."""
        with tempfile.TemporaryDirectory() as sb_dir:
            sb_cart = Path(sb_dir) / "cart.py"
            # Intermediate edit 1
            sb_cart.write_text("def calculate_total(prices, discount=0.0):\n    return sum(prices) - 5\n", encoding="utf-8")
            # Intermediate edit 2
            sb_cart.write_text("def calculate_total(prices, discount=0.0):\n    return sum(prices) - discount\n", encoding="utf-8")

            diff = compute_clean_diff(repo_path=str(self.repo), sandbox_path=sb_dir, files_touched=["cart.py"])
            self.assertIsNotNone(diff)
            # Must have only one diff header block for cart.py
            self.assertEqual(diff.count("--- a/cart.py"), 1)
            self.assertEqual(diff.count("+++ b/cart.py"), 1)
            self.assertIn("-    return sum(prices)", diff)
            self.assertIn("+    return sum(prices) - discount", diff)


if __name__ == "__main__":
    unittest.main()

