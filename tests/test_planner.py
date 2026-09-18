# tests/test_planner.py

from collections.abc import Iterator, Sequence
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agents.planner import (
    PlannerAgent,
    get_repo_orientation,
    make_planner_node,
    parse_plan_output,
)
from core.llm.provider import LLMMessage, LLMProvider, LLMResponse
from core.state import create_initial_state


class MockLLM(LLMProvider):
    def __init__(self, response: str):
        super().__init__(model="mock-planner-model")
        self.response = response

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self.response, model=self.model)

    def generate_stream(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.response


class TestPlannerAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # Create README.md
        (self.repo / "README.md").write_text("# Shopping Cart Service\nHandles cart calculations.", encoding="utf-8")

        # Create AGENTS.md
        (self.repo / "AGENTS.md").write_text("Conventions: Never modify schema without migration.", encoding="utf-8")

        # Create subdirectories and files
        src_dir = self.repo / "src"
        src_dir.mkdir()
        (src_dir / "cart.py").write_text("def calculate_total(): pass", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_repo_orientation(self):
        orientation = get_repo_orientation(str(self.repo))
        self.assertIn("Shopping Cart Service", orientation)
        self.assertIn("AGENTS.md Conventions:", orientation)
        self.assertIn("Never modify schema without migration", orientation)
        self.assertIn("- src/", orientation)
        self.assertIn("- cart.py", orientation)

    def test_parse_plan_output(self):
        llm_output = (
            "Here is the plan:\n"
            "1. Modify calculate_total() in src/cart.py to deduct discount\n"
            "2. Add discount deduction logic and validation\n"
            "3. Run unit tests to verify the fix\n"
        )
        # Without is_bug_fix, the procedural step "Run unit tests..." is filtered, leaving 2 code steps
        steps = parse_plan_output(llm_output, default_request="Add discount support", is_bug_fix=False)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], "Modify calculate_total() in src/cart.py to deduct discount")
        self.assertEqual(steps[1], "Add discount deduction logic and validation")

    def test_parse_plan_output_bug_fix_clamped_to_one_step(self):
        llm_output = (
            "1. In src/cart.py, add validation check for price >= 0\n"
            "2. In src/cart.py, add validation check for quantity >= 0\n"
            "3. Verify that tests pass\n"
        )
        steps = parse_plan_output(llm_output, default_request="Fix cart negative price bug", is_bug_fix=True)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], "In src/cart.py, add validation check for price >= 0")

    def test_planner_agent_run(self):
        plan_text = (
            "1. Add negative check for discount in src/cart.py\n"
            "2. Update return value to subtract discount\n"
        )
        mock_llm = MockLLM(plan_text)
        planner = PlannerAgent(llm=mock_llm)

        # Bug fix request -> clamped to 1 step
        state = create_initial_state(
            user_request="Fix cart to deduct discount",
            repo_path=str(self.repo),
        )

        updates = planner.run(state)
        self.assertEqual(len(updates["plan"]), 1)
        self.assertEqual(updates["current_step_index"], 0)
        self.assertEqual(updates["status"], "retrieving")
        self.assertTrue(len(updates["logs"]) > 0)


if __name__ == "__main__":
    unittest.main()
