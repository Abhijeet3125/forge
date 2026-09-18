# tests/test_approval_gate.py

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestrator import (
    build_full_graph,
    make_approval_gate_node,
    make_commit_node,
    resume_task_with_approval,
)
from core.sandbox.docker_runner import DockerRunner
from core.state import create_initial_state


class TestApprovalGate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # Initialize a real git repo
        subprocess.run(["git", "init"], cwd=str(self.repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@test.local", "commit", "--allow-empty", "-m", "init"],
            cwd=str(self.repo),
            capture_output=True,
            check=True,
        )

        # Add an initial file
        self.cart_py = self.repo / "cart.py"
        self.cart_py.write_text("def total(): return 0\n", encoding="utf-8")
        subprocess.run(["git", "add", "cart.py"], cwd=str(self.repo), check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@test.local", "commit", "-m", "add cart"],
            cwd=str(self.repo),
            check=True,
        )

        # Create a sandbox copy with modified code
        self.runner = DockerRunner(enable_docker=False)
        self.session = self.runner.create_session(repo_path=str(self.repo), force_local=True)
        self.runner.write_file(self.session, "cart.py", "def total(): return 42\n")

    def tearDown(self):
        self.runner.cleanup(self.session)
        self.temp_dir.cleanup()

    def test_approval_granted_commits_to_real_repo(self):
        state = create_initial_state(
            user_request="Update total to return 42",
            repo_path=str(self.repo),
        )
        state["sandbox_path"] = self.session.sandbox_path
        state["files_touched"] = ["cart.py"]
        state["status"] = "awaiting_approval"

        # User approves (y)
        final_state = resume_task_with_approval(state=state, approved=True)

        self.assertEqual(final_state["status"], "done")
        self.assertTrue(final_state["user_approved"])

        # Verify change was applied to the real repository
        self.assertEqual(self.cart_py.read_text(encoding="utf-8"), "def total(): return 42\n")

        # Verify git commit was created in git log
        log_proc = subprocess.run(
            ["git", "log", "-n", "1", "--oneline"],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
        )
        self.assertIn("Fix: Update total to return 42", log_proc.stdout)

    def test_approval_rejected_leaves_repo_uncommitted(self):
        state = create_initial_state(
            user_request="Update total to return 42",
            repo_path=str(self.repo),
        )
        state["sandbox_path"] = self.session.sandbox_path
        state["files_touched"] = ["cart.py"]
        state["status"] = "awaiting_approval"
        state["proposed_diff"] = "+def total(): return 42\n"

        # User rejects (n)
        final_state = resume_task_with_approval(state=state, approved=False)

        self.assertEqual(final_state["status"], "done")
        self.assertFalse(final_state["user_approved"])

        # Real repository must remain uncommitted and unchanged
        self.assertEqual(self.cart_py.read_text(encoding="utf-8"), "def total(): return 0\n")

        # Diff remains available in state for user to inspect
        self.assertEqual(final_state["proposed_diff"], "+def total(): return 42\n")

    def test_approval_gate_node(self):
        gate_node = make_approval_gate_node(require_approval=True)

        # Without decision -> status awaiting_approval
        res_waiting = gate_node({"user_approved": None})
        self.assertEqual(res_waiting["status"], "awaiting_approval")

        # With approval -> status awaiting_approval (ready for commit)
        res_approved = gate_node({"user_approved": True})
        self.assertEqual(res_approved["status"], "awaiting_approval")

        # With rejection -> status done (uncommitted)
        res_rejected = gate_node({"user_approved": False})
        self.assertEqual(res_rejected["status"], "done")


if __name__ == "__main__":
    unittest.main()
