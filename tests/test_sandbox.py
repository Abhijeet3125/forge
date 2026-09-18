# tests/test_sandbox.py

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sandbox.docker_runner import DockerRunner, WarmPool
from core.tools.run_shell import make_sandboxed_run_shell_tool
from core.tools.run_tests import make_sandboxed_run_tests_tool
from core.tools.write_file import make_sandboxed_write_file_tool


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # Create original sample files
        (self.repo / "main.py").write_text("x = 1\n", encoding="utf-8")
        test_file = self.repo / "test_main.py"
        test_file.write_text(
            "import unittest\n"
            "from main import x\n\n"
            "class TestMain(unittest.TestCase):\n"
            "    def test_val(self):\n"
            "        self.assertEqual(x, 1)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        self.runner = DockerRunner(enable_docker=False)  # Test sandbox isolation logic

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_session_and_repo_isolation(self):
        session = self.runner.create_session(repo_path=str(self.repo), force_local=True)
        try:
            # Sandbox path must be different from host repo path
            self.assertNotEqual(session.sandbox_path, str(self.repo))
            self.assertTrue(Path(session.sandbox_path).exists())
            self.assertTrue((Path(session.sandbox_path) / "main.py").exists())

            # Write file in sandbox
            res = self.runner.write_file(session, "main.py", "x = 42\n")
            self.assertTrue(res.success)
            self.assertIsNotNone(res.diff)
            assert res.diff is not None
            self.assertIn("-x = 1", res.diff)
            self.assertIn("+x = 42", res.diff)

            # CRITICAL CHECK: Host repo file must remain completely untouched!
            self.assertEqual((self.repo / "main.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertEqual((Path(session.sandbox_path) / "main.py").read_text(encoding="utf-8"), "x = 42\n")

        finally:
            self.runner.cleanup(session)
            self.assertFalse(Path(session.sandbox_path).exists())

    def test_sandboxed_command_execution(self):
        session = self.runner.create_session(repo_path=str(self.repo), force_local=True)
        try:
            # Allowed command
            res = self.runner.run_command(session, "echo 'inside sandbox'")
            self.assertTrue(res.success)
            self.assertEqual(res.exit_code, 0)
            self.assertIn("inside sandbox", res.stdout)

            # Disallowed command
            bad_res = self.runner.run_command(session, "curl https://malicious.com")
            self.assertFalse(bad_res.success)
            self.assertIsNotNone(bad_res.error)
            assert bad_res.error is not None
            self.assertIn("not permitted in sandbox", bad_res.error)

        finally:
            self.runner.cleanup(session)

    def test_sandboxed_test_runner(self):
        session = self.runner.create_session(repo_path=str(self.repo), force_local=True)
        try:
            # Run test initially passing in sandbox
            test_res = self.runner.run_tests(session, test_cmd="python3 test_main.py")
            self.assertTrue(test_res.passed)
            self.assertEqual(test_res.exit_code, 0)

            # Break code in sandbox
            self.runner.write_file(session, "main.py", "x = 999\n")
            broken_res = self.runner.run_tests(session, test_cmd="python3 test_main.py")
            self.assertFalse(broken_res.passed)
            self.assertNotEqual(broken_res.exit_code, 0)

            # Host repo is STILL untouched
            self.assertEqual((self.repo / "main.py").read_text(encoding="utf-8"), "x = 1\n")
        finally:
            self.runner.cleanup(session)

    def test_sandboxed_tool_wrappers(self):
        session = self.runner.create_session(repo_path=str(self.repo), force_local=True)
        try:
            write_tool = make_sandboxed_write_file_tool(session, runner=self.runner)
            shell_tool = make_sandboxed_run_shell_tool(session, runner=self.runner)
            tests_tool = make_sandboxed_run_tests_tool(session, runner=self.runner, test_cmd="python3 test_main.py")

            msg = write_tool.invoke({"path": "new_mod.py", "content": "z = 99\n"})
            self.assertIn("Created 'new_mod.py' in sandbox", msg)

            out = shell_tool.invoke({"cmd": "echo 'sandboxed'"})
            self.assertIn("sandboxed", out)

            report = tests_tool.invoke({})
            self.assertIn("Sandbox Test Suite: PASSED", report)
        finally:
            self.runner.cleanup(session)

    def test_warm_pool(self):
        pool = WarmPool(runner=self.runner, pool_size=1)
        session = pool.acquire(repo_path=str(self.repo))
        self.assertTrue(Path(session.sandbox_path).exists())
        pool.release(session)
        self.assertFalse(Path(session.sandbox_path).exists())


if __name__ == "__main__":
    unittest.main()
