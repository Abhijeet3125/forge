import sys
from pathlib import Path
import tempfile
import unittest

# Ensure workspace root is in sys.path when running test file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tools.read_file import _read_file_impl, make_read_file_tool
from core.tools.write_file import _write_file_impl, make_write_file_tool
from core.tools.run_shell import _run_shell_impl, make_run_shell_tool
from core.tools.run_tests import _run_tests_impl, detect_test_command, make_run_tests_tool


class TestReadFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_file_basic(self):
        sample = self.root / "hello.py"
        sample.write_text("print('hello')\nprint('world')\n", encoding="utf-8")

        result = _read_file_impl("hello.py", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.content)
        assert result.content is not None
        self.assertIn("1\tprint('hello')", result.content)
        self.assertIn("2\tprint('world')", result.content)
        self.assertIsNone(result.error)

    def test_read_file_without_line_numbers(self):
        sample = self.root / "data.txt"
        sample.write_text("raw text line", encoding="utf-8")

        result = _read_file_impl("data.txt", repo_root=str(self.root), with_line_numbers=False)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "raw text line")

    def test_read_file_leading_slash(self):
        sub = self.root / "src"
        sub.mkdir()
        sample = sub / "app.py"
        sample.write_text("x = 1", encoding="utf-8")

        result = _read_file_impl("/src/app.py", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.content)
        assert result.content is not None
        self.assertIn("x = 1", result.content)

    def test_read_file_traversal_blocked(self):
        result = _read_file_impl("../secret.txt", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("outside the repo root", result.error)

    def test_read_file_not_found(self):
        result = _read_file_impl("missing.py", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("File not found", result.error)

    def test_read_file_is_directory(self):
        folder = self.root / "subfolder"
        folder.mkdir()

        result = _read_file_impl("subfolder", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("Not a file", result.error)

    def test_read_file_empty(self):
        empty = self.root / "empty.py"
        empty.write_text("", encoding="utf-8")

        result = _read_file_impl("empty.py", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertEqual(result.content, "(empty file)")

    def test_read_file_binary_detected(self):
        bin_file = self.root / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\xff\xfe")

        result = _read_file_impl("data.bin", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("binary", result.error.lower())

    def test_read_file_tool_wrapper(self):
        sample = self.root / "test.py"
        sample.write_text("a = 10", encoding="utf-8")

        tool = make_read_file_tool(repo_root=str(self.root))
        output = tool.invoke({"path": "test.py"})
        self.assertIn("1\ta = 10", output)

        err_output = tool.invoke({"path": "non_existent.py"})
        self.assertTrue(err_output.startswith("ERROR:"))


class TestWriteFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_file_new_file(self):
        result = _write_file_impl("new.py", content="print('new')", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertTrue(result.is_new_file)
        self.assertEqual((self.root / "new.py").read_text(encoding="utf-8"), "print('new')")
        self.assertIsNotNone(result.diff)
        assert result.diff is not None
        self.assertIn("+print('new')", result.diff)

    def test_write_file_nested_auto_mkdir(self):
        result = _write_file_impl("nested/dir/app.py", content="y = 2", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertTrue((self.root / "nested" / "dir" / "app.py").is_file())
        self.assertEqual((self.root / "nested" / "dir" / "app.py").read_text(encoding="utf-8"), "y = 2")

    def test_write_file_overwrite_diff(self):
        target = self.root / "calc.py"
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        new_content = "def add(a, b):\n    return a + b\n"
        result = _write_file_impl("calc.py", content=new_content, repo_root=str(self.root))

        self.assertTrue(result.success)
        self.assertFalse(result.is_new_file)
        self.assertEqual(target.read_text(encoding="utf-8"), new_content)
        self.assertIsNotNone(result.diff)
        assert result.diff is not None
        self.assertIn("-    return a - b", result.diff)
        self.assertIn("+    return a + b", result.diff)

    def test_write_file_traversal_blocked(self):
        result = _write_file_impl("../escaped.py", content="bad", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("outside the repo root", result.error)

    def test_write_file_git_dir_forbidden(self):
        git_dir = self.root / ".git"
        git_dir.mkdir()

        result = _write_file_impl(".git/config", content="bad", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn(".git directory is forbidden", result.error)

    def test_write_file_tool_wrapper(self):
        tool = make_write_file_tool(repo_root=str(self.root))
        msg = tool.invoke({"path": "tool_test.py", "content": "val = 42\n"})
        self.assertIn("Created 'tool_test.py'", msg)
        self.assertTrue((self.root / "tool_test.py").exists())


class TestRunShell(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_shell_allowed_command(self):
        result = _run_shell_impl("echo 'hello shell'", repo_root=str(self.root))
        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello shell", result.stdout)

    def test_run_shell_nonzero_exit_code(self):
        result = _run_shell_impl("python3 -c 'import sys; sys.exit(5)'", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 5)

    def test_run_shell_disallowed_command(self):
        result = _run_shell_impl("curl https://google.com", repo_root=str(self.root))
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("not permitted", result.error)

    def test_run_shell_timeout(self):
        result = _run_shell_impl(
            "python3 -c 'import time; time.sleep(5)'",
            repo_root=str(self.root),
            timeout=1,
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("timed out", result.error)

    def test_run_shell_tool_wrapper(self):
        tool = make_run_shell_tool(repo_root=str(self.root))
        output = tool.invoke({"cmd": "echo 'ok'"})
        self.assertIn("Exit code: 0", output)
        self.assertIn("ok", output)


class TestRunTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detect_test_command(self):
        self.assertEqual(detect_test_command(str(self.root)), "pytest")

        (self.root / "uv.lock").touch()
        self.assertEqual(detect_test_command(str(self.root)), "uv run pytest")

        js_repo = self.root / "js_repo"
        js_repo.mkdir()
        (js_repo / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
        self.assertEqual(detect_test_command(str(js_repo)), "npm test")

    def test_run_tests_passing(self):
        test_file = self.root / "test_sample.py"
        test_file.write_text(
            "import unittest\n\n"
            "class DummyTest(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertEqual(1 + 1, 2)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        result = _run_tests_impl(
            repo_root=str(self.root),
            test_cmd=f"python3 {test_file.name}",
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.failing_tests), 0)

    def test_run_tests_failing_and_compact_extraction(self):
        test_file = self.root / "test_sample.py"
        test_file.write_text(
            "import unittest\n\n"
            "class DummyTest(unittest.TestCase):\n"
            "    def test_fail(self):\n"
            "        self.assertTrue(False, 'Total must be positive')\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        result = _run_tests_impl(
            repo_root=str(self.root),
            test_cmd=f"python3 {test_file.name}",
        )
        self.assertFalse(result.passed)
        self.assertNotEqual(result.exit_code, 0)
        self.assertTrue(any("test_fail" in t for t in result.failing_tests))
        self.assertIsNotNone(result.compact_failure)
        self.assertIn("Total must be positive", result.compact_failure)

    def test_run_tests_tool_wrapper(self):
        test_file = self.root / "test_dummy.py"
        test_file.write_text(
            "import unittest\n\n"
            "class PassTest(unittest.TestCase):\n"
            "    def test_it(self):\n"
            "        pass\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        tool = make_run_tests_tool(repo_root=str(self.root), test_cmd=f"python3 {test_file.name}")
        report = tool.invoke({})
        self.assertIn("Test Suite: PASSED", report)

    def test_extract_pytest_summary_multiple_failures_and_targeting(self):
        from core.tools.run_tests import _extract_pytest_summary

        sample_stdout = """
=================================== FAILURES ===================================
________________ test_check_permission_denies_insufficient_role ________________

    def test_check_permission_denies_insufficient_role():
>       assert check_permission(customer_ctx, "admin") is False
E       AssertionError: assert True is False

tests/test_auth.py:16: AssertionError
_______________ test_add_item_rejects_negative_quantity_or_price _______________

    def test_add_item_rejects_negative_quantity_or_price():
>       with pytest.raises(ValueError):
E       Failed: DID NOT RAISE ValueError

tests/test_cart.py:23: Failed
=========================== short test summary info ============================
FAILED tests/test_auth.py::test_check_permission_denies_insufficient_role - AssertionError...
FAILED tests/test_cart.py::test_add_item_rejects_negative_quantity_or_price
========================= 2 failed, 5 passed in 0.05s =========================
"""
        # Test full test names extracted without truncation
        failing, summary, compact = _extract_pytest_summary(sample_stdout, "", target_files=["src/cart.py"])
        self.assertEqual(len(failing), 2)
        self.assertEqual(failing[0], "tests/test_auth.py::test_check_permission_denies_insufficient_role")
        self.assertEqual(failing[1], "tests/test_cart.py::test_add_item_rejects_negative_quantity_or_price")
        self.assertIn("2 failed, 5 passed", summary)
        # Because target_files mentions cart.py, compact_failure must be from the cart test block
        self.assertIsNotNone(compact)
        self.assertIn("test_add_item_rejects_negative_quantity_or_price", compact)
        self.assertIn("DID NOT RAISE ValueError", compact)


if __name__ == "__main__":
    unittest.main()

