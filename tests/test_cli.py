# tests/test_cli.py

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from click.testing import CliRunner
import httpx

from cli.main import cli


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Forge — Autonomous Coding Agent Loop", result.output)
        self.assertIn("fix", result.output)
        self.assertIn("index", result.output)
        self.assertIn("status", result.output)
        self.assertIn("serve", result.output)

    def test_status_unreachable_server(self):
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
            result = self.runner.invoke(cli, ["status", "--server", "http://127.0.0.1:59999"])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Could not connect to Forge server", result.output)

    def test_status_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("httpx.Client.get", return_value=mock_resp):
            result = self.runner.invoke(cli, ["status"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No tasks recorded yet.", result.output)

    def test_status_task_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "task_id": "t1234567",
                "status": "done",
                "user_request": "Fix discount calculation in cart",
                "repo_path": "/tmp/repo",
                "created_at": "2026-09-15T12:00:00Z",
            }
        ]

        with patch("httpx.Client.get", return_value=mock_resp):
            result = self.runner.invoke(cli, ["status"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("t1234567", result.output)
            self.assertIn("done", result.output)
            self.assertIn("Fix discount calculation", result.output)

    def test_status_single_task_detail(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "task_id": "abc89012",
            "status": "done",
            "user_request": "Refactor parser",
            "repo_path": "/tmp/myrepo",
            "created_at": "2026-09-15T12:00:00Z",
            "plan": ["Inspect parser.py", "Add null check"],
            "current_step_index": 2,
            "files_touched": ["parser.py"],
            "test_result": {
                "passed": True,
                "summary": "12 passed in 0.4s",
                "failing_tests": [],
            },
            "proposed_diff": "--- a/parser.py\n+++ b/parser.py\n@@ -1 +1,2 @@\n+if val is None: return\n",
        }

        with patch("httpx.Client.get", return_value=mock_resp):
            result = self.runner.invoke(cli, ["status", "abc89012"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("abc89012", result.output)
            self.assertIn("PASSED", result.output)
            self.assertIn("Inspect parser.py", result.output)
            self.assertIn("Add null check", result.output)
            self.assertIn("parser.py", result.output)

    def test_index_command_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "files_scanned": 15,
            "chunks_extracted": 42,
            "chunks_indexed": 42,
            "elapsed_seconds": 1.45,
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            result = self.runner.invoke(cli, ["index", ".", "--reset"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Indexing complete!", result.output)
            self.assertIn("Files scanned:    15", result.output)
            self.assertIn("Chunks extracted: 42", result.output)
            self.assertIn("Chunks indexed:   42", result.output)

    def test_fix_command_flow_with_approval(self):
        # 1. Mock create task POST
        create_resp = MagicMock()
        create_resp.status_code = 201
        create_resp.json.return_value = {"task_id": "fix12345", "status": "planning"}

        # 2. Mock SSE stream lines
        sse_lines = [
            b"event: log\n",
            b'data: {"timestamp": "2026-09-15T12:00:00Z", "node": "planner", "message": "Generated 1 step plan", "level": "info"}\n',
            b"\n",
            b"event: log\n",
            b'data: {"timestamp": "2026-09-15T12:00:01Z", "node": "tester", "message": "Tests passed: 5 passed", "level": "success"}\n',
            b"\n",
            b"event: awaiting_approval\n",
            b'data: {"task_id": "fix12345", "files_touched": ["utils.py"], "proposed_diff": "--- a/utils.py\\n+++ b/utils.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"}\n',
            b"\n",
            b"event: complete\n",
            b'data: {"status": "done"}\n',
            b"\n",
        ]

        stream_mock = MagicMock()
        stream_mock.status_code = 200
        stream_mock.iter_lines.return_value = iter(sse_lines)

        # Context manager for client.stream
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = stream_mock
        stream_cm.__exit__.return_value = False

        # 3. Mock approve task POST
        approve_resp = MagicMock()
        approve_resp.status_code = 200
        approve_resp.json.return_value = {"task_id": "fix12345", "status": "done", "user_approved": True}

        def fake_post(url, *args, **kwargs):
            if "/tasks/fix12345/approve" in url:
                return approve_resp
            return create_resp

        with patch("httpx.Client.post", side_effect=fake_post), patch("httpx.Client.stream", return_value=stream_cm):
            # User responds "y" to confirmation prompt
            result = self.runner.invoke(cli, ["fix", "Fix utility bug", "--repo", "."], input="y\n")

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Task created: ID=fix12345", result.output)
            self.assertIn("[PLANNER]", result.output)
            self.assertIn("Generated 1 step plan", result.output)
            self.assertIn("[TESTER]", result.output)
            self.assertIn("Tests passed", result.output)
            self.assertIn("APPROVAL GATE", result.output)
            self.assertIn("utils.py", result.output)
            self.assertIn("Forge loop completed successfully!", result.output)


if __name__ == "__main__":
    unittest.main()
