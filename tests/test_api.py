# tests/test_api.py

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from api.server import (
    app,
    set_llm_provider_override,
    task_store,
)
from core.llm.provider import LLMMessage, LLMProvider, LLMResponse
from core.retrieval.embeddings import EmbeddingModel
from core.sandbox.docker_runner import DockerRunner
from core.state import create_initial_state, make_log_entry


class MockLLM(LLMProvider):
    def __init__(self):
        super().__init__(model="mock-model")

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        content = "1. Step one\n```diff\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```"
        return LLMResponse(content=content, model=self.model)

    def generate_stream(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.generate(messages).content


class MockEmbeddingModel(EmbeddingModel):
    def __init__(self, dim: int = 64):
        super().__init__(model_name="mock-api-embedding-model")
        self.dim = dim

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, query: str, use_prefix: bool = True) -> list[float]:
        return self._hash_text(query)

    def _hash_text(self, text: str) -> list[float]:
        import hashlib
        import math

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b / 255.0) - 0.5 for b in h[: self.dim]]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name).resolve()
        self.client = TestClient(app)

        # Initialize real git repository
        subprocess.run(["git", "init"], cwd=str(self.repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@test.local", "commit", "--allow-empty", "-m", "init"],
            cwd=str(self.repo),
            capture_output=True,
            check=True,
        )

        # Create sample file
        self.app_py = self.repo / "app.py"
        self.app_py.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.py"], cwd=str(self.repo), check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@test.local", "commit", "-m", "add app"],
            cwd=str(self.repo),
            check=True,
        )

        # Inject mock LLM provider so no Ollama daemon is needed
        self.mock_llm = MockLLM()
        set_llm_provider_override(self.mock_llm)
        task_store.clear()

    def tearDown(self):
        set_llm_provider_override(None)
        task_store.clear()
        self.temp_dir.cleanup()

    def test_healthz(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_create_task_invalid_repo(self):
        resp = self.client.post("/tasks", json={
            "description": "Fix something",
            "repo_path": "/nonexistent/path/to/repo",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("does not exist", resp.json()["detail"])

    def test_create_and_get_task(self):
        resp = self.client.post("/tasks", json={
            "description": "Fix add function",
            "repo_path": str(self.repo),
            "max_retries": 2,
            "auto_approve": False,
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        task_id = data["task_id"]
        self.assertTrue(task_id)
        self.assertEqual(data["user_request"], "Fix add function")
        self.assertEqual(data["repo_path"], str(self.repo))

        # Retrieve the created task
        get_resp = self.client.get(f"/tasks/{task_id}")
        self.assertEqual(get_resp.status_code, 200)
        get_data = get_resp.json()
        self.assertEqual(get_data["task_id"], task_id)

    def test_get_nonexistent_task(self):
        resp = self.client.get("/tasks/nonexistent_123")
        self.assertEqual(resp.status_code, 404)

    def test_list_tasks(self):
        # Create two tasks
        self.client.post("/tasks", json={"description": "Task 1", "repo_path": str(self.repo)})
        self.client.post("/tasks", json={"description": "Task 2", "repo_path": str(self.repo)})

        resp = self.client.get("/tasks")
        self.assertEqual(resp.status_code, 200)
        tasks = resp.json()
        self.assertGreaterEqual(len(tasks), 2)
        descriptions = [t["user_request"] for t in tasks]
        self.assertIn("Task 1", descriptions)
        self.assertIn("Task 2", descriptions)

    def test_stream_existing_logs(self):
        # Create a task directly in task_store
        initial_state = create_initial_state(
            user_request="Test stream",
            repo_path=str(self.repo),
        )
        initial_state["logs"].append(make_log_entry("coder", "Testing SSE stream replay", level="info"))
        initial_state["status"] = "done"

        record = task_store.create_task(initial_state)

        resp = self.client.get(f"/tasks/{record.task_id}/stream")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])
        body = resp.text
        self.assertIn("Testing SSE stream replay", body)
        self.assertIn("event: complete", body)

    def test_approve_task_success(self):
        # Prepare sandbox copy with changes
        runner = DockerRunner(enable_docker=False)
        session = runner.create_session(repo_path=str(self.repo), force_local=True)
        runner.write_file(session, "app.py", "def add(a, b):\n    # fixed\n    return a + b\n")

        state = create_initial_state(
            user_request="Fix add comments",
            repo_path=str(self.repo),
        )
        state["sandbox_path"] = session.sandbox_path
        state["files_touched"] = ["app.py"]
        state["status"] = "awaiting_approval"
        state["proposed_diff"] = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n+# fixed\n     return a + b\n"

        record = task_store.create_task(state)

        # Call approve endpoint
        resp = self.client.post(f"/tasks/{record.task_id}/approve", json={"approved": True})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "done")
        self.assertTrue(data["user_approved"])

        # Real repository should now have the modified content committed
        content = (self.repo / "app.py").read_text(encoding="utf-8")
        self.assertIn("# fixed", content)

        # Check git log
        res = subprocess.run(["git", "log", "-1", "--pretty=%B"], cwd=str(self.repo), capture_output=True, text=True)
        self.assertIn("Fix add comments", res.stdout)

    def test_approve_task_rejected(self):
        # Prepare sandbox copy with changes
        runner = DockerRunner(enable_docker=False)
        session = runner.create_session(repo_path=str(self.repo), force_local=True)
        runner.write_file(session, "app.py", "def add(a, b): return 999\n")

        state = create_initial_state(
            user_request="Bad change",
            repo_path=str(self.repo),
        )
        state["sandbox_path"] = session.sandbox_path
        state["files_touched"] = ["app.py"]
        state["status"] = "awaiting_approval"

        record = task_store.create_task(state)

        # Reject approval
        resp = self.client.post(f"/tasks/{record.task_id}/approve", json={"approved": False})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "done")
        self.assertFalse(data["user_approved"])

        # Real repository must remain unchanged
        content = (self.repo / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("999", content)

    def test_approve_task_invalid_state(self):
        state = create_initial_state(user_request="Not ready", repo_path=str(self.repo))
        state["status"] = "planning"
        record = task_store.create_task(state)

        resp = self.client.post(f"/tasks/{record.task_id}/approve", json={"approved": True})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not awaiting approval", resp.json()["detail"])

    def test_index_repo_endpoint(self):
        from unittest.mock import patch

        # Create a file to index
        (self.repo / "helper.py").write_text("def helper_fn():\n    '''Helper docstring.'''\n    return True\n")

        test_db = self.temp_dir.name + "/test_chroma"
        with patch("core.retrieval.vectorstore.get_embedding_model", return_value=MockEmbeddingModel()):
            resp = self.client.post("/index", json={
                "repo_path": str(self.repo),
                "db_path": test_db,
                "reset": True,
            })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["files_scanned"], 2)
        self.assertGreaterEqual(data["chunks_extracted"], 1)
        self.assertGreaterEqual(data["chunks_indexed"], 1)
        self.assertGreaterEqual(data["elapsed_seconds"], 0.0)

    def test_stream_awaiting_approval_single_event(self):
        state = create_initial_state(user_request="Ready for approval", repo_path=str(self.repo))
        state["status"] = "awaiting_approval"
        record = task_store.create_task(state)

        resp = self.client.get(f"/tasks/{record.task_id}/stream")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text.count("event: awaiting_approval"), 1)


if __name__ == "__main__":
    unittest.main()
