# api/server.py

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, cast
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (
    ApproveTaskRequest,
    CreateTaskRequest,
    IndexRequest,
    IndexResponse,
    LogEntryModel,
    TaskResponse,
    TaskSummaryResponse,
    TestResultModel,
)
from core.llm.ollama_provider import DEFAULT_OLLAMA_MODEL, OllamaProvider
from core.llm.provider import LLMProvider
from core.orchestrator import resume_task_with_approval, run_full_loop
from core.retrieval.indexer import index_repository
from core.retrieval.vectorstore import VectorStore
from core.sandbox.docker_runner import get_docker_runner
from core.state import TaskState, create_initial_state, make_log_entry


class TaskRecord:
    """Thread-safe task tracking record."""

    def __init__(self, task_id: str, state: TaskState, auto_approve: bool = False, test_cmd: str | None = None):
        self.task_id = task_id
        self.state = state
        self.auto_approve = auto_approve
        self.test_cmd = test_cmd
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.lock = threading.RLock()
        self.subscribers: set[asyncio.Queue] = set()
        self.last_log_count = 0
        self.is_running = False
        self.has_broadcast_awaiting_approval = False

    def to_response(self) -> TaskResponse:
        with self.lock:
            s = self.state
            tr = s.get("test_result")
            test_model = (
                TestResultModel(
                    passed=tr["passed"],
                    summary=tr["summary"],
                    failing_tests=tr.get("failing_tests", []),
                )
                if tr
                else None
            )

            log_models = [
                LogEntryModel(
                    timestamp=l["timestamp"],
                    node=l["node"],
                    message=l["message"],
                    level=l.get("level", "info"),
                )
                for l in s.get("logs", [])
            ]

            return TaskResponse(
                task_id=self.task_id,
                status=s.get("status", "planning"),
                user_request=s["user_request"],
                repo_path=s["repo_path"],
                plan=s.get("plan", []),
                current_step_index=s.get("current_step_index", 0),
                files_touched=s.get("files_touched", []),
                proposed_diff=s.get("proposed_diff"),
                test_result=test_model,
                retry_count=s.get("retry_count", 0),
                max_retries=s.get("max_retries", 3),
                user_approved=s.get("user_approved"),
                logs=log_models,
                created_at=self.created_at,
                updated_at=self.updated_at,
            )

    def to_summary(self) -> TaskSummaryResponse:
        with self.lock:
            return TaskSummaryResponse(
                task_id=self.task_id,
                status=self.state.get("status", "planning"),
                user_request=self.state["user_request"],
                repo_path=self.repo_path,
                created_at=self.created_at,
            )

    @property
    def repo_path(self) -> str:
        return self.state["repo_path"]


class TaskStore:
    """In-memory store for task execution records."""

    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    def create_task(self, state: TaskState, auto_approve: bool = False, test_cmd: str | None = None) -> TaskRecord:
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, state=state, auto_approve=auto_approve, test_cmd=test_cmd)
        with self._lock:
            self._tasks[task_id] = record
        return record

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskRecord]:
        with self._lock:
            records = list(self._tasks.values())
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()


# Global state containers
task_store = TaskStore()
server_state: dict[str, Any] = {
    "loop": None,
    "llm_provider_override": None,
}


def set_llm_provider_override(provider: LLMProvider | None) -> None:
    """Allows test fixtures to inject a mock or scripted LLM provider."""
    server_state["llm_provider_override"] = provider


def get_llm_provider(model: str | None = None) -> LLMProvider:
    if server_state["llm_provider_override"] is not None:
        return server_state["llm_provider_override"]
    return OllamaProvider(model=model or DEFAULT_OLLAMA_MODEL)


def broadcast_event(task_record: TaskRecord, event_type: str, data: Mapping[str, Any]) -> None:
    """Thread-safe event broadcast to all connected SSE clients."""
    loop: asyncio.AbstractEventLoop | None = server_state.get("loop")
    if loop is None or loop.is_closed():
        return

    payload = {"event": event_type, "data": data}
    with task_record.lock:
        for queue in list(task_record.subscribers):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except Exception:
                pass


def run_task_worker(task_record: TaskRecord, llm: LLMProvider) -> None:
    """Background thread worker running the orchestrator loop."""
    task_record.is_running = True

    def on_state_update(updated_state: TaskState) -> None:
        with task_record.lock:
            task_record.state = cast(TaskState, dict(updated_state))
            task_record.updated_at = datetime.now(timezone.utc).isoformat()
            new_logs = task_record.state.get("logs", [])[task_record.last_log_count :]
            task_record.last_log_count = len(task_record.state.get("logs", []))

        # Broadcast each new log entry
        for entry in new_logs:
            broadcast_event(task_record, "log", entry)

        # Broadcast status update
        resp = task_record.to_response()
        broadcast_event(task_record, "status", resp.model_dump())

        if resp.status == "awaiting_approval" and not task_record.has_broadcast_awaiting_approval:
            task_record.has_broadcast_awaiting_approval = True
            broadcast_event(task_record, "awaiting_approval", resp.model_dump())

    try:
        final_state = run_full_loop(
            state=task_record.state,
            llm=llm,
            test_cmd=task_record.test_cmd,
            auto_approve=task_record.auto_approve,
            callback=on_state_update,
        )
        with task_record.lock:
            task_record.state = cast(TaskState, dict(final_state))
            task_record.updated_at = datetime.now(timezone.utc).isoformat()
            new_logs = task_record.state.get("logs", [])[task_record.last_log_count :]
            task_record.last_log_count = len(task_record.state.get("logs", []))

        for entry in new_logs:
            broadcast_event(task_record, "log", entry)

        final_resp = task_record.to_response()
        if final_resp.status == "awaiting_approval" and not task_record.has_broadcast_awaiting_approval:
            task_record.has_broadcast_awaiting_approval = True
            broadcast_event(task_record, "awaiting_approval", final_resp.model_dump())
        elif final_resp.status in ("done", "failed"):
            broadcast_event(task_record, "complete", final_resp.model_dump())

    except Exception as exc:
        import traceback
        traceback.print_exc()
        with task_record.lock:
            err_log = make_log_entry("orchestrator", f"Fatal execution error: {exc}", level="error")
            task_record.state["logs"] = list(task_record.state.get("logs") or []) + [err_log]
            task_record.state["status"] = "failed"
            task_record.updated_at = datetime.now(timezone.utc).isoformat()

        broadcast_event(task_record, "log", err_log)
        resp = task_record.to_response()
        broadcast_event(task_record, "complete", resp.model_dump())

    finally:
        task_record.is_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the running asyncio event loop for thread-safe cross-thread calls
    server_state["loop"] = asyncio.get_running_loop()
    yield
    server_state["loop"] = None


app = FastAPI(
    title="Forge Coding Agent API",
    description="HTTP and Server-Sent Events API for autonomous software engineering loops",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(req: CreateTaskRequest) -> TaskResponse:
    repo_path = Path(req.repo_path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target repository path does not exist or is not a directory: {req.repo_path}",
        )

    initial_state = create_initial_state(
        user_request=req.description,
        repo_path=str(repo_path),
        max_retries=req.max_retries,
    )

    record = task_store.create_task(initial_state, auto_approve=req.auto_approve, test_cmd=req.test_cmd)
    llm = get_llm_provider(model=req.model)

    # Launch execution in background worker thread
    worker_thread = threading.Thread(
        target=run_task_worker,
        args=(record, llm),
        daemon=True,
    )
    worker_thread.start()

    return record.to_response()


@app.get("/tasks", response_model=list[TaskSummaryResponse])
def list_tasks() -> list[TaskSummaryResponse]:
    records = task_store.list_tasks()
    return [r.to_summary() for r in records]


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str) -> TaskResponse:
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found")
    return record.to_response()


@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request):
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        with record.lock:
            record.subscribers.add(queue)
            # Snapshot history to immediately send to this subscriber
            existing_logs = list(record.state.get("logs", []))
            current_resp = record.to_response()

        try:
            # 1. Yield all historical log entries
            for entry in existing_logs:
                yield f"event: log\ndata: {json.dumps(entry)}\n\n"

            # 2. Yield current status snapshot
            yield f"event: status\ndata: {json.dumps(current_resp.model_dump())}\n\n"

            # 3. If already awaiting_approval, send event
            if current_resp.status == "awaiting_approval":
                yield f"event: awaiting_approval\ndata: {json.dumps(current_resp.model_dump())}\n\n"

            # 4. If already completed, send complete event and exit
            if current_resp.status in ("done", "failed"):
                yield f"event: complete\ndata: {json.dumps(current_resp.model_dump())}\n\n"
                return

            # 5. Stream live incoming events
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send SSE keep-alive comment
                    yield ": ping\n\n"
                    continue

                event_type = event.get("event", "message")
                data_json = json.dumps(event.get("data", {}))
                yield f"event: {event_type}\ndata: {data_json}\n\n"

                if event_type == "complete":
                    break

        finally:
            with record.lock:
                record.subscribers.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tasks/{task_id}/approve", response_model=TaskResponse)
def approve_task(task_id: str, req: ApproveTaskRequest) -> TaskResponse:
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found")

    with record.lock:
        if record.state.get("status") != "awaiting_approval":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task '{task_id}' is not awaiting approval (current status: {record.state.get('status')})",
            )

        start_log_count = len(record.state.get("logs", []))
        updated_state = resume_task_with_approval(record.state, approved=req.approved)
        record.state = updated_state
        record.updated_at = datetime.now(timezone.utc).isoformat()
        new_logs = record.state.get("logs", [])[start_log_count:]
        record.last_log_count = len(record.state.get("logs", []))

    # Clean up sandbox session if still allocated
    sandbox_path = updated_state.get("sandbox_path")
    if sandbox_path and Path(sandbox_path).exists():
        import shutil
        shutil.rmtree(sandbox_path, ignore_errors=True)

    # Broadcast logs from commit/resume
    for entry in new_logs:
        broadcast_event(record, "log", entry)

    resp = record.to_response()
    broadcast_event(record, "status", resp.model_dump())
    broadcast_event(record, "complete", resp.model_dump())

    return resp


@app.post("/index", response_model=IndexResponse)
def index_repo(req: IndexRequest) -> IndexResponse:
    repo_path = Path(req.repo_path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Repository path does not exist or is not a directory: {req.repo_path}",
        )

    if req.reset:
        vs = VectorStore(db_path=req.db_path)
        vs.reset()

    stats = index_repository(repo_path=str(repo_path), db_path=req.db_path)

    return IndexResponse(
        files_scanned=stats.files_scanned,
        chunks_extracted=stats.chunks_extracted,
        chunks_indexed=stats.chunks_indexed,
        elapsed_seconds=stats.elapsed_seconds,
    )