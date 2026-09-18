# api/schemas.py

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

from core.state import TaskStatus


class CreateTaskRequest(BaseModel):
    description: str = Field(..., description="Plain-English description of the task or bug to fix")
    repo_path: str = Field(..., description="Local path to the target repository")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum test-failure retry attempts")
    auto_approve: bool = Field(default=False, description="Automatically commit upon passing tests without human prompt")
    model: str | None = Field(default=None, description="Optional override for the LLM model name")
    test_cmd: str | None = Field(default=None, description="Optional explicit test command override (e.g. 'uv run pytest')")


class ApproveTaskRequest(BaseModel):
    approved: bool = Field(..., description="True to approve and commit; False to reject without committing")


class LogEntryModel(BaseModel):
    timestamp: str
    node: str
    message: str
    level: Literal["info", "warning", "error", "success"] = "info"


class TestResultModel(BaseModel):
    passed: bool
    summary: str
    failing_tests: list[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    user_request: str
    repo_path: str
    plan: list[str] = Field(default_factory=list)
    current_step_index: int = 0
    files_touched: list[str] = Field(default_factory=list)
    proposed_diff: str | None = None
    test_result: TestResultModel | None = None
    retry_count: int = 0
    max_retries: int = 3
    user_approved: bool | None = None
    logs: list[LogEntryModel] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TaskSummaryResponse(BaseModel):
    task_id: str
    status: TaskStatus
    user_request: str
    repo_path: str
    created_at: str


class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="Path to repo to index")
    db_path: str = Field(default="chroma_db", description="Persistent ChromaDB storage directory")
    reset: bool = Field(default=False, description="Whether to clear existing collection before indexing")


class IndexResponse(BaseModel):
    files_scanned: int
    chunks_extracted: int
    chunks_indexed: int
    elapsed_seconds: float
