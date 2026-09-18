# cli/main.py

import json
import os
from pathlib import Path
import sys
from typing import Any

import click
import httpx

DEFAULT_SERVER_URL = os.environ.get("FORGE_SERVER_URL", "http://127.0.0.1:8000")

NODE_COLORS = {
    "planner": "cyan",
    "retriever": "blue",
    "coder": "magenta",
    "tester": "yellow",
    "approval": "bright_green",
    "commit": "green",
    "orchestrator": "white",
}

STATUS_COLORS = {
    "planning": "cyan",
    "retrieving": "blue",
    "coding": "magenta",
    "testing": "yellow",
    "retrying": "yellow",
    "awaiting_approval": "bright_cyan",
    "done": "green",
    "failed": "red",
}


def print_log_entry(log: dict[str, Any]) -> None:
    """Pretty prints a single structured log entry to the terminal."""
    node = str(log.get("node", "system")).lower()
    msg = str(log.get("message", ""))
    level = str(log.get("level", "info")).lower()

    node_color = NODE_COLORS.get(node, "white")
    badge = f"[{node.upper()}]"

    click.secho(f"{badge:<14} ", fg=node_color, bold=True, nl=False)

    if level == "error":
        click.secho(msg, fg="red", bold=True)
    elif level == "warning":
        click.secho(msg, fg="yellow")
    elif level == "success":
        click.secho(msg, fg="green", bold=True)
    else:
        click.echo(msg)


def print_diff(diff_text: str) -> None:
    """Pretty prints a unified diff with syntax coloring."""
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            click.secho(line, bold=True)
        elif line.startswith("+"):
            click.secho(line, fg="green")
        elif line.startswith("-"):
            click.secho(line, fg="red")
        elif line.startswith("@@"):
            click.secho(line, fg="cyan")
        else:
            click.echo(line)


@click.group()
@click.version_option("0.1.0", prog_name="forge")
def cli():
    """Forge — Autonomous Coding Agent Loop."""
    pass


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the Forge FastAPI server."""
    import uvicorn

    click.secho(f"Starting Forge API server at http://{host}:{port}...", fg="cyan", bold=True)
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


@cli.command("index")
@click.argument("repo_path", default=".", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--server", default=DEFAULT_SERVER_URL, help="Forge server URL")
@click.option("--db-path", default="chroma_db", help="Persistent ChromaDB storage directory")
@click.option("--reset", is_flag=True, default=False, help="Clear existing index before indexing")
def index(repo_path: str, server: str, db_path: str, reset: bool):
    """Index a repository into the vector database for semantic retrieval."""
    abs_repo = str(Path(repo_path).resolve())
    server = server.rstrip("/")

    click.echo(f"Indexing repository: {click.style(abs_repo, bold=True)}")
    if reset:
        click.secho("Resetting existing collection before indexing...", fg="yellow")

    payload = {
        "repo_path": abs_repo,
        "db_path": db_path,
        "reset": reset,
    }

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{server}/index", json=payload)
            if resp.status_code != 200:
                detail = resp.json().get("detail", resp.text)
                click.secho(f"Error ({resp.status_code}): {detail}", fg="red", err=True)
                sys.exit(1)

            data = resp.json()

        click.secho("\nIndexing complete!", fg="green", bold=True)
        click.echo(f"  Files scanned:    {data.get('files_scanned')}")
        click.echo(f"  Chunks extracted: {data.get('chunks_extracted')}")
        click.echo(f"  Chunks indexed:   {data.get('chunks_indexed')}")
        click.echo(f"  Elapsed time:     {data.get('elapsed_seconds', 0):.2f}s")

    except httpx.ConnectError:
        click.secho(f"Error: Could not connect to Forge server at {server}.", fg="red", err=True)
        click.secho("Is the server running? Start it with: forge serve", fg="yellow", err=True)
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Error indexing repository: {exc}", fg="red", err=True)
        sys.exit(1)


@cli.command("fix")
@click.argument("description")
@click.option("--repo", default=".", type=click.Path(file_okay=False, dir_okay=True), help="Target repository path")
@click.option("--server", default=DEFAULT_SERVER_URL, help="Forge server URL")
@click.option("--max-retries", default=3, type=int, help="Maximum test retry attempts")
@click.option("--auto-approve", is_flag=True, default=False, help="Auto-commit upon passing tests without prompt")
@click.option("--model", default=None, help="Override LLM model name")
@click.option("--test-cmd", default=None, help="Explicit test command override (e.g. 'uv run pytest')")
def fix(description: str, repo: str, server: str, max_retries: int, auto_approve: bool, model: str | None, test_cmd: str | None):
    """Run the autonomous loop to fix a bug or implement a request."""
    abs_repo = str(Path(repo).resolve())
    server = server.rstrip("/")

    click.secho("Initializing task...", fg="cyan")
    click.echo(f"  Target repo: {click.style(abs_repo, bold=True)}")
    click.echo(f"  Request:     {click.style(description, bold=True)}")
    if test_cmd:
        click.echo(f"  Test cmd:    {click.style(test_cmd, bold=True)}")

    create_payload = {
        "description": description,
        "repo_path": abs_repo,
        "max_retries": max_retries,
        "auto_approve": auto_approve,
        "model": model,
        "test_cmd": test_cmd,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{server}/tasks", json=create_payload)
            if resp.status_code != 201:
                detail = resp.json().get("detail", resp.text)
                click.secho(f"Error ({resp.status_code}): {detail}", fg="red", err=True)
                sys.exit(1)

            task_data = resp.json()
            task_id = task_data["task_id"]

    except httpx.ConnectError:
        click.secho(f"Error: Could not connect to Forge server at {server}.", fg="red", err=True)
        click.secho("Ensure the server is running (`forge serve`).", fg="yellow", err=True)
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Error creating task: {exc}", fg="red", err=True)
        sys.exit(1)

    click.secho(f"Task created: ID={task_id}", fg="green", bold=True)
    click.echo("-" * 60)

    stream_url = f"{server}/tasks/{task_id}/stream"
    seen_logs: set[str] = set()
    has_prompted_approval = False
    current_event = "message"
    final_status = "unknown"
    last_data: dict[str, Any] = {}

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("GET", stream_url) as response:
                if response.status_code != 200:
                    click.secho(f"Failed to stream task events: HTTP {response.status_code}", fg="red", err=True)
                    sys.exit(1)

                for raw_line in response.iter_lines():
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="replace").strip()
                    else:
                        line = raw_line.strip()

                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        continue

                    if line.startswith("data:"):
                        data_str = line.split(":", 1)[1].strip()
                        try:
                            data = json.loads(data_str)
                        except Exception:
                            continue

                        # Process event by type
                        if current_event == "log":
                            # Deduplicate logs by timestamp + message
                            log_key = f"{data.get('timestamp')}:{data.get('message')}"
                            if log_key not in seen_logs:
                                seen_logs.add(log_key)
                                print_log_entry(data)

                        elif current_event == "awaiting_approval":
                            if has_prompted_approval:
                                current_event = "message"
                                continue
                            has_prompted_approval = True

                            click.echo("\n" + "=" * 60)
                            click.secho("APPROVAL GATE: All plan steps and tests passed!", fg="bright_cyan", bold=True)

                            files = data.get("files_touched", [])
                            if files:
                                click.echo(f"Files modified: {', '.join(files)}")

                            diff = data.get("proposed_diff")
                            if diff:
                                click.echo("\nProposed Diff:")
                                print_diff(diff)

                            click.echo("=" * 60)

                            # Prompt user for approval
                            approved = click.confirm(
                                click.style("\nDo you want to commit these changes to the repository?", fg="bright_cyan", bold=True),
                                default=True,
                            )

                            # Submit approval decision
                            approve_resp = client.post(
                                f"{server}/tasks/{task_id}/approve",
                                json={"approved": approved},
                                timeout=60.0,
                            )
                            if approve_resp.status_code != 200:
                                click.secho(f"Failed to submit approval: {approve_resp.text}", fg="red")

                        elif current_event == "complete":
                            final_status = data.get("status", "done")
                            # Flush any logs in the final response that were not streamed individually
                            for entry in data.get("logs", []):
                                log_key = f"{entry.get('timestamp')}:{entry.get('message')}"
                                if log_key not in seen_logs:
                                    seen_logs.add(log_key)
                                    print_log_entry(entry)
                            last_data = data
                            break

                        current_event = "message"

    except (KeyboardInterrupt, SystemExit):
        click.secho("\nStreaming interrupted by user.", fg="yellow")
        sys.exit(130)
    except Exception as exc:
        click.secho(f"\nStream disconnected: {exc}", fg="red", err=True)

    click.echo("-" * 60)
    if final_status == "done":
        click.secho("Forge loop completed successfully!", fg="green", bold=True)
        sys.exit(0)
    elif final_status == "failed":
        test_res = last_data.get("test_result") if isinstance(last_data, dict) else None
        if test_res and not test_res.get("passed"):
            click.secho(f"Forge loop stopped: tests failed after retries.", fg="red", bold=True)
            click.echo(f"  Summary: {test_res.get('summary')}")
            if test_res.get("failing_tests"):
                click.echo("  Failing tests:")
                for ft in test_res.get("failing_tests"):
                    click.echo(f"    - {ft}")
        else:
            click.secho("Forge loop stopped: task failed.", fg="red", bold=True)
        sys.exit(1)
    else:
        click.echo(f"Task finished with status: {final_status}")


@cli.command("status")
@click.argument("task_id", required=False)
@click.option("--server", default=DEFAULT_SERVER_URL, help="Forge server URL")
def status(task_id: str | None, server: str):
    """View status of a specific task or list all recent tasks."""
    server = server.rstrip("/")

    try:
        with httpx.Client(timeout=15.0) as client:
            if task_id:
                resp = client.get(f"{server}/tasks/{task_id}")
                if resp.status_code == 404:
                    click.secho(f"Task '{task_id}' not found.", fg="red", err=True)
                    sys.exit(1)
                elif resp.status_code != 200:
                    click.secho(f"Error fetching task: {resp.text}", fg="red", err=True)
                    sys.exit(1)

                data = resp.json()
                st = data.get("status", "unknown")
                st_color = STATUS_COLORS.get(st, "white")

                click.secho(f"\nTask ID:   {data.get('task_id')}", bold=True)
                click.echo(f"Status:    {click.style(st.upper(), fg=st_color, bold=True)}")
                click.echo(f"Request:   {data.get('user_request')}")
                click.echo(f"Repo:      {data.get('repo_path')}")
                click.echo(f"Created:   {data.get('created_at')}")

                plan = data.get("plan", [])
                if plan:
                    click.secho("\nPlan Steps:", bold=True)
                    cur_idx = data.get("current_step_index", 0)
                    for i, step in enumerate(plan):
                        mark = "[x]" if i < cur_idx else "[ ]"
                        click.echo(f"  {mark} Step {i + 1}: {step}")

                files = data.get("files_touched", [])
                if files:
                    click.echo(f"\nFiles touched: {', '.join(files)}")

                tr = data.get("test_result")
                if tr:
                    t_passed = tr.get("passed")
                    t_color = "green" if t_passed else "red"
                    click.echo(f"Test Suite:    {click.style('PASSED' if t_passed else 'FAILED', fg=t_color, bold=True)} ({tr.get('summary')})")

                diff = data.get("proposed_diff")
                if diff:
                    click.secho("\nProposed Diff:", bold=True)
                    print_diff(diff)

            else:
                resp = client.get(f"{server}/tasks")
                if resp.status_code != 200:
                    click.secho(f"Error fetching tasks: {resp.text}", fg="red", err=True)
                    sys.exit(1)

                tasks = resp.json()
                if not tasks:
                    click.echo("No tasks recorded yet.")
                    return

                click.secho(f"\n{'ID':<10} {'STATUS':<18} {'CREATED':<22} {'REQUEST'}", bold=True)
                click.echo("-" * 75)
                for t in tasks:
                    tid = t.get("task_id", "")
                    st = t.get("status", "")
                    st_color = STATUS_COLORS.get(st, "white")
                    created = t.get("created_at", "")[:19].replace("T", " ")
                    req_short = t.get("user_request", "")[:35]
                    if len(t.get("user_request", "")) > 35:
                        req_short += "..."
                    click.echo(f"{tid:<10} {click.style(st, fg=st_color):<27} {created:<22} {req_short}")
                click.echo()

    except httpx.ConnectError:
        click.secho(f"Error: Could not connect to Forge server at {server}.", fg="red", err=True)
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
