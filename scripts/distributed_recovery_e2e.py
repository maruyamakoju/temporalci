from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from temporalci.utils import atomic_write_json
from temporalci.utils import utc_now_iso


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO


def _print_event(message: str) -> None:
    print(f"[distributed-recovery-e2e] {message}", flush=True)


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected JSON payload from {url}: {parsed!r}")
    return parsed


def _get_json(url: str, *, timeout_sec: float = 30.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected JSON payload from {url}: {parsed!r}")
    return parsed


def _is_port_open(*, host: str, port: int, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _service_reachable_from_url(url: str, *, default_port: int) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or default_port)
    return _is_port_open(host=host, port=port)


def _compose_base_command(compose_file: Path) -> list[str]:
    if shutil.which("docker"):
        return ["docker", "compose", "-f", str(compose_file)]
    if shutil.which("docker-compose"):
        return ["docker-compose", "-f", str(compose_file)]
    raise RuntimeError("docker compose command not found; install Docker Desktop/Engine")


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_sec: float = 300.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _print_event(f"run command: {' '.join(command)}")
    return subprocess.run(  # noqa: S603
        command,
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _start_process(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    logs_dir: Path,
) -> ManagedProcess:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    _print_event(f"start process: {name} -> {' '.join(command)}")
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=str(cwd),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ManagedProcess(
        name=name,
        command=command,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
    )


def _stop_process(
    managed: ManagedProcess,
    *,
    force: bool,
    timeout_sec: float = 10.0,
) -> int | None:
    process = managed.process
    try:
        if process.poll() is None:
            if force:
                process.kill()
            else:
                process.terminate()
            try:
                process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    finally:
        managed.log_handle.close()
    return process.returncode


def _wait_for_coordinator(
    *,
    coordinator_url: str,
    timeout_sec: float,
    poll_sec: float,
    coordinator_process: ManagedProcess | None = None,
) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{coordinator_url.rstrip('/')}/healthz"
    while time.time() < deadline:
        if coordinator_process is not None:
            return_code = coordinator_process.process.poll()
            if return_code is not None:
                log_tail = _read_log_tail(coordinator_process.log_path)
                raise RuntimeError(
                    "coordinator process exited before health check passed; "
                    f"exit_code={return_code} log_tail={log_tail!r}"
                )
        try:
            payload = _get_json(health_url, timeout_sec=5.0)
        except Exception:
            time.sleep(poll_sec)
            continue
        if str(payload.get("status", "")) == "ok":
            return
        time.sleep(poll_sec)
    raise RuntimeError(f"coordinator did not become healthy before timeout: {health_url}")


def _wait_for_run_status(
    *,
    coordinator_url: str,
    run_id: str,
    target_statuses: set[str],
    timeout_sec: float,
    poll_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    run_url = f"{coordinator_url.rstrip('/')}/runs/{run_id}"
    while time.time() < deadline:
        payload = _get_json(run_url, timeout_sec=10.0)
        status = str(payload.get("status", "unknown"))
        if status in target_statuses:
            return payload
        if "running" in target_statuses and status in {"completed", "failed"}:
            raise RuntimeError(
                "run reached terminal state before worker-kill injection; "
                f"status={status} run_id={run_id}"
            )
        time.sleep(poll_sec)
    wanted = ", ".join(sorted(target_statuses))
    raise RuntimeError(f"timeout waiting for run_id={run_id} status in {{{wanted}}}")


def _build_inline_suite_yaml(*, sleep_sec: float) -> str:
    return "\n".join(
        [
            "version: 1",
            'project: "distributed"',
            'suite_name: "recovery_e2e"',
            "models:",
            '  - name: "mock-delayed"',
            '    adapter: "mock"',
            "    params:",
            f"      sleep_sec: {max(0.0, float(sleep_sec))}",
            "tests:",
            '  - id: "recovery_case"',
            '    type: "generation"',
            "    prompts:",
            '      - "distributed recovery smoke prompt"',
            "    seeds: [0]",
            "metrics:",
            '  - name: "vbench_temporal"',
            "gates:",
            '  - metric: "vbench_temporal.score"',
            '    op: ">="',
            "    value: 0.0",
            "artifacts:",
            '  video: "none"',
        ]
    )


def _read_log_tail(path: Path, *, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(lines[-max(1, int(max_lines)):]).strip()


def _require_python_module(module_name: str, *, hint: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing python dependency '{module_name}'. {hint}") from exc


def _verify_db_and_queue_state(
    *,
    postgres_dsn: str,
    redis_url: str,
    queue_name: str,
    processing_queue_name: str,
    run_id: str,
    require_empty_queues: bool,
) -> dict[str, Any]:
    try:
        import psycopg
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("psycopg is required for DB verification") from exc

    try:
        import redis
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("redis python client is required for queue verification") from exc

    with psycopg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, status, attempts, worker_id, error
                FROM tasks
                WHERE run_id=%s
                ORDER BY created_at ASC;
                """,
                (run_id,),
            )
            rows = cur.fetchall()
            cur.execute(
                """
                SELECT status
                FROM runs
                WHERE run_id=%s;
                """,
                (run_id,),
            )
            run_row = cur.fetchone()

    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one task row for run_id={run_id}, got {len(rows)}")
    task_id, task_status, attempts, worker_id, task_error = rows[0]
    if str(task_status) != "completed":
        raise RuntimeError(f"task did not complete after recovery: status={task_status}")
    if int(attempts) < 2:
        raise RuntimeError(f"task attempts expected >=2 after recovery, got {attempts}")
    if task_error:
        raise RuntimeError(f"task error should be empty after successful recovery: {task_error}")
    if not run_row or str(run_row[0]) != "completed":
        raise RuntimeError(f"run status in DB expected 'completed', got {run_row!r}")

    client = redis.from_url(redis_url, decode_responses=True)
    queue_depth = int(client.llen(queue_name))
    processing_depth = int(client.llen(processing_queue_name))
    if require_empty_queues and (queue_depth != 0 or processing_depth != 0):
        raise RuntimeError(
            "queue not drained after completion: "
            f"queue_depth={queue_depth} processing_depth={processing_depth}"
        )

    return {
        "task_id": str(task_id),
        "task_status": str(task_status),
        "task_attempts": int(attempts),
        "task_worker_id": str(worker_id) if worker_id is not None else None,
        "queue_depth": queue_depth,
        "processing_depth": processing_depth,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed recovery E2E: kill worker while processing, "
            "requeue tasks, and verify successful recovery."
        )
    )
    parser.add_argument("--artifacts-dir", default="artifacts/distributed-recovery-e2e")
    parser.add_argument("--coordinator-host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=18080)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--use-existing-coordinator", action="store_true")
    parser.add_argument("--postgres-dsn", default="postgresql://temporalci:temporalci@localhost:5432/temporalci")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--queue-name", default="")
    parser.add_argument("--task-lease-sec", type=int, default=300)
    parser.add_argument("--heartbeat-interval-sec", type=int, default=5)
    parser.add_argument("--worker-poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--task-sleep-sec", type=float, default=20.0)
    parser.add_argument("--kill-delay-sec", type=float, default=1.0)
    parser.add_argument("--wait-running-timeout-sec", type=float, default=90.0)
    parser.add_argument("--completion-timeout-sec", type=float, default=240.0)
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--baseline-mode", default="none")
    parser.add_argument("--skip-compose", action="store_true")
    parser.add_argument("--compose-file", default="infra/docker-compose.yml")
    parser.add_argument("--compose-services", default="postgres,redis")
    parser.add_argument("--compose-down", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = artifacts_dir / "logs"
    summary_path = artifacts_dir / "distributed_recovery_summary.json"

    queue_name = str(args.queue_name).strip()
    if not queue_name:
        if args.use_existing_coordinator:
            queue_name = "temporalci:tasks"
        else:
            queue_name = f"temporalci:e2e:{uuid.uuid4().hex[:8]}"
    processing_queue_name = f"{queue_name}:processing"
    require_empty_queues = not (args.use_existing_coordinator and not str(args.queue_name).strip())

    coordinator_url = str(args.coordinator_url).strip()
    if not coordinator_url:
        coordinator_url = f"http://{args.coordinator_host}:{int(args.coordinator_port)}"

    env = os.environ.copy()
    env.update(
        {
            "TEMPORALCI_POSTGRES_DSN": args.postgres_dsn,
            "TEMPORALCI_REDIS_URL": args.redis_url,
            "TEMPORALCI_QUEUE_NAME": queue_name,
            "TEMPORALCI_PROCESSING_QUEUE_NAME": processing_queue_name,
            "TEMPORALCI_ARTIFACTS_DIR": str(artifacts_dir / "runs"),
            "TEMPORALCI_TASK_LEASE_SEC": str(max(30, int(args.task_lease_sec))),
            "TEMPORALCI_HEARTBEAT_INTERVAL_SEC": str(max(5, int(args.heartbeat_interval_sec))),
        }
    )

    summary: dict[str, Any] = {
        "started_at_utc": utc_now_iso(),
        "status": "running",
        "artifacts_dir": str(artifacts_dir),
        "coordinator_url": coordinator_url,
        "queue_name": queue_name,
        "processing_queue_name": processing_queue_name,
        "require_empty_queues": require_empty_queues,
    }
    managed_processes: list[ManagedProcess] = []
    coordinator_process: ManagedProcess | None = None
    started_compose = False
    compose_base_command: list[str] | None = None
    exit_code = 1

    try:
        _require_python_module("psycopg", hint="Install optional dependency set: pip install -e .[distributed]")
        _require_python_module("redis", hint="Install optional dependency set: pip install -e .[distributed]")
        if not args.use_existing_coordinator:
            _require_python_module(
                "fastapi",
                hint="Install optional dependency set: pip install -e .[distributed]",
            )
            _require_python_module(
                "uvicorn",
                hint="Install optional dependency set: pip install -e .[distributed]",
            )

        compose_file = Path(args.compose_file).resolve()
        summary["compose_file"] = str(compose_file)
        summary["compose_skipped"] = bool(args.skip_compose)
        if not args.skip_compose:
            services = [part.strip() for part in str(args.compose_services).split(",") if part.strip()]
            if not services:
                raise RuntimeError("compose services list is empty")
            postgres_reachable = _service_reachable_from_url(args.postgres_dsn, default_port=5432)
            redis_reachable = _service_reachable_from_url(args.redis_url, default_port=6379)
            summary["service_reachable_before"] = {
                "postgres": postgres_reachable,
                "redis": redis_reachable,
            }
            if not (postgres_reachable and redis_reachable):
                compose_base_command = _compose_base_command(compose_file)
                _run_command(
                    [*compose_base_command, "up", "-d", *services],
                    cwd=repo_root,
                    timeout_sec=300.0,
                )
                started_compose = True
                time.sleep(3.0)
                summary["compose_services_started"] = services
            else:
                summary["compose_services_started"] = []

        if not args.use_existing_coordinator:
            coordinator_process = _start_process(
                name="coordinator",
                command=[
                    sys.executable,
                    "-m",
                    "temporalci.coordinator.cli",
                    "serve",
                    "--host",
                    str(args.coordinator_host),
                    "--port",
                    str(int(args.coordinator_port)),
                ],
                cwd=repo_root,
                env=env,
                logs_dir=logs_dir,
            )
            managed_processes.append(coordinator_process)

        _wait_for_coordinator(
            coordinator_url=coordinator_url,
            timeout_sec=45.0,
            poll_sec=max(0.2, float(args.poll_sec)),
            coordinator_process=coordinator_process,
        )
        summary["coordinator_healthy"] = True

        worker1 = _start_process(
            name="worker1",
            command=[
                sys.executable,
                "-m",
                "temporalci.coordinator.cli",
                "worker",
                "--coordinator-url",
                coordinator_url,
                "--worker-id",
                "recovery-worker-1",
                "--poll-interval-sec",
                str(float(args.worker_poll_interval_sec)),
            ],
            cwd=repo_root,
            env=env,
            logs_dir=logs_dir,
        )
        managed_processes.append(worker1)

        inline_suite_root = artifacts_dir / "inline_suite_root"
        inline_suite_root.mkdir(parents=True, exist_ok=True)
        create_payload = {
            "suite_yaml": _build_inline_suite_yaml(sleep_sec=float(args.task_sleep_sec)),
            "suite_root": str(inline_suite_root),
            "model_name": "mock-delayed",
            "artifacts_dir": str(artifacts_dir / "runs"),
            "fail_on_regression": True,
            "baseline_mode": str(args.baseline_mode),
            "upload_artifacts": False,
        }
        created = _post_json(f"{coordinator_url.rstrip('/')}/runs", create_payload, timeout_sec=30.0)
        run_id = str(created.get("run_id", ""))
        task_id = str(created.get("task_id", ""))
        if not run_id or not task_id:
            raise RuntimeError(f"coordinator /runs returned unexpected payload: {created}")
        summary["run"] = {"run_id": run_id, "task_id": task_id}
        _print_event(f"created run_id={run_id} task_id={task_id}")

        running_payload = _wait_for_run_status(
            coordinator_url=coordinator_url,
            run_id=run_id,
            target_statuses={"running"},
            timeout_sec=float(args.wait_running_timeout_sec),
            poll_sec=max(0.2, float(args.poll_sec)),
        )
        summary["running_snapshot"] = running_payload
        _print_event(f"run transitioned to running; injecting worker kill in {args.kill_delay_sec}s")
        time.sleep(max(0.0, float(args.kill_delay_sec)))
        worker1_exit = _stop_process(worker1, force=True)
        summary["worker1_exit_code"] = worker1_exit
        _print_event(f"worker1 stopped exit_code={worker1_exit}")

        requeue_processing = _post_json(
            f"{coordinator_url.rstrip('/')}/admin/requeue_processing?limit=100",
            {},
            timeout_sec=30.0,
        )
        requeue_stale = _post_json(
            f"{coordinator_url.rstrip('/')}/admin/requeue_stale?limit=100",
            {},
            timeout_sec=30.0,
        )
        summary["requeue_processing"] = requeue_processing
        summary["requeue_stale"] = requeue_stale
        _print_event(
            "requeue complete "
            f"processing_moved={requeue_processing.get('moved')} "
            f"stale_moved={requeue_stale.get('moved')}"
        )

        worker2 = _start_process(
            name="worker2",
            command=[
                sys.executable,
                "-m",
                "temporalci.coordinator.cli",
                "worker",
                "--coordinator-url",
                coordinator_url,
                "--worker-id",
                "recovery-worker-2",
                "--poll-interval-sec",
                str(float(args.worker_poll_interval_sec)),
            ],
            cwd=repo_root,
            env=env,
            logs_dir=logs_dir,
        )
        managed_processes.append(worker2)

        final_payload = _wait_for_run_status(
            coordinator_url=coordinator_url,
            run_id=run_id,
            target_statuses={"completed", "failed"},
            timeout_sec=float(args.completion_timeout_sec),
            poll_sec=max(0.2, float(args.poll_sec)),
        )
        summary["final_snapshot"] = final_payload
        final_status = str(final_payload.get("status", "unknown"))
        run_status = str((final_payload.get("payload") or {}).get("status", "unknown"))
        _print_event(f"run terminal status={final_status} run_status={run_status}")
        if final_status != "completed":
            raise RuntimeError(f"run ended with coordinator status={final_status}")
        if run_status != "PASS":
            raise RuntimeError(f"run payload status expected PASS, got {run_status}")

        verification = _verify_db_and_queue_state(
            postgres_dsn=args.postgres_dsn,
            redis_url=args.redis_url,
            queue_name=queue_name,
            processing_queue_name=processing_queue_name,
            run_id=run_id,
            require_empty_queues=require_empty_queues,
        )
        summary["verification"] = verification
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["error_type"] = exc.__class__.__name__
        _print_event(f"error: {exc}")
        exit_code = 1
    finally:
        stopped: list[dict[str, Any]] = []
        already_stopped: set[int] = set()
        for managed in reversed(managed_processes):
            proc_id = id(managed.process)
            if proc_id in already_stopped:
                continue
            already_stopped.add(proc_id)
            return_code = _stop_process(managed, force=False)
            stopped.append(
                {
                    "name": managed.name,
                    "exit_code": return_code,
                    "log_path": str(managed.log_path),
                }
            )
        summary["stopped_processes"] = stopped

        if started_compose and args.compose_down and compose_base_command:
            try:
                _run_command([*compose_base_command, "down"], cwd=repo_root, timeout_sec=120.0)
                summary["compose_down"] = "ok"
            except Exception as exc:  # noqa: BLE001
                summary["compose_down"] = f"error:{exc}"

        summary["finished_at_utc"] = utc_now_iso()
        atomic_write_json(summary_path, summary)
        _print_event(f"summary written: {summary_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
