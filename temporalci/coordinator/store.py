from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from temporalci.coordinator.config import CoordinatorSettings
from temporalci.errors import CoordinatorError
from temporalci.utils import utc_now_iso


def _load_psycopg() -> Any:
    try:
        import psycopg
    except Exception as exc:  # noqa: BLE001
        raise CoordinatorError(
            "psycopg is required for coordinator store. "
            "Install optional dependencies for distributed mode."
        ) from exc
    return psycopg


def _load_redis_client() -> Any:
    try:
        import redis
    except Exception as exc:  # noqa: BLE001
        raise CoordinatorError(
            "redis client is required for coordinator store. "
            "Install optional dependencies for distributed mode."
        ) from exc
    return redis


@dataclass(slots=True)
class CreateRunRequest:
    suite_path: str | None
    suite_yaml: str | None
    suite_root: str | None
    model_name: str | None
    artifacts_dir: str
    fail_on_regression: bool
    baseline_mode: str
    upload_artifacts: bool


class CoordinatorStore:
    def __init__(self, settings: CoordinatorSettings) -> None:
        self.settings = settings
        psycopg = _load_psycopg()
        redis_module = _load_redis_client()
        self._conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
        self._redis = redis_module.from_url(settings.redis_url, decode_responses=True)

    def init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            suite_path TEXT NOT NULL,
            model_name TEXT NULL,
            artifacts_dir TEXT NOT NULL,
            fail_on_regression BOOLEAN NOT NULL,
            baseline_mode TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            payload JSONB NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            status TEXT NOT NULL,
            worker_id TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ NULL,
            finished_at TIMESTAMPTZ NULL,
            leased_until TIMESTAMPTZ NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            payload JSONB NOT NULL,
            result JSONB NULL,
            error TEXT NULL
        );

        ALTER TABLE tasks ADD COLUMN IF NOT EXISTS leased_until TIMESTAMPTZ NULL;
        ALTER TABLE tasks ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;

        CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status_leased_until ON tasks(status, leased_until);
        """
        with self._conn.cursor() as cur:
            cur.execute(sql)

    def create_run(self, req: CreateRunRequest) -> dict[str, Any]:
        if not req.suite_path and not req.suite_yaml:
            raise ValueError("one of suite_path or suite_yaml is required")

        run_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        suite_path = req.suite_path or "__inline_suite__"
        payload = {
            "run_id": run_id,
            "task_id": task_id,
            "suite_path": suite_path,
            "suite_yaml": req.suite_yaml,
            "suite_root": req.suite_root,
            "model_name": req.model_name,
            "artifacts_dir": req.artifacts_dir,
            "fail_on_regression": req.fail_on_regression,
            "baseline_mode": req.baseline_mode,
            "upload_artifacts": req.upload_artifacts,
        }
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, status, suite_path, model_name, artifacts_dir,
                    fail_on_regression, baseline_mode, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                """,
                (
                    run_id,
                    "queued",
                    suite_path,
                    req.model_name,
                    req.artifacts_dir,
                    req.fail_on_regression,
                    req.baseline_mode,
                    json.dumps(payload),
                ),
            )
            cur.execute(
                """
                INSERT INTO tasks (task_id, run_id, status, payload)
                VALUES (%s, %s, %s, %s::jsonb);
                """,
                (task_id, run_id, "queued", json.dumps(payload)),
            )
        self._redis.lpush(self.settings.queue_name, task_id)
        return payload

    def claim_task(
        self,
        *,
        worker_id: str,
        timeout_sec: int = 5,
        lease_sec: int,
    ) -> dict[str, Any] | None:
        task_id = self._redis.brpoplpush(
            self.settings.queue_name,
            self.settings.processing_queue_name,
            timeout=timeout_sec,
        )
        if not task_id:
            return None
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET
                    status=%s,
                    worker_id=%s,
                    started_at=NOW(),
                    leased_until=NOW() + (%s || ' seconds')::interval,
                    attempts=attempts + 1
                WHERE task_id=%s
                RETURNING run_id, payload;
                """,
                ("running", worker_id, int(lease_sec), task_id),
            )
            row = cur.fetchone()
        if not row:
            self._redis.lrem(self.settings.processing_queue_name, 0, task_id)
            return None
        run_id = row[0]
        payload_raw = row[1]
        payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
        payload["task_id"] = task_id
        payload["run_id"] = run_id
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status=%s, updated_at=NOW()
                WHERE run_id=%s;
                """,
                ("running", run_id),
            )
        return payload

    def heartbeat_task(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_sec: int,
    ) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET leased_until=NOW() + (%s || ' seconds')::interval
                WHERE task_id=%s AND worker_id=%s AND status='running'
                RETURNING task_id;
                """,
                (int(lease_sec), task_id, worker_id),
            )
            row = cur.fetchone()
        return row is not None

    def complete_task(
        self,
        *,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        run_status = "completed" if status == "completed" else "failed"
        self._redis.lrem(self.settings.processing_queue_name, 0, task_id)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status=%s, finished_at=NOW(), result=%s::jsonb, error=%s
                WHERE task_id=%s
                RETURNING run_id;
                """,
                (
                    status,
                    json.dumps(result or {}),
                    error,
                    task_id,
                ),
            )
            row = cur.fetchone()
            if not row:
                return
            run_id = row[0]
            cur.execute(
                """
                UPDATE runs
                SET status=%s, updated_at=NOW(), payload=%s::jsonb
                WHERE run_id=%s;
                """,
                (
                    run_status,
                    json.dumps(result or {"error": error, "updated_at": utc_now_iso()}),
                    run_id,
                ),
            )

    def _mark_task_queued(self, *, task_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET
                    status='queued',
                    worker_id=NULL,
                    started_at=NULL,
                    leased_until=NULL
                WHERE task_id=%s
                RETURNING run_id;
                """,
                (task_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            run_id = str(row[0])
            cur.execute(
                """
                UPDATE runs
                SET status='queued', updated_at=NOW()
                WHERE run_id=%s AND status NOT IN ('completed', 'failed');
                """,
                (run_id,),
            )
        return True

    def requeue_processing_tasks(self, limit: int = 100) -> int:
        moved = 0
        for _ in range(limit):
            task_id = self._redis.rpop(self.settings.processing_queue_name)
            if not task_id:
                break
            if not self._mark_task_queued(task_id=task_id):
                continue
            self._redis.lpush(self.settings.queue_name, task_id)
            moved += 1
        return moved

    def requeue_stale_running_tasks(self, limit: int = 100) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id
                FROM tasks
                WHERE status='running' AND leased_until IS NOT NULL AND leased_until < NOW()
                ORDER BY leased_until ASC
                LIMIT %s;
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
            stale_task_ids = [str(row[0]) for row in rows]

        if not stale_task_ids:
            return 0

        moved = 0
        for task_id in stale_task_ids:
            marked = self._mark_task_queued(task_id=task_id)
            self._redis.lrem(self.settings.processing_queue_name, 0, task_id)
            if not marked:
                continue
            self._redis.lpush(self.settings.queue_name, task_id)
            moved += 1
        return moved

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, status, suite_path, model_name, artifacts_dir,
                       fail_on_regression, baseline_mode, created_at, updated_at, payload
                FROM runs WHERE run_id=%s;
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        payload_raw = row[9]
        payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw or "{}")
        return {
            "run_id": row[0],
            "status": row[1],
            "suite_path": row[2],
            "model_name": row[3],
            "artifacts_dir": row[4],
            "fail_on_regression": row[5],
            "baseline_mode": row[6],
            "created_at": str(row[7]),
            "updated_at": str(row[8]),
            "payload": payload,
        }
