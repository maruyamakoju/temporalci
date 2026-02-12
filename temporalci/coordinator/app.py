from __future__ import annotations

from typing import Any

from temporalci.constants import BASELINE_MODES
from temporalci.coordinator.config import CoordinatorSettings
from temporalci.coordinator.store import CoordinatorStore, CreateRunRequest
from temporalci.errors import CoordinatorError


def create_app() -> Any:
    try:
        from fastapi import FastAPI
        from fastapi import HTTPException
        from pydantic import BaseModel
    except Exception as exc:  # noqa: BLE001
        raise CoordinatorError(
            "fastapi and pydantic are required for coordinator app. "
            "Install optional dependencies for distributed mode."
        ) from exc

    settings = CoordinatorSettings.from_env()
    store = CoordinatorStore(settings=settings)
    store.init_schema()

    class RunCreateBody(BaseModel):
        suite_path: str | None = None
        suite_yaml: str | None = None
        suite_root: str | None = None
        model_name: str | None = None
        artifacts_dir: str | None = None
        fail_on_regression: bool = True
        baseline_mode: str = "latest_pass"
        upload_artifacts: bool = False

    class TaskCompleteBody(BaseModel):
        status: str
        result: dict[str, Any] | None = None
        error: str | None = None

    app = FastAPI(title="TemporalCI Coordinator", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs")
    def create_run(body: RunCreateBody) -> dict[str, Any]:
        if not body.suite_path and not body.suite_yaml:
            raise HTTPException(status_code=400, detail="one of suite_path or suite_yaml is required")
        if body.baseline_mode not in BASELINE_MODES:
            available = ", ".join(sorted(BASELINE_MODES))
            raise HTTPException(
                status_code=400,
                detail=f"invalid baseline_mode '{body.baseline_mode}'. choose: {available}",
            )
        req = CreateRunRequest(
            suite_path=body.suite_path,
            suite_yaml=body.suite_yaml,
            suite_root=body.suite_root,
            model_name=body.model_name,
            artifacts_dir=body.artifacts_dir or settings.default_artifacts_dir,
            fail_on_regression=body.fail_on_regression,
            baseline_mode=body.baseline_mode,
            upload_artifacts=body.upload_artifacts,
        )
        return store.create_run(req=req)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        payload = store.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="run not found")
        return payload

    @app.post("/workers/{worker_id}/claim")
    def claim(worker_id: str, timeout_sec: int = 5, lease_sec: int | None = None) -> dict[str, Any]:
        task = store.claim_task(
            worker_id=worker_id,
            timeout_sec=timeout_sec,
            lease_sec=lease_sec or settings.task_lease_sec,
        )
        if task is None:
            return {"task": None}
        return {"task": task}

    @app.post("/tasks/{task_id}/heartbeat")
    def heartbeat(task_id: str, worker_id: str, lease_sec: int | None = None) -> dict[str, bool]:
        ok = store.heartbeat_task(
            task_id=task_id,
            worker_id=worker_id,
            lease_sec=lease_sec or settings.task_lease_sec,
        )
        return {"ok": ok}

    @app.post("/tasks/{task_id}/complete")
    def complete(task_id: str, body: TaskCompleteBody) -> dict[str, bool]:
        store.complete_task(
            task_id=task_id,
            status=body.status,
            result=body.result,
            error=body.error,
        )
        return {"ok": True}

    @app.post("/admin/requeue_processing")
    def requeue_processing(limit: int = 100) -> dict[str, int]:
        moved = store.requeue_processing_tasks(limit=limit)
        return {"moved": moved}

    @app.post("/admin/requeue_stale")
    def requeue_stale(limit: int = 100) -> dict[str, int]:
        moved = store.requeue_stale_running_tasks(limit=limit)
        return {"moved": moved}

    return app
