from __future__ import annotations

import json
import threading
import tempfile
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Any, cast

from temporalci.config import load_suite
from temporalci.coordinator.config import CoordinatorSettings
from temporalci.coordinator.minio_artifacts import MinioArtifactUploader
from temporalci.engine import run_suite
from temporalci.utils import as_bool


class CoordinatorWorker:
    def __init__(
        self,
        *,
        coordinator_url: str,
        worker_id: str | None = None,
        poll_interval_sec: float = 2.0,
        settings: CoordinatorSettings | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.worker_id = worker_id or str(uuid.uuid4())
        self.poll_interval_sec = poll_interval_sec
        self.settings = settings or CoordinatorSettings.from_env()

    def run_forever(self) -> None:
        while True:
            task = self._claim_task()
            if task is None:
                time.sleep(self.poll_interval_sec)
                continue
            self._execute_task(task)

    def _claim_task(self) -> dict[str, Any] | None:
        url = (
            f"{self.coordinator_url}/workers/{self.worker_id}/claim"
            f"?timeout_sec=2&lease_sec={self.settings.task_lease_sec}"
        )
        try:
            request = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        task = payload.get("task")
        if not isinstance(task, dict):
            return None
        return task

    def _execute_task(self, task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        run_id = str(task["run_id"])
        status = "completed"
        error: str | None = None
        result: dict[str, Any] | None = None
        temp_suite_file: Path | None = None
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            kwargs={"task_id": task_id, "stop_event": heartbeat_stop},
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            suite_yaml = task.get("suite_yaml")
            if isinstance(suite_yaml, str) and suite_yaml.strip():
                temp_parent = self._resolve_temp_suite_parent(task=task)
                temp_parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".yaml",
                    encoding="utf-8",
                    delete=False,
                    dir=str(temp_parent),
                ) as handle:
                    handle.write(suite_yaml)
                    suite_file = Path(handle.name)
                    temp_suite_file = suite_file
            else:
                suite_file = Path(task["suite_path"])

            suite = load_suite(suite_file)
            result = cast("dict[str, Any]", run_suite(
                suite=suite,
                model_name=task.get("model_name"),
                artifacts_dir=task.get("artifacts_dir", self.settings.default_artifacts_dir),
                fail_on_regression=as_bool(task.get("fail_on_regression", True), default=True),
                baseline_mode=str(task.get("baseline_mode", "latest_pass")),
            ))
            result["coordinator_run_id"] = run_id

            if as_bool(task.get("upload_artifacts", False), default=False):
                uploader = MinioArtifactUploader(settings=self.settings)
                run_dir = Path(str(result["run_dir"]))
                keys = uploader.upload_run_directory(
                    run_dir=run_dir,
                    prefix=f"runs/{run_id}",
                )
                result["artifact_keys"] = keys
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
            if temp_suite_file is not None:
                try:
                    temp_suite_file.unlink()
                except OSError:
                    pass

        try:
            self._complete_task(task_id=task_id, status=status, result=result, error=error)
        except Exception:  # noqa: BLE001
            # Keep worker loop alive even if callback endpoint is temporarily unavailable.
            pass

    def _complete_task(
        self,
        *,
        task_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        payload = {
            "status": status,
            "result": result,
            "error": error,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.coordinator_url}/tasks/{task_id}/complete",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30):
            pass

    def _heartbeat_loop(self, *, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(timeout=float(self.settings.heartbeat_interval_sec)):
            try:
                self._send_heartbeat(task_id=task_id)
            except Exception:  # noqa: BLE001
                # Heartbeat failures should not stop ongoing run execution.
                continue

    def _send_heartbeat(self, *, task_id: str) -> None:
        request = urllib.request.Request(
            (
                f"{self.coordinator_url}/tasks/{task_id}/heartbeat"
                f"?worker_id={self.worker_id}&lease_sec={self.settings.task_lease_sec}"
            ),
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15):
            pass

    def _resolve_temp_suite_parent(self, *, task: dict[str, Any]) -> Path:
        suite_root = task.get("suite_root")
        if isinstance(suite_root, str) and suite_root.strip():
            candidate = Path(suite_root).expanduser()
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        return Path.cwd()
