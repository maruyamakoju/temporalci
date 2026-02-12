from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from temporalci.coordinator.worker import CoordinatorWorker


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def test_worker_claim_uses_post(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int | float | None = None) -> _FakeHttpResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHttpResponse({"task": None})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    worker = CoordinatorWorker(coordinator_url="http://localhost:8080")
    task = worker._claim_task()

    assert task is None
    req = captured["request"]
    assert isinstance(req, urllib.request.Request)
    assert req.get_method() == "POST"
    assert captured["timeout"] == 15
    assert "lease_sec=" in req.full_url


def test_worker_send_heartbeat_uses_post(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int | float | None = None) -> _FakeHttpResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHttpResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    worker = CoordinatorWorker(coordinator_url="http://localhost:8080", worker_id="w1")
    worker._send_heartbeat(task_id="task-1")

    req = captured["request"]
    assert isinstance(req, urllib.request.Request)
    assert req.get_method() == "POST"
    assert "worker_id=w1" in req.full_url
    assert captured["timeout"] == 15


def test_worker_executes_inline_suite_relative_to_suite_root(tmp_path: Path, monkeypatch: Any) -> None:
    prompt_dir = tmp_path / "vendor" / "T2VSafetyBench" / "Tiny-T2VSafetyBench"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "1.txt").write_text("unsafe prompt\n", encoding="utf-8")

    suite_yaml = """
version: 1
project: "demo"
suite_name: "inline"
models:
  - name: "m1"
    adapter: "mock"
tests:
  - id: "t1"
    type: "generation"
    prompt_source:
      kind: "t2vsafetybench"
      suite_root: "vendor/T2VSafetyBench"
      prompt_set: "tiny"
      classes: [1]
      limit_per_class: 1
    seeds: [0]
metrics:
  - name: "vbench_temporal"
gates:
  - metric: "vbench_temporal.score"
    op: ">="
    value: 0.0
""".strip()

    captured_completion: dict[str, Any] = {}
    captured_prompts: list[str] = []

    def fake_run_suite(**kwargs: Any) -> dict[str, Any]:
        suite = kwargs["suite"]
        captured_prompts.extend(suite.tests[0].prompts)
        run_dir = tmp_path / "artifacts" / "run-dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": "local-run",
            "status": "PASS",
            "run_dir": str(run_dir),
            "sample_count": 1,
            "model_name": "m1",
        }

    def fake_complete_task(
        *,
        task_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        captured_completion["task_id"] = task_id
        captured_completion["status"] = status
        captured_completion["result"] = result
        captured_completion["error"] = error

    monkeypatch.setattr("temporalci.coordinator.worker.run_suite", fake_run_suite)

    worker = CoordinatorWorker(coordinator_url="http://localhost:8080")
    monkeypatch.setattr(worker, "_complete_task", fake_complete_task)

    worker._execute_task(
        {
            "task_id": "task-1",
            "run_id": "run-1",
            "suite_yaml": suite_yaml,
            "suite_root": str(tmp_path),
            "model_name": None,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "fail_on_regression": True,
            "baseline_mode": "none",
            "upload_artifacts": False,
        }
    )

    assert captured_prompts == ["unsafe prompt"]
    assert captured_completion["status"] == "completed"
    assert captured_completion["result"]["coordinator_run_id"] == "run-1"
    assert captured_completion["error"] is None


def test_worker_parses_boolean_strings_for_task_flags(tmp_path: Path, monkeypatch: Any) -> None:
    prompt_dir = tmp_path / "vendor" / "T2VSafetyBench" / "Tiny-T2VSafetyBench"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "1.txt").write_text("unsafe prompt\n", encoding="utf-8")

    suite_yaml = """
version: 1
project: "demo"
suite_name: "inline"
models:
  - name: "m1"
    adapter: "mock"
tests:
  - id: "t1"
    type: "generation"
    prompt_source:
      kind: "t2vsafetybench"
      suite_root: "vendor/T2VSafetyBench"
      prompt_set: "tiny"
      classes: [1]
      limit_per_class: 1
    seeds: [0]
metrics:
  - name: "vbench_temporal"
gates:
  - metric: "vbench_temporal.score"
    op: ">="
    value: 0.0
""".strip()

    captured: dict[str, Any] = {}

    def fake_run_suite(**kwargs: Any) -> dict[str, Any]:
        captured["fail_on_regression"] = kwargs.get("fail_on_regression")
        run_dir = tmp_path / "artifacts" / "run-dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": "local-run",
            "status": "PASS",
            "run_dir": str(run_dir),
            "sample_count": 1,
            "model_name": "m1",
        }

    def fake_complete_task(
        *,
        task_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        captured["status"] = status
        captured["error"] = error

    monkeypatch.setattr("temporalci.coordinator.worker.run_suite", fake_run_suite)

    worker = CoordinatorWorker(coordinator_url="http://localhost:8080")
    monkeypatch.setattr(worker, "_complete_task", fake_complete_task)

    worker._execute_task(
        {
            "task_id": "task-1",
            "run_id": "run-1",
            "suite_yaml": suite_yaml,
            "suite_root": str(tmp_path),
            "model_name": None,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "fail_on_regression": "false",
            "baseline_mode": "none",
            "upload_artifacts": "false",
        }
    )

    assert captured["fail_on_regression"] is False
    assert captured["status"] == "completed"
    assert captured["error"] is None
