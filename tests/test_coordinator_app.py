from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


def test_create_run_accepts_json_body(monkeypatch: Any) -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception:  # noqa: BLE001
        pytest.skip("fastapi test client not available")

    from temporalci.coordinator import app as app_module

    @dataclass
    class _FakeSettings:
        default_artifacts_dir: str = "artifacts"
        task_lease_sec: int = 300

    class _FakeStore:
        last_suite_yaml: str | None = None
        last_baseline_mode: str | None = None

        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def init_schema(self) -> None:
            return None

        def create_run(self, req: Any) -> dict[str, Any]:
            _FakeStore.last_suite_yaml = req.suite_yaml
            _FakeStore.last_baseline_mode = req.baseline_mode
            return {"run_id": "run-1", "task_id": "task-1"}

        def get_run(self, run_id: str) -> dict[str, Any] | None:
            if run_id != "run-1":
                return None
            return {"run_id": "run-1", "status": "queued", "payload": {}}

        def claim_task(self, **_: Any) -> dict[str, Any] | None:
            return None

        def heartbeat_task(self, **_: Any) -> bool:
            return True

        def complete_task(self, **_: Any) -> None:
            return None

        def requeue_processing_tasks(self, limit: int = 100) -> int:
            _ = limit
            return 0

        def requeue_stale_running_tasks(self, limit: int = 100) -> int:
            _ = limit
            return 0

    def _fake_from_env(_: Any) -> _FakeSettings:
        return _FakeSettings()

    monkeypatch.setattr(app_module, "CoordinatorStore", _FakeStore)
    monkeypatch.setattr(
        app_module.CoordinatorSettings,
        "from_env",
        classmethod(_fake_from_env),
    )

    client = TestClient(app_module.create_app())
    response = client.post(
        "/runs",
        json={
            "suite_yaml": "version: 1\nproject: demo\nsuite_name: demo\nmodels: []\ntests: []\nmetrics: []\ngates: []\n",
            "suite_root": ".",
            "baseline_mode": "none",
            "fail_on_regression": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert _FakeStore.last_suite_yaml is not None
    assert _FakeStore.last_baseline_mode == "none"
