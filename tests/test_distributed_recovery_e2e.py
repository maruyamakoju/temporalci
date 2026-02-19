from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from scripts.distributed_recovery_e2e import _build_inline_suite_yaml
from scripts.distributed_recovery_e2e import _service_reachable_from_url


def test_build_inline_suite_yaml_contains_expected_fields() -> None:
    payload = _build_inline_suite_yaml(sleep_sec=1.25)
    assert 'suite_name: "recovery_e2e"' in payload
    assert "sleep_sec: 1.25" in payload
    assert 'metric: "vbench_temporal.score"' in payload


def test_service_reachable_from_url_parses_default_port(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_is_port_open(*, host: str, port: int, timeout_sec: float = 1.0) -> bool:
        captured["host"] = host
        captured["port"] = port
        captured["timeout_sec"] = timeout_sec
        return True

    monkeypatch.setattr("scripts.distributed_recovery_e2e._is_port_open", fake_is_port_open)

    ok = _service_reachable_from_url(
        "postgresql://temporalci:temporalci@db.example/temporalci",
        default_port=5432,
    )
    assert ok is True
    assert captured["host"] == "db.example"
    assert captured["port"] == 5432


@pytest.mark.integration
def test_distributed_recovery_e2e_integration(tmp_path: Path) -> None:
    if os.getenv("RUN_DISTRIBUTED_E2E") != "1":
        pytest.skip("set RUN_DISTRIBUTED_E2E=1 to run distributed recovery E2E")

    use_compose = os.getenv("RUN_DISTRIBUTED_E2E_USE_COMPOSE", "0") == "1"
    if use_compose and not (shutil.which("docker") or shutil.which("docker-compose")):
        pytest.skip("docker compose not available")

    cmd = [
        sys.executable,
        "scripts/distributed_recovery_e2e.py",
        "--artifacts-dir",
        str(tmp_path / "distributed-recovery"),
        "--coordinator-port",
        str(int(os.getenv("RUN_DISTRIBUTED_E2E_PORT", "18080"))),
        "--task-sleep-sec",
        os.getenv("RUN_DISTRIBUTED_E2E_TASK_SLEEP_SEC", "12"),
        "--kill-delay-sec",
        os.getenv("RUN_DISTRIBUTED_E2E_KILL_DELAY_SEC", "1"),
        "--completion-timeout-sec",
        os.getenv("RUN_DISTRIBUTED_E2E_TIMEOUT_SEC", "300"),
    ]
    if not use_compose:
        cmd.append("--skip-compose")
    use_existing_coordinator = os.getenv("RUN_DISTRIBUTED_E2E_USE_EXISTING_COORDINATOR", "0") == "1"
    if use_existing_coordinator:
        cmd.append("--use-existing-coordinator")

    queue_name = str(os.getenv("RUN_DISTRIBUTED_E2E_QUEUE_NAME", "")).strip()
    if queue_name:
        cmd.extend(["--queue-name", queue_name])
    elif not use_existing_coordinator:
        cmd.extend(["--queue-name", f"temporalci:test:{uuid.uuid4().hex[:8]}"])

    for env_name, option in (
        ("RUN_DISTRIBUTED_E2E_POSTGRES_DSN", "--postgres-dsn"),
        ("RUN_DISTRIBUTED_E2E_REDIS_URL", "--redis-url"),
        ("RUN_DISTRIBUTED_E2E_COORDINATOR_URL", "--coordinator-url"),
    ):
        value = str(os.getenv(env_name, "")).strip()
        if value:
            cmd.extend([option, value])

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        details = f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n"
        pytest.fail(f"distributed recovery E2E failed with code={result.returncode}\n{details}")
