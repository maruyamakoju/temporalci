from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.autopilot_96h import _safe_release_runtime_memory
from scripts.autopilot_96h import _safe_remove_pid_file
from scripts.autopilot_96h import _prune_model_runs
from scripts.autopilot_96h import _write_json
from scripts.autopilot_96h import _write_terminal_status
from scripts.stop_autopilot_background import _mark_status_stopped
from scripts.stop_autopilot_background import main as stop_main


def test_prune_model_runs_keeps_latest(tmp_path: Path) -> None:
    model_root = tmp_path / "project" / "suite" / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    # Timestamp-like run IDs sort lexicographically by recency.
    run_ids = [
        "20260208T100000000000Z",
        "20260208T100100000000Z",
        "20260208T100200000000Z",
    ]
    for run_id in run_ids:
        (model_root / run_id).mkdir(parents=True, exist_ok=True)
    (model_root / "runs.jsonl").write_text("", encoding="utf-8")

    payload = _prune_model_runs(model_root=model_root, keep_last=2)

    assert payload["enabled"] is True
    assert payload["deleted"] == 1
    assert payload["deleted_run_ids"] == ["20260208T100000000000Z"]
    remaining = sorted(child.name for child in model_root.iterdir() if child.is_dir())
    assert remaining == ["20260208T100100000000Z", "20260208T100200000000Z"]


def test_write_json_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    _write_json(path, {"state": "one"})
    _write_json(path, {"state": "two"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "two"


def test_write_terminal_status_includes_reason(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    _write_terminal_status(
        status_path=path,
        state="finished",
        cycle=3,
        deadline=0.0,
        stop_reason="unexpected_candidate_pass",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "finished"
    assert payload["cycle"] == 3
    assert payload["stop_reason"] == "unexpected_candidate_pass"
    assert "finished_at_utc" in payload


def test_mark_status_stopped_preserves_context(tmp_path: Path) -> None:
    status_path = tmp_path / "autopilot_status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "cycle": 124,
                "deadline_utc": "2026-02-12T13:51:12.553935+00:00",
                "last_status": "ok",
            }
        ),
        encoding="utf-8",
    )

    ok = _mark_status_stopped(
        status_path=status_path,
        pid=1234,
        reason="terminated_by_stop_script",
    )

    assert ok is True
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "stopped"
    assert payload["stopped_pid"] == 1234
    assert payload["stop_reason"] == "terminated_by_stop_script"
    assert payload["cycle"] == 124
    assert payload["deadline_utc"] == "2026-02-12T13:51:12.553935+00:00"
    assert payload["last_status"] == "ok"
    assert "finished_at_utc" in payload


def test_memory_cleanup_runs_without_cuda() -> None:
    payload = _safe_release_runtime_memory(clear_cuda_cache=False)
    assert "gc_collected" in payload
    assert payload["cuda_cache_cleared"] is False


def test_safe_remove_pid_file(tmp_path: Path) -> None:
    pid_file = tmp_path / "autopilot.pid"
    pid_file.write_text("{}", encoding="utf-8")
    _safe_remove_pid_file(pid_file)
    assert not pid_file.exists()


def test_stop_main_missing_pid_finished_state_is_idempotent(
    tmp_path: Path, monkeypatch: object
) -> None:
    pid_path = tmp_path / "autopilot.pid"
    status_path = tmp_path / "autopilot_status.json"
    status_path.write_text(json.dumps({"state": "finished"}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_autopilot_background.py",
            "--pid-file",
            str(pid_path),
            "--status-file",
            str(status_path),
        ],
    )

    assert stop_main() == 0


def test_stop_main_missing_pid_running_repairs_status(tmp_path: Path, monkeypatch: object) -> None:
    pid_path = tmp_path / "autopilot.pid"
    status_path = tmp_path / "autopilot_status.json"
    status_path.write_text(
        json.dumps({"state": "running", "cycle": 7, "deadline_utc": "2026-02-13T00:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stop_autopilot_background.py",
            "--pid-file",
            str(pid_path),
            "--status-file",
            str(status_path),
        ],
    )

    assert stop_main() == 0
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "stopped"
    assert payload["stop_reason"] == "pid_file_missing"
