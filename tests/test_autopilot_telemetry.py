from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.autopilot_telemetry import _compute_stop_reason
from scripts.autopilot_telemetry import _compute_cycle_time_sec_from_runs_tail
from scripts.autopilot_telemetry import _hash_last_runs_line
from scripts.autopilot_telemetry import _parse_gpu_apps_csv
from scripts.autopilot_telemetry import _parse_gpu_util_csv
from scripts.autopilot_telemetry import main as telemetry_main


def test_hash_last_runs_line(tmp_path: Path) -> None:
    runs_path = tmp_path / "autopilot_runs.jsonl"
    runs_path.write_text('{"event":"cycle_start"}\n{"event":"cycle_end"}\n', encoding="utf-8")

    digest = _hash_last_runs_line(runs_path)

    assert digest is not None
    assert len(digest) == 16


def test_parse_gpu_apps_csv_sums_matching_pid() -> None:
    raw = "\n".join(
        [
            "111, 100",
            "222, 50",
            "111, 25",
        ]
    )

    assert _parse_gpu_apps_csv(raw=raw, pid=111) == 125.0
    assert _parse_gpu_apps_csv(raw=raw, pid=999) == 0.0


def test_parse_gpu_util_csv_averages_rows() -> None:
    raw = "\n".join(
        [
            "40 %",
            "60 %",
        ]
    )
    assert _parse_gpu_util_csv(raw=raw) == 50.0


def test_compute_stop_reason_prefers_max_samples() -> None:
    sample = {
        "pid": 1234,
        "pid_alive": True,
        "state": "running",
    }
    reason = _compute_stop_reason(
        sample=sample,
        sample_count=5,
        max_samples=5,
        stop_on_pid_dead=True,
        stop_on_terminal_state=True,
    )
    assert reason == "max_samples_reached"


def test_compute_cycle_time_sec_from_runs_tail(tmp_path: Path) -> None:
    runs_path = tmp_path / "autopilot_runs.jsonl"
    runs_path.write_text(
        '{"event":"cycle_start","started_at_utc":"2026-02-10T00:00:00+00:00"}\n'
        '{"event":"cycle_end","started_at_utc":"2026-02-10T00:00:00+00:00","finished_at_utc":"2026-02-10T00:00:03.250000+00:00"}\n',
        encoding="utf-8",
    )

    value = _compute_cycle_time_sec_from_runs_tail(runs_path)

    assert value == 3.25


def test_telemetry_main_writes_stop_reason_with_max_samples(
    tmp_path: Path, monkeypatch: object
) -> None:
    out_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_telemetry.py",
            "--artifacts-dir",
            str(tmp_path),
            "--output-file",
            str(out_path),
            "--max-samples",
            "1",
            "--no-stop-on-pid-dead",
            "--no-stop-on-terminal-state",
        ],
    )

    assert telemetry_main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert payload["telemetry_stop_reason"] == "max_samples_reached"
