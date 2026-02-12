from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.autopilot_summary import _build_summary
from scripts.autopilot_summary import _quantile
from scripts.autopilot_summary import main as summary_main


def test_quantile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _quantile(values, 0.5) == 2.5
    assert _quantile(values, 0.0) == 1.0
    assert _quantile(values, 1.0) == 4.0


def test_build_summary_includes_run_and_telemetry_stats(tmp_path: Path) -> None:
    runs_rows = [
        {
            "event": "cycle_end",
            "status": "ok",
            "started_at_utc": "2026-02-12T00:00:00+00:00",
            "finished_at_utc": "2026-02-12T00:10:00+00:00",
        },
        {
            "event": "cycle_end",
            "status": "error",
            "started_at_utc": "2026-02-12T01:00:00+00:00",
            "finished_at_utc": "2026-02-12T01:30:00+00:00",
        },
    ]
    telemetry_rows = [
        {
            "timestamp_utc": "2026-02-12T00:00:00+00:00",
            "rss_bytes": 100.0,
            "gpu_mem_mb": 10.0,
            "gpu_util": 40.0,
        },
        {
            "timestamp_utc": "2026-02-12T01:00:00+00:00",
            "rss_bytes": 300.0,
            "gpu_mem_mb": 20.0,
            "gpu_util": 60.0,
            "telemetry_stop_reason": "pid_dead",
        },
    ]
    summary = _build_summary(
        artifacts_dir=tmp_path,
        runs_rows=runs_rows,
        telemetry_rows=telemetry_rows,
        status_payload={"state": "running", "cycle": 2},
    )

    assert summary["runs"]["cycle_end_total"] == 2
    assert summary["runs"]["cycle_status_counts"]["ok"] == 1
    assert summary["runs"]["cycle_status_counts"]["error"] == 1
    assert summary["telemetry"]["samples_total"] == 2
    assert summary["telemetry"]["rss_bytes"]["first"] == 100.0
    assert summary["telemetry"]["rss_bytes"]["last"] == 300.0
    assert summary["telemetry"]["stop_reason_counts"]["pid_dead"] == 1


def test_summary_main_writes_json_and_md(tmp_path: Path, monkeypatch: object) -> None:
    runs_path = tmp_path / "autopilot_runs.jsonl"
    runs_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "cycle_end",
                        "status": "ok",
                        "started_at_utc": "2026-02-12T00:00:00+00:00",
                        "finished_at_utc": "2026-02-12T00:05:00+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry_path = tmp_path / "autopilot_telemetry.jsonl"
    telemetry_path.write_text(
        json.dumps({"timestamp_utc": "2026-02-12T00:01:00+00:00", "rss_bytes": 100}) + "\n",
        encoding="utf-8",
    )
    status_path = tmp_path / "autopilot_status.json"
    status_path.write_text(json.dumps({"state": "running", "cycle": 1}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_summary.py",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert summary_main() == 0
    assert (tmp_path / "autopilot_summary.json").exists()
    assert (tmp_path / "autopilot_summary.md").exists()
