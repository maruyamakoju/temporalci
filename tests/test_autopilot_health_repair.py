from __future__ import annotations

import json
import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

from scripts.check_autopilot_health import main as health_main


def test_health_repair_marks_stale_running_as_stale_stopped(
    tmp_path: Path, monkeypatch: object
) -> None:
    status_path = tmp_path / "autopilot_status.json"
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "cycle": 12,
                "phase": "cycle_end",
                "started_at_utc": old.isoformat(),
                "deadline_utc": "2026-02-13T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_autopilot_health.py",
            "--artifacts-dir",
            str(tmp_path),
            "--max-stale-sec",
            "10",
            "--repair",
            "--repair-state",
            "stale_stopped",
        ],
    )

    assert health_main() == 1
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "stale_stopped"
    assert payload["repair_reason"] in {"running_stale", "running_stale_pid_dead"}


def test_health_repair_skips_non_running_state(tmp_path: Path, monkeypatch: object) -> None:
    status_path = tmp_path / "autopilot_status.json"
    status_path.write_text(json.dumps({"state": "finished"}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_autopilot_health.py",
            "--artifacts-dir",
            str(tmp_path),
            "--repair",
        ],
    )

    assert health_main() == 1
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["state"] == "finished"
