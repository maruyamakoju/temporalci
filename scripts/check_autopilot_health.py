from __future__ import annotations

import argparse
import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from temporalci.autopilot_utils import pid_exists
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import safe_write_json
from temporalci.autopilot_utils import utc_now_iso

def _iso_age_sec(raw: Any) -> float | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - value).total_seconds())


def _tail_line(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:  # noqa: BLE001
        return None
    if not lines:
        return None
    return lines[-1]


def _status_fields(
    *,
    status_payload: dict[str, Any],
    pid_alive: bool,
    max_stale_sec: int,
) -> dict[str, Any]:
    state = str(status_payload.get("state", "unknown"))
    phase = str(status_payload.get("phase", ""))
    cycle = int(status_payload.get("cycle", 0) or 0)
    age_candidates = [
        _iso_age_sec(status_payload.get("finished_at_utc")),
        _iso_age_sec(status_payload.get("started_at_utc")),
    ]
    age_candidates = [value for value in age_candidates if value is not None]
    stale_age_sec = age_candidates[0] if age_candidates else None
    stale = stale_age_sec is not None and stale_age_sec > float(max_stale_sec)
    healthy = pid_alive and state == "running" and not stale
    return {
        "state": state,
        "phase": phase,
        "cycle": cycle,
        "stale_age_sec": stale_age_sec,
        "stale": stale,
        "healthy": healthy,
    }


def _repair_status(
    *,
    status_path: Path,
    status_payload: dict[str, Any],
    pid: int,
    pid_alive: bool,
    stale: bool,
    repair_state: str,
) -> tuple[bool, str]:
    reason = "running_pid_dead"
    if stale and not pid_alive:
        reason = "running_stale_pid_dead"
    elif stale:
        reason = "running_stale"

    payload = dict(status_payload)
    payload["state"] = repair_state
    payload["phase"] = "repair"
    payload["finished_at_utc"] = utc_now_iso()
    payload["repair_reason"] = reason
    payload["repair_source"] = "check_autopilot_health"
    payload["repair_pid_alive"] = pid_alive
    payload["repair_stale"] = stale
    if pid > 0:
        payload["stopped_pid"] = pid
    ok = safe_write_json(status_path, payload)
    return ok, reason


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TemporalCI autopilot health.")
    parser.add_argument("--artifacts-dir", default="artifacts/autopilot-96h")
    parser.add_argument("--pid-file", default="autopilot.pid")
    parser.add_argument("--status-file", default="autopilot_status.json")
    parser.add_argument("--runs-file", default="autopilot_runs.jsonl")
    parser.add_argument("--max-stale-sec", type=int, default=1800)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair stale/invalid running status to a terminal state.",
    )
    parser.add_argument(
        "--repair-state",
        default="stale_stopped",
        choices=["stale_stopped", "stopped"],
        help="State written when --repair is applied.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    pid_path = Path(args.pid_file)
    if not pid_path.is_absolute():
        pid_path = artifacts_dir / pid_path
    status_path = Path(args.status_file)
    if not status_path.is_absolute():
        status_path = artifacts_dir / status_path
    runs_path = Path(args.runs_file)
    if not runs_path.is_absolute():
        runs_path = artifacts_dir / runs_path

    pid_payload = read_json_dict(pid_path) or {}
    status_payload = read_json_dict(status_path) or {}

    pid = int(pid_payload.get("pid", 0) or 0)
    pid_alive = pid_exists(pid) if pid > 0 else False
    fields = _status_fields(
        status_payload=status_payload,
        pid_alive=pid_alive,
        max_stale_sec=int(args.max_stale_sec),
    )
    state = str(fields["state"])
    phase = str(fields["phase"])
    cycle = int(fields["cycle"])
    stale_age_sec = fields["stale_age_sec"]
    stale = bool(fields["stale"])
    healthy = bool(fields["healthy"])

    repair_attempted = False
    repair_applied = False
    repair_reason = ""
    if args.repair and state == "running" and (stale or not pid_alive):
        repair_attempted = True
        repair_applied, repair_reason = _repair_status(
            status_path=status_path,
            status_payload=status_payload,
            pid=pid,
            pid_alive=pid_alive,
            stale=stale,
            repair_state=str(args.repair_state),
        )
        if repair_applied:
            status_payload = read_json_dict(status_path) or status_payload
            fields = _status_fields(
                status_payload=status_payload,
                pid_alive=pid_alive,
                max_stale_sec=int(args.max_stale_sec),
            )
            state = str(fields["state"])
            phase = str(fields["phase"])
            cycle = int(fields["cycle"])
            stale_age_sec = fields["stale_age_sec"]
            stale = bool(fields["stale"])
            healthy = bool(fields["healthy"])

    tail = _tail_line(runs_path)
    payload = {
        "healthy": healthy,
        "pid": pid,
        "pid_alive": pid_alive,
        "state": state,
        "phase": phase,
        "cycle": cycle,
        "stale_age_sec": stale_age_sec,
        "stale_threshold_sec": int(args.max_stale_sec),
        "stale": stale,
        "pid_file": str(pid_path),
        "status_file": str(status_path),
        "runs_file": str(runs_path),
        "last_runs_line": tail,
        "repair_attempted": repair_attempted,
        "repair_applied": repair_applied,
        "repair_reason": repair_reason or None,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"healthy={payload['healthy']} pid_alive={pid_alive} "
            f"state={state} phase={phase} cycle={cycle} stale={stale}"
        )
        if stale_age_sec is not None:
            print(f"stale_age_sec={stale_age_sec:.1f} threshold={int(args.max_stale_sec)}")
        print(f"pid_file={pid_path}")
        print(f"status_file={status_path}")
        print(f"runs_file={runs_path}")

    return 0 if payload["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
