from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from temporalci.autopilot_utils import pid_exists
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import safe_write_json
from temporalci.autopilot_utils import terminate_pid
from temporalci.autopilot_utils import utc_now_iso

TERMINAL_STATES = {"finished", "stopped", "stale_stopped"}


def _mark_status_stopped(*, status_path: Path, pid: int, reason: str) -> bool:
    previous = read_json_dict(status_path) or {}
    payload: dict[str, Any] = {
        "state": "stopped",
        "finished_at_utc": utc_now_iso(),
        "stopped_pid": pid,
        "stop_reason": reason,
    }
    if "cycle" in previous:
        payload["cycle"] = previous["cycle"]
    if "deadline_utc" in previous:
        payload["deadline_utc"] = previous["deadline_utc"]
    if "last_status" in previous:
        payload["last_status"] = previous["last_status"]
    return safe_write_json(status_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop detached TemporalCI autopilot process.")
    parser.add_argument("--pid-file", default="artifacts/autopilot-96h/autopilot.pid")
    parser.add_argument(
        "--status-file",
        default="",
        help="Autopilot status JSON path. Defaults to sibling autopilot_status.json.",
    )
    parser.add_argument("--keep-pid-file", action="store_true")
    args = parser.parse_args()

    pid_path = Path(args.pid_file).resolve()
    status_path = (
        Path(args.status_file).resolve()
        if args.status_file
        else pid_path.parent / "autopilot_status.json"
    )
    status_payload = read_json_dict(status_path) or {}
    state = str(status_payload.get("state", "unknown"))
    if not pid_path.exists():
        if state in TERMINAL_STATES:
            print(f"autopilot already stopped (state={state})")
            return 0
        if state == "running":
            _mark_status_stopped(status_path=status_path, pid=0, reason="pid_file_missing")
            print("autopilot status repaired: pid file missing, state marked stopped")
            return 0
        print(f"pid file not found: {pid_path}")
        return 1

    payload = read_json_dict(pid_path)
    if payload is None:
        if state in TERMINAL_STATES:
            print(f"autopilot already stopped (state={state})")
            return 0
        if state == "running":
            _mark_status_stopped(status_path=status_path, pid=0, reason="invalid_pid_file_json")
            print("autopilot status repaired: invalid pid json, state marked stopped")
            if not args.keep_pid_file:
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            return 0
        print(f"invalid pid file json: {pid_path}")
        return 1
    pid = int(payload.get("pid", 0) or 0)
    if pid <= 0:
        if state in TERMINAL_STATES:
            print(f"autopilot already stopped (state={state})")
            return 0
        if state == "running":
            _mark_status_stopped(status_path=status_path, pid=0, reason="invalid_pid_file_payload")
            print("autopilot status repaired: invalid pid payload, state marked stopped")
            if not args.keep_pid_file:
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            return 0
        print(f"invalid pid file payload: {pid_path}")
        return 1
    if not pid_exists(pid):
        print(f"autopilot already stopped pid={pid}")
        _mark_status_stopped(status_path=status_path, pid=pid, reason="already_stopped")
        if not args.keep_pid_file:
            try:
                pid_path.unlink()
            except OSError:
                pass
        return 0

    ok = terminate_pid(pid)
    if not ok:
        print(f"failed to terminate pid={pid}")
        return 1

    print(f"autopilot_stopped pid={pid}")
    _mark_status_stopped(status_path=status_path, pid=pid, reason="terminated_by_stop_script")
    if not args.keep_pid_file:
        try:
            pid_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
