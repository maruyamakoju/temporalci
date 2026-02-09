from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from temporalci.autopilot_utils import pid_exists
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import terminate_pid
from temporalci.autopilot_utils import utc_now_iso

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch TemporalCI 96h autopilot in detached background mode."
    )
    parser.add_argument("--hours", type=float, default=96.0)
    parser.add_argument("--cooldown-sec", type=float, default=300.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--baseline", default="examples/svd_regression_fast.yaml")
    parser.add_argument("--candidate", default="examples/svd_regression_fast_degraded.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifacts-dir", default="artifacts/autopilot-96h")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--pid-file", default="")
    parser.add_argument("--log-file", default="session.log")
    parser.add_argument("--status-file", default="autopilot_status.json")
    parser.add_argument("--keep-last-runs", type=int, default=0)
    parser.add_argument("--stop-on-unexpected-pass", action="store_true")
    parser.add_argument(
        "--skip-memory-cleanup",
        action="store_true",
        help="Pass through to autopilot_96h.py to disable gc/cuda cleanup.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Terminate existing autopilot process from pid file and replace it.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pid_path = Path(args.pid_file).resolve() if args.pid_file else artifacts_dir / "autopilot.pid"
    log_path = artifacts_dir / args.log_file
    replaced_pid: int | None = None

    if pid_path.exists():
        existing = read_json_dict(pid_path) or {}
        existing_pid = int(existing.get("pid", 0) or 0)
        if existing_pid > 0 and pid_exists(existing_pid):
            if not args.replace_existing:
                print(f"autopilot already running pid={existing_pid} (use --replace-existing)")
                print(f"pid_file={pid_path}")
                return 1
            if not terminate_pid(existing_pid):
                print(f"failed to terminate existing pid={existing_pid}")
                return 1
            replaced_pid = existing_pid

    cmd = [
        sys.executable,
        "-u",
        "scripts/autopilot_96h.py",
        "--hours",
        str(args.hours),
        "--cooldown-sec",
        str(args.cooldown_sec),
        "--max-cycles",
        str(args.max_cycles),
        "--baseline",
        args.baseline,
        "--candidate",
        args.candidate,
        "--artifacts-dir",
        str(artifacts_dir),
        "--pid-file",
        str(pid_path),
        "--status-file",
        args.status_file,
        "--keep-last-runs",
        str(args.keep_last_runs),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.coordinator_url:
        cmd.extend(["--coordinator-url", args.coordinator_url])
    if args.stop_on_unexpected_pass:
        cmd.append("--stop-on-unexpected-pass")
    if args.skip_memory_cleanup:
        cmd.append("--skip-memory-cleanup")

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    child_env = os.environ.copy()
    child_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    child_env.setdefault("TQDM_DISABLE", "1")

    with log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=creationflags,
            env=child_env,
        )

    payload = {
        "pid": proc.pid,
        "started_at_utc": utc_now_iso(),
        "cmd": cmd,
        "log_path": str(log_path),
        "artifacts_dir": str(artifacts_dir),
        "replaced_pid": replaced_pid,
    }
    pid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"autopilot_launched pid={proc.pid}")
    print(f"pid_file={pid_path}")
    print(f"log_file={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
