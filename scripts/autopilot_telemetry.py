from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from temporalci.autopilot_utils import pid_exists
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import utc_now_iso

TERMINAL_STATES = {"finished", "stopped", "stale_stopped", "failed"}


def _safe_append_jsonl(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False


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


def _hash_last_runs_line(path: Path) -> str | None:
    line = _tail_line(path)
    if line is None:
        return None
    return hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]


def _parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _compute_cycle_time_sec_from_runs_tail(path: Path) -> float | None:
    line = _tail_line(path)
    if line is None:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("event", "")) != "cycle_end":
        return None
    started = _parse_iso(payload.get("started_at_utc"))
    finished = _parse_iso(payload.get("finished_at_utc"))
    if started is None or finished is None:
        return None
    delta = (finished - started).total_seconds()
    if delta < 0:
        return None
    return round(float(delta), 3)


def _parse_numeric_token(raw: str) -> float | None:
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    for token in ("mib", "mb", "%"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace(" ", "")
    if cleaned in {"n/a", "[notsupported]", "[not supported]", "notsupported", "nan"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_gpu_apps_csv(*, raw: str, pid: int) -> float | None:
    if pid <= 0:
        return None
    total = 0.0
    matched = False
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
        except ValueError:
            continue
        if row_pid != pid:
            continue
        value = _parse_numeric_token(parts[1])
        if value is None:
            continue
        total += value
        matched = True
    if not matched:
        return 0.0
    return round(total, 3)


def _parse_gpu_util_csv(*, raw: str) -> float | None:
    values: list[float] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        first_column = text.split(",")[0].strip()
        value = _parse_numeric_token(first_column)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _query_rss_bytes(pid: int) -> tuple[int | None, str | None]:
    if pid <= 0:
        return None, "pid_missing"
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return None, "psutil_not_available"
    try:
        process = psutil.Process(pid)
        return int(process.memory_info().rss), None
    except Exception as exc:  # noqa: BLE001
        return None, f"psutil_error:{exc.__class__.__name__}"


def _query_gpu_metrics(*, pid: int) -> tuple[float | None, float | None, list[str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None, ["nvidia_smi_not_found"]

    errors: list[str] = []
    gpu_mem_mb: float | None = None
    gpu_util: float | None = None

    apps_command = [
        nvidia_smi,
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        apps_result = subprocess.run(  # noqa: S603
            apps_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if apps_result.returncode == 0:
            gpu_mem_mb = _parse_gpu_apps_csv(raw=apps_result.stdout, pid=pid)
        else:
            errors.append(f"nvidia_smi_apps_rc:{apps_result.returncode}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"nvidia_smi_apps_error:{exc.__class__.__name__}")

    util_command = [
        nvidia_smi,
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        util_result = subprocess.run(  # noqa: S603
            util_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if util_result.returncode == 0:
            gpu_util = _parse_gpu_util_csv(raw=util_result.stdout)
        else:
            errors.append(f"nvidia_smi_util_rc:{util_result.returncode}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"nvidia_smi_util_error:{exc.__class__.__name__}")

    return gpu_mem_mb, gpu_util, errors


def _resolve_file(path: Path, *, artifacts_dir: Path) -> Path:
    return path if path.is_absolute() else artifacts_dir / path


def _collect_sample(
    *,
    pid_path: Path,
    status_path: Path,
    runs_path: Path,
) -> dict[str, Any]:
    pid_payload = read_json_dict(pid_path) or {}
    status_payload = read_json_dict(status_path) or {}

    pid = int(pid_payload.get("pid", 0) or 0)
    pid_alive = pid_exists(pid) if pid > 0 else False
    cycle = int(status_payload.get("cycle", 0) or 0)
    state = str(status_payload.get("state", "unknown"))
    phase = str(status_payload.get("phase", ""))

    rss_bytes, rss_error = _query_rss_bytes(pid)
    gpu_mem_mb, gpu_util, gpu_errors = _query_gpu_metrics(pid=pid)

    sample: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "pid": pid,
        "pid_alive": pid_alive,
        "cycle": cycle,
        "state": state,
        "phase": phase,
        "rss_bytes": rss_bytes,
        "gpu_mem_mb": gpu_mem_mb,
        "gpu_util": gpu_util,
        "last_runs_tail_hash": _hash_last_runs_line(runs_path),
        "cycle_time_sec": _compute_cycle_time_sec_from_runs_tail(runs_path),
    }
    if rss_error:
        sample["rss_error"] = rss_error
    if gpu_errors:
        sample["gpu_errors"] = gpu_errors
    return sample


def _compute_stop_reason(
    *,
    sample: dict[str, Any],
    sample_count: int,
    max_samples: int,
    stop_on_pid_dead: bool,
    stop_on_terminal_state: bool,
) -> str | None:
    if max_samples > 0 and sample_count >= max_samples:
        return "max_samples_reached"
    if stop_on_pid_dead:
        if int(sample.get("pid", 0) or 0) <= 0:
            return "pid_missing"
        if not bool(sample.get("pid_alive", False)):
            return "pid_dead"
    if stop_on_terminal_state:
        state = str(sample.get("state", "unknown"))
        if state in TERMINAL_STATES:
            return f"terminal_state:{state}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sidecar telemetry logger for TemporalCI autopilot.")
    parser.add_argument("--artifacts-dir", default="artifacts/autopilot-96h")
    parser.add_argument("--pid-file", default="autopilot.pid")
    parser.add_argument("--status-file", default="autopilot_status.json")
    parser.add_argument("--runs-file", default="autopilot_runs.jsonl")
    parser.add_argument("--output-file", default="autopilot_telemetry.jsonl")
    parser.add_argument("--interval-sec", type=float, default=60.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means run forever.")
    parser.add_argument(
        "--stop-on-pid-dead",
        action="store_true",
        default=True,
        help="Stop when autopilot pid disappears.",
    )
    parser.add_argument(
        "--no-stop-on-pid-dead",
        dest="stop_on_pid_dead",
        action="store_false",
    )
    parser.add_argument(
        "--stop-on-terminal-state",
        action="store_true",
        default=True,
        help="Stop when status state is terminal.",
    )
    parser.add_argument(
        "--no-stop-on-terminal-state",
        dest="stop_on_terminal_state",
        action="store_false",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    pid_path = _resolve_file(Path(args.pid_file), artifacts_dir=artifacts_dir)
    status_path = _resolve_file(Path(args.status_file), artifacts_dir=artifacts_dir)
    runs_path = _resolve_file(Path(args.runs_file), artifacts_dir=artifacts_dir)
    output_path = _resolve_file(Path(args.output_file), artifacts_dir=artifacts_dir)

    sample_count = 0
    try:
        while True:
            sample_count += 1
            sample = _collect_sample(
                pid_path=pid_path,
                status_path=status_path,
                runs_path=runs_path,
            )
            stop_reason = _compute_stop_reason(
                sample=sample,
                sample_count=sample_count,
                max_samples=int(args.max_samples),
                stop_on_pid_dead=bool(args.stop_on_pid_dead),
                stop_on_terminal_state=bool(args.stop_on_terminal_state),
            )
            if stop_reason:
                sample["telemetry_stop_reason"] = stop_reason

            if not _safe_append_jsonl(output_path, sample):
                print(f"warning: failed to append telemetry line to {output_path}", flush=True)

            print(
                "telemetry_sample "
                f"count={sample_count} pid={sample['pid']} alive={sample['pid_alive']} "
                f"state={sample['state']} cycle={sample['cycle']} "
                f"rss={sample['rss_bytes']} gpu_mem={sample['gpu_mem_mb']} gpu_util={sample['gpu_util']}",
                flush=True,
            )
            if stop_reason:
                print(f"telemetry_stop reason={stop_reason}", flush=True)
                return 0

            time.sleep(max(0.1, float(args.interval_sec)))
    except KeyboardInterrupt:
        payload = {
            "timestamp_utc": utc_now_iso(),
            "telemetry_stop_reason": "keyboard_interrupt",
            "sample_count": sample_count,
        }
        _safe_append_jsonl(output_path, payload)
        print("telemetry_stop reason=keyboard_interrupt", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
