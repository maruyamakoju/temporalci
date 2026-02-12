from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from temporalci.config import load_suite, select_model
from temporalci.engine import run_suite
from temporalci.utils import safe_write_json, utc_now_iso


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Safe I/O wrappers
# ---------------------------------------------------------------------------


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    if not safe_write_json(path, payload):
        print(f"warning: failed to write status file {path}", flush=True)


def _safe_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: failed to append jsonl {path}: {exc}", flush=True)


def _safe_remove_pid_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        return


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------


def _safe_release_runtime_memory(*, clear_cuda_cache: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"gc_collected": 0, "cuda_cache_cleared": False}
    try:
        payload["gc_collected"] = int(gc.collect())
    except Exception as exc:  # noqa: BLE001
        payload["gc_error"] = str(exc)

    if not clear_cuda_cache:
        return payload

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            payload["cuda_cache_cleared"] = True
    except Exception as exc:  # noqa: BLE001
        payload["cuda_error"] = str(exc)
    return payload


# ---------------------------------------------------------------------------
# Run directory pruning
# ---------------------------------------------------------------------------


def _prune_model_runs(*, model_root: Path, keep_last: int) -> dict[str, Any]:
    if keep_last <= 0:
        return {"enabled": False}
    if not model_root.exists():
        return {
            "enabled": True,
            "model_root": str(model_root),
            "kept": 0,
            "deleted": 0,
            "deleted_run_ids": [],
        }

    run_dirs = sorted(
        [child for child in model_root.iterdir() if child.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    stale = run_dirs[keep_last:]
    deleted_run_ids: list[str] = []
    for run_dir in stale:
        shutil.rmtree(run_dir, ignore_errors=True)
        deleted_run_ids.append(run_dir.name)

    return {
        "enabled": True,
        "model_root": str(model_root),
        "kept": min(keep_last, len(run_dirs)),
        "deleted": len(deleted_run_ids),
        "deleted_run_ids": deleted_run_ids,
    }


# ---------------------------------------------------------------------------
# Gate pair runner
# ---------------------------------------------------------------------------


def _run_gate_pair(
    *,
    baseline_suite_path: Path,
    candidate_suite_path: Path,
    artifacts_dir: Path,
    model_name: str | None,
) -> dict[str, Any]:
    baseline_suite = load_suite(baseline_suite_path)
    baseline_result = run_suite(
        suite=baseline_suite,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
        baseline_mode="none",
    )

    candidate_suite = load_suite(candidate_suite_path)
    candidate_result = run_suite(
        suite=candidate_suite,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
        baseline_mode="latest_pass",
    )

    return {
        "baseline": {
            "run_id": baseline_result["run_id"],
            "status": baseline_result["status"],
            "run_dir": baseline_result["run_dir"],
            "vbench_score": baseline_result.get("metrics", {})
            .get("vbench_temporal", {})
            .get("score"),
        },
        "candidate": {
            "run_id": candidate_result["run_id"],
            "status": candidate_result["status"],
            "run_dir": candidate_result["run_dir"],
            "regression_failed": candidate_result.get("regression_failed"),
            "gate_failed": candidate_result.get("gate_failed"),
            "vbench_score": candidate_result.get("metrics", {})
            .get("vbench_temporal", {})
            .get("score"),
        },
        "exit_code": 0 if candidate_result.get("status") == "PASS" else 2,
    }


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def _write_terminal_status(
    *,
    status_path: Path,
    state: str,
    cycle: int,
    deadline: float,
    stop_reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "finished_at_utc": utc_now_iso(),
        "cycle": cycle,
        "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
    }
    if stop_reason:
        payload["stop_reason"] = stop_reason
    _safe_write_json(status_path, payload)


# ---------------------------------------------------------------------------
# Single cycle execution
# ---------------------------------------------------------------------------


def _run_cycle(
    *,
    cycle: int,
    args: argparse.Namespace,
    baseline_suite_path: Path,
    candidate_suite_path: Path,
    artifacts_dir: Path,
    log_path: Path,
    status_path: Path,
    deadline: float,
    model_root_for_cleanup: Path | None,
) -> str | None:
    """Execute one autopilot cycle. Returns a stop reason, or ``None`` to continue."""
    cycle_payload: dict[str, Any] = {
        "cycle": cycle,
        "started_at_utc": utc_now_iso(),
        "baseline_suite": str(baseline_suite_path),
        "candidate_suite": str(candidate_suite_path),
        "artifacts_dir": str(artifacts_dir),
    }
    _safe_write_json(
        status_path,
        {
            "state": "running",
            "started_at_utc": cycle_payload["started_at_utc"],
            "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
            "cycle": cycle,
            "phase": "cycle_start",
        },
    )
    _safe_append_jsonl(log_path, {"event": "cycle_start", **cycle_payload})

    stop_reason: str | None = None
    try:
        result = _run_gate_pair(
            baseline_suite_path=baseline_suite_path,
            candidate_suite_path=candidate_suite_path,
            artifacts_dir=artifacts_dir,
            model_name=args.model,
        )
        cycle_payload["result"] = result
        cycle_payload["status"] = "ok"
        print(
            f"cycle={cycle} baseline={result['baseline']['status']} "
            f"candidate={result['candidate']['status']} exit={result['exit_code']}",
            flush=True,
        )

        if args.stop_on_unexpected_pass and result["exit_code"] == 0:
            stop_reason = "unexpected_candidate_pass"
            cycle_payload["stop_reason"] = stop_reason
    except Exception as exc:  # noqa: BLE001
        cycle_payload["status"] = "error"
        cycle_payload["error"] = str(exc)
        cycle_payload["traceback"] = traceback.format_exc()
        print(f"cycle={cycle} error={exc}", flush=True)

    # Requeue stale tasks in coordinator if configured
    if args.coordinator_url:
        try:
            payload = _post_json(
                f"{args.coordinator_url.rstrip('/')}/admin/requeue_stale?limit=100",
                {},
            )
            cycle_payload["requeue_stale"] = payload
        except Exception as exc:  # noqa: BLE001
            cycle_payload["requeue_stale_error"] = str(exc)

    # Prune old runs
    if model_root_for_cleanup is not None:
        cycle_payload["cleanup"] = _prune_model_runs(
            model_root=model_root_for_cleanup,
            keep_last=int(args.keep_last_runs),
        )

    # Release memory
    if not args.skip_memory_cleanup:
        cycle_payload["memory_cleanup"] = _safe_release_runtime_memory(clear_cuda_cache=True)

    cycle_payload["finished_at_utc"] = utc_now_iso()
    _safe_append_jsonl(log_path, {"event": "cycle_end", **cycle_payload})
    _safe_write_json(
        status_path,
        {
            "state": "running",
            "finished_at_utc": cycle_payload["finished_at_utc"],
            "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
            "cycle": cycle,
            "phase": "cycle_end",
            "last_status": cycle_payload.get("status"),
            "last_result": cycle_payload.get("result"),
            "last_error": cycle_payload.get("error"),
            "last_cleanup": cycle_payload.get("cleanup"),
            "last_memory_cleanup": cycle_payload.get("memory_cleanup"),
        },
    )
    return stop_reason


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unattended long-run autopilot for TemporalCI regression-gate demos."
    )
    parser.add_argument("--hours", type=float, default=96.0)
    parser.add_argument("--cooldown-sec", type=float, default=300.0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited until hours")
    parser.add_argument("--baseline", default="examples/svd_regression_fast.yaml")
    parser.add_argument("--candidate", default="examples/svd_regression_fast_degraded.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifacts-dir", default="artifacts/autopilot")
    parser.add_argument(
        "--pid-file", default="",
        help="Optional pid metadata file to delete when autopilot exits.",
    )
    parser.add_argument("--log-file", default="autopilot_runs.jsonl")
    parser.add_argument(
        "--status-file", default="autopilot_status.json",
        help="Status heartbeat JSON file written each cycle under artifacts dir.",
    )
    parser.add_argument(
        "--keep-last-runs", type=int, default=0,
        help="If >0, keep only the latest N run directories per model root.",
    )
    parser.add_argument(
        "--stop-on-unexpected-pass", action="store_true",
        help="Stop if candidate unexpectedly passes (exit_code=0).",
    )
    parser.add_argument(
        "--coordinator-url", default="",
        help="If set, call /admin/requeue_stale each cycle for distributed recovery.",
    )
    parser.add_argument(
        "--skip-memory-cleanup", action="store_true",
        help="Disable per-cycle gc/cuda cache cleanup.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    baseline_suite_path = Path(args.baseline).resolve()
    candidate_suite_path = Path(args.candidate).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / args.log_file
    status_path_raw = Path(args.status_file)
    status_path = status_path_raw if status_path_raw.is_absolute() else artifacts_dir / status_path_raw
    pid_path: Path | None = None
    if args.pid_file:
        pid_path_raw = Path(args.pid_file)
        pid_path = pid_path_raw if pid_path_raw.is_absolute() else artifacts_dir / pid_path_raw

    if not baseline_suite_path.exists():
        print(f"baseline suite not found: {baseline_suite_path}")
        _safe_remove_pid_file(pid_path)
        return 1
    if not candidate_suite_path.exists():
        print(f"candidate suite not found: {candidate_suite_path}")
        _safe_remove_pid_file(pid_path)
        return 1

    model_root_for_cleanup: Path | None = None
    if int(args.keep_last_runs) > 0:
        baseline_suite = load_suite(baseline_suite_path)
        selected_model = select_model(baseline_suite, args.model)
        model_root_for_cleanup = (
            artifacts_dir / baseline_suite.project / baseline_suite.suite_name / selected_model.name
        )

    deadline = time.time() + max(0.0, float(args.hours) * 3600.0)
    cycle = 0

    print(
        "autopilot_start "
        f"baseline={baseline_suite_path} candidate={candidate_suite_path} "
        f"deadline_utc={datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat()}",
        flush=True,
    )
    _safe_write_json(
        status_path,
        {
            "state": "running",
            "started_at_utc": utc_now_iso(),
            "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
            "cycle": cycle,
            "artifacts_dir": str(artifacts_dir),
            "baseline_suite": str(baseline_suite_path),
            "candidate_suite": str(candidate_suite_path),
        },
    )

    while time.time() < deadline:
        if int(args.max_cycles) > 0 and cycle >= int(args.max_cycles):
            break

        cycle += 1
        started = time.time()

        stop_reason = _run_cycle(
            cycle=cycle,
            args=args,
            baseline_suite_path=baseline_suite_path,
            candidate_suite_path=candidate_suite_path,
            artifacts_dir=artifacts_dir,
            log_path=log_path,
            status_path=status_path,
            deadline=deadline,
            model_root_for_cleanup=model_root_for_cleanup,
        )

        if stop_reason is not None:
            _write_terminal_status(
                status_path=status_path, state="finished",
                cycle=cycle, deadline=deadline, stop_reason=stop_reason,
            )
            _safe_remove_pid_file(pid_path)
            print(f"stopping: {stop_reason}", flush=True)
            return 2

        if int(args.max_cycles) > 0 and cycle >= int(args.max_cycles):
            break
        if time.time() >= deadline:
            break

        elapsed = time.time() - started
        sleep_sec = max(0.0, float(args.cooldown_sec) - elapsed)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    print("autopilot_finished", flush=True)
    _write_terminal_status(
        status_path=status_path, state="finished", cycle=cycle, deadline=deadline
    )
    _safe_remove_pid_file(pid_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
