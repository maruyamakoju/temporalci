from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit distributed run and wait for completion."
    )
    parser.add_argument("--coordinator-url", default="http://localhost:8080")
    parser.add_argument("--suite", default="examples/regression_core.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--artifacts-dir", default="artifacts/distributed-smoke")
    parser.add_argument("--baseline-mode", default="none")
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    if not suite_path.exists():
        print(f"suite file not found: {suite_path}")
        return 1

    create_payload = {
        "suite_yaml": suite_path.read_text(encoding="utf-8"),
        "suite_root": str(suite_path.parent),
        "model_name": args.model,
        "artifacts_dir": args.artifacts_dir,
        "fail_on_regression": True,
        "baseline_mode": args.baseline_mode,
        "upload_artifacts": bool(args.upload_artifacts),
    }
    created = _post_json(f"{args.coordinator_url.rstrip('/')}/runs", create_payload)
    run_id = str(created.get("run_id", ""))
    if not run_id:
        print(f"unexpected create payload: {created}")
        return 1

    print(f"created run_id={run_id}")
    deadline = time.time() + args.timeout_sec
    while time.time() < deadline:
        run_payload = _get_json(f"{args.coordinator_url.rstrip('/')}/runs/{run_id}")
        status = str(run_payload.get("status", "unknown"))
        result = run_payload.get("payload", {})
        if status in {"completed", "failed"}:
            run_status = result.get("status")
            print(f"coordinator_status={status} run_status={run_status}")
            if status == "failed":
                print(json.dumps(run_payload, indent=2))
                return 1
            if run_status == "PASS":
                return 0
            if run_status == "FAIL":
                return 2
            print(f"unexpected run payload status: {run_status}")
            print(json.dumps(run_payload, indent=2))
            return 1
        time.sleep(args.poll_sec)

    print(f"timeout waiting for run completion: run_id={run_id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
