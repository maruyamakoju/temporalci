from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _print_summary(result: dict[str, object]) -> None:
    print(f"status={result['status']} run_id={result['run_id']}")
    print(f"run_dir={result['run_dir']}")
    print(f"model={result['model_name']} samples={result['sample_count']}")
    print(f"gate_failed={result['gate_failed']} regression_failed={result['regression_failed']}")

    gates = result.get("gates", [])
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            status = "PASS" if gate.get("passed") else "FAIL"
            print(
                "gate "
                f"{gate.get('metric')} {gate.get('op')} {gate.get('value')} "
                f"actual={gate.get('actual')} => {status}"
            )

    regressions = result.get("regressions", [])
    if isinstance(regressions, list):
        for item in regressions:
            if not isinstance(item, dict):
                continue
            status = "REGRESSED" if item.get("regressed") else "OK"
            print(
                "regression "
                f"{item.get('metric')} baseline={item.get('baseline')} "
                f"current={item.get('current')} delta={item.get('delta')} => {status}"
            )


def _run_in_process(
    *,
    baseline: str,
    candidate: str,
    artifacts_dir: str,
    model: str | None,
) -> int:
    from temporalci.config import load_suite
    from temporalci.engine import run_suite

    baseline_suite = load_suite(Path(baseline))
    baseline_result = run_suite(
        suite=baseline_suite,
        model_name=model,
        artifacts_dir=artifacts_dir,
        baseline_mode="none",
    )
    _print_summary(baseline_result)
    if baseline_result.get("status") != "PASS":
        print(f"baseline run failed with status={baseline_result.get('status')}")
        return 2

    candidate_suite = load_suite(Path(candidate))
    candidate_result = run_suite(
        suite=candidate_suite,
        model_name=model,
        artifacts_dir=artifacts_dir,
        baseline_mode="latest_pass",
    )
    _print_summary(candidate_result)

    if candidate_result.get("status") == "PASS":
        print("candidate suite passed (no regression gate failure)")
        return 0
    print("candidate suite failed as expected by gate/regression")
    return 2


def _run_via_subprocess(
    *,
    baseline: str,
    candidate: str,
    artifacts_dir: str,
    model: str | None,
    print_json: bool,
) -> int:
    baseline_cmd = [
        sys.executable,
        "-m",
        "temporalci",
        "run",
        baseline,
        "--artifacts-dir",
        artifacts_dir,
        "--baseline-mode",
        "none",
    ]
    if model:
        baseline_cmd.extend(["--model", model])
    if print_json:
        baseline_cmd.append("--print-json")

    baseline_code = _run(baseline_cmd)
    if baseline_code != 0:
        print(f"baseline run failed with exit={baseline_code}")
        return baseline_code

    candidate_cmd = [
        sys.executable,
        "-m",
        "temporalci",
        "run",
        candidate,
        "--artifacts-dir",
        artifacts_dir,
        "--baseline-mode",
        "latest_pass",
    ]
    if model:
        candidate_cmd.extend(["--model", model])
    if print_json:
        candidate_cmd.append("--print-json")

    candidate_code = _run(candidate_cmd)
    if candidate_code == 0:
        print("candidate suite passed (no regression gate failure)")
    elif candidate_code == 2:
        print("candidate suite failed as expected by gate/regression")
    else:
        print(f"candidate suite failed with runtime/config error exit={candidate_code}")
    return candidate_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline then candidate suite to demonstrate TemporalCI regression gate"
    )
    parser.add_argument("--baseline", default="examples/svd_regression_fast.yaml")
    parser.add_argument("--candidate", default="examples/svd_regression_fast_degraded.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts/svd-demo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run via CLI subprocesses instead of in-process engine calls",
    )
    args = parser.parse_args()

    if args.subprocess:
        return _run_via_subprocess(
            baseline=args.baseline,
            candidate=args.candidate,
            artifacts_dir=args.artifacts_dir,
            model=args.model,
            print_json=bool(args.print_json),
        )

    if args.print_json:
        print("--print-json is supported only with --subprocess")
        return 1
    return _run_in_process(
        baseline=args.baseline,
        candidate=args.candidate,
        artifacts_dir=args.artifacts_dir,
        model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
