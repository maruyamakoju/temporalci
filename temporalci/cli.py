from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from temporalci.adapters import available_adapters
from temporalci.constants import BASELINE_MODES
from temporalci.config import SuiteValidationError  # ConfigError alias
from temporalci.config import load_suite
from temporalci.engine import run_suite
from temporalci.metrics import available_metrics
from temporalci.sprt_calibration import sprt_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporalci", description="TemporalCI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run a suite")
    run_cmd.add_argument("suite", help="Path to suite yaml")
    run_cmd.add_argument("--model", help="Model name in suite", default=None)
    run_cmd.add_argument("--artifacts-dir", default="artifacts", help="Artifact root directory")
    run_cmd.add_argument(
        "--ignore-regression",
        action="store_true",
        help="Do not fail CI on regression against previous run",
    )
    run_cmd.add_argument(
        "--baseline-mode",
        choices=sorted(BASELINE_MODES),
        default="latest_pass",
        help="How baseline run is chosen for regression comparison",
    )
    run_cmd.add_argument(
        "--print-json",
        action="store_true",
        help="Print full run payload as JSON",
    )

    validate_cmd = sub.add_parser("validate", help="Validate a suite file")
    validate_cmd.add_argument("suite", help="Path to suite yaml")

    list_cmd = sub.add_parser("list", help="List available adapters and metrics")
    list_cmd.add_argument(
        "--json",
        action="store_true",
        help="Print as JSON",
    )

    sprt_cmd = sub.add_parser("sprt", help="SPRT calibration/apply/check utilities")
    sprt_cmd.add_argument(
        "sprt_args",
        nargs=argparse.REMAINDER,
        help="Pass-through args for 'calibrate|apply|check'",
    )
    return parser


def _print_summary(result: dict[str, Any]) -> None:
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
    if isinstance(regressions, list) and regressions:
        for item in regressions:
            if not isinstance(item, dict):
                continue
            status = "REGRESSED" if item.get("regressed") else "OK"
            print(
                "regression "
                f"{item.get('metric')} baseline={item.get('baseline')} "
                f"current={item.get('current')} delta={item.get('delta')} => {status}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = {
            "adapters": available_adapters(),
            "metrics": available_metrics(),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("adapters:", ", ".join(payload["adapters"]))
            print("metrics:", ", ".join(payload["metrics"]))
        return 0

    if args.command == "validate":
        try:
            suite = load_suite(Path(args.suite))
        except SuiteValidationError as exc:
            print(f"config error: {exc}")
            return 1
        print(
            f"valid suite: project={suite.project} suite={suite.suite_name} "
            f"models={len(suite.models)} tests={len(suite.tests)}"
        )
        return 0

    if args.command == "sprt":
        if not args.sprt_args:
            print("usage: temporalci sprt <calibrate|apply|check> ...")
            return 1
        try:
            return sprt_main(args.sprt_args)
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command != "run":
        parser.error(f"unknown command: {args.command}")

    try:
        suite = load_suite(Path(args.suite))
        result = run_suite(
            suite=suite,
            model_name=args.model,
            artifacts_dir=args.artifacts_dir,
            fail_on_regression=not args.ignore_regression,
            baseline_mode=args.baseline_mode,
        )
    except SuiteValidationError as exc:
        print(f"config error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1

    if args.print_json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)

    return 0 if result.get("status") == "PASS" else 2
