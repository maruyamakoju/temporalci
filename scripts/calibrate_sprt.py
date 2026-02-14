from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.config import load_suite
from temporalci.config import select_model
from temporalci.engine import _paired_deltas_for_gate
from temporalci.engine import _read_sprt_params
from temporalci.engine import run_suite
from temporalci.types import GateSpec
from temporalci.types import SuiteSpec
from temporalci.utils import atomic_write_json
from temporalci.utils import utc_now_iso


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    if left == right:
        return ordered[left]
    frac = pos - left
    return ordered[left] + (ordered[right] - ordered[left]) * frac


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _mad_sigma(values: list[float]) -> float | None:
    if not values:
        return None
    center = _quantile(values, 0.5)
    if center is None:
        return None
    deviations = [abs(value - center) for value in values]
    mad = _quantile(deviations, 0.5)
    if mad is None:
        return None
    # Normal consistency constant under Gaussian assumption.
    return 1.4826 * float(mad)


def _estimate_required_pairs(
    *,
    alpha: float,
    beta: float,
    effect_size: float,
    sigma: float,
) -> dict[str, float] | None:
    if not (0.0 < alpha < 0.5):
        return None
    if not (0.0 < beta < 0.5):
        return None
    if effect_size <= 0.0 or sigma <= 0.0:
        return None
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    drift_per_pair = (effect_size * effect_size) / (2.0 * sigma * sigma)
    if drift_per_pair <= 0.0:
        return None
    return {
        "upper_threshold": upper,
        "lower_threshold": lower,
        "drift_per_pair": drift_per_pair,
        "required_pairs_upper": upper / drift_per_pair,
        "required_pairs_lower": abs(lower) / drift_per_pair,
    }


def _resolve_sprt_gate(suite: SuiteSpec, gate_metric: str | None) -> GateSpec:
    sprt_gates = [gate for gate in suite.gates if gate.method == "sprt_regression"]
    if not sprt_gates:
        raise ValueError("suite does not define any method=sprt_regression gate")
    if gate_metric is None:
        if len(sprt_gates) == 1:
            return sprt_gates[0]
        metrics = ", ".join(gate.metric for gate in sprt_gates)
        raise ValueError(f"multiple sprt gates found. use --gate-metric. available: {metrics}")
    for gate in sprt_gates:
        if gate.metric == gate_metric:
            return gate
    metrics = ", ".join(gate.metric for gate in sprt_gates)
    raise ValueError(f"gate metric '{gate_metric}' not found in sprt gates: {metrics}")


def _resolve_run_json_path(
    *,
    artifacts_dir: Path,
    suite: SuiteSpec,
    model_name: str,
    run_id: str,
) -> Path:
    return artifacts_dir / suite.project / suite.suite_name / model_name / run_id / "run.json"


def _load_run_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"run.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run payload must be object: {path}")
    return payload


def _delta_summary(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "mad_sigma": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(deltas),
        "mean": mean(deltas),
        "std": _sample_std(deltas),
        "mad_sigma": _mad_sigma(deltas),
        "min": min(deltas),
        "p50": _quantile(deltas, 0.5),
        "p90": _quantile(deltas, 0.9),
        "p95": _quantile(deltas, 0.95),
        "max": max(deltas),
    }


def _load_suite_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"suite YAML root must be a mapping: {path}")
    return payload


def _write_suite_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(rendered, encoding="utf-8")


def _apply_recommended_params_to_suite(
    *,
    suite_path: Path,
    gate_metric: str,
    recommended_params: dict[str, Any],
    apply_out: Path | None,
    apply_inplace: bool,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    payload = _load_suite_yaml(suite_path)
    gates = payload.get("gates")
    if not isinstance(gates, list):
        raise ValueError("suite YAML does not contain a valid top-level 'gates' list")

    target_gate: dict[str, Any] | None = None
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if str(gate.get("method", "threshold")).strip().lower() != "sprt_regression":
            continue
        if str(gate.get("metric", "")) == gate_metric:
            target_gate = gate
            break
    if target_gate is None:
        raise ValueError(f"target gate not found for metric: {gate_metric}")

    params = target_gate.get("params")
    if not isinstance(params, dict):
        params = {}
        target_gate["params"] = params

    before = {
        "sigma_mode": params.get("sigma_mode"),
        "sigma": params.get("sigma"),
        "min_pairs": params.get("min_pairs"),
    }
    params["sigma_mode"] = "fixed"
    params["sigma"] = float(recommended_params["sigma"])
    if recommended_params.get("min_pairs") is not None:
        params["min_pairs"] = int(recommended_params["min_pairs"])

    after = {
        "sigma_mode": params.get("sigma_mode"),
        "sigma": params.get("sigma"),
        "min_pairs": params.get("min_pairs"),
    }

    if apply_inplace:
        backup_path = suite_path.with_suffix(suite_path.suffix + ".bak")
        shutil.copyfile(suite_path, backup_path)
        output_path = suite_path
    else:
        if apply_out is None:
            raise ValueError("apply_out must be set when apply_inplace is false")
        output_path = apply_out

    _write_suite_yaml(output_path, payload)
    return output_path, {"before": before, "after": after}


def _count_mismatch_runs(
    *,
    run_summaries: list[dict[str, Any]],
    min_paired_ratio: float,
) -> int:
    return sum(
        1
        for row in run_summaries
        if float(row.get("paired_ratio") or 0.0) < float(min_paired_ratio)
    )


def _evaluate_checks(
    *,
    summary: dict[str, Any],
    fail_if_no_deltas: bool,
    min_total_deltas: int | None,
    max_mismatch_runs: int | None,
    max_recommended_sigma: float | None,
    min_recommended_sigma: float | None,
) -> list[str]:
    failures: list[str] = []
    delta_count = int(summary.get("delta_summary", {}).get("count") or 0)
    recommended_sigma = float(summary.get("recommended_params", {}).get("sigma") or 0.0)
    run_summaries = summary.get("run_summaries", [])
    run_rows = run_summaries if isinstance(run_summaries, list) else []
    min_paired_ratio = float(summary.get("sprt_params", {}).get("min_paired_ratio") or 1.0)
    mismatch_runs = _count_mismatch_runs(run_summaries=run_rows, min_paired_ratio=min_paired_ratio)

    if fail_if_no_deltas and delta_count == 0:
        failures.append("delta_count is 0 but --fail-if-no-deltas is set")
    if min_total_deltas is not None and delta_count < int(min_total_deltas):
        failures.append(
            f"delta_count={delta_count} is below --min-total-deltas={int(min_total_deltas)}"
        )
    if max_mismatch_runs is not None and mismatch_runs > int(max_mismatch_runs):
        failures.append(
            f"mismatch_runs={mismatch_runs} exceeds --max-mismatch-runs={int(max_mismatch_runs)}"
        )
    if max_recommended_sigma is not None and recommended_sigma > float(max_recommended_sigma):
        failures.append(
            f"recommended_sigma={recommended_sigma} exceeds "
            f"--max-recommended-sigma={float(max_recommended_sigma)}"
        )
    if min_recommended_sigma is not None and recommended_sigma < float(min_recommended_sigma):
        failures.append(
            f"recommended_sigma={recommended_sigma} is below "
            f"--min-recommended-sigma={float(min_recommended_sigma)}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate SPRT fixed-sigma settings from repeated no-change runs."
    )
    parser.add_argument("--suite", required=True, help="Path to suite YAML")
    parser.add_argument(
        "--gate-metric",
        default=None,
        help="SPRT gate metric path (required only when multiple SPRT gates exist)",
    )
    parser.add_argument("--model", default=None, help="Model name from suite (optional)")
    parser.add_argument("--runs", type=int, default=8, help="Number of candidate repeats")
    parser.add_argument("--artifacts-dir", default="artifacts/sprt-calibration")
    parser.add_argument(
        "--baseline-run-id",
        default=None,
        help="Existing baseline run_id (if omitted, baseline run is created first)",
    )
    parser.add_argument("--output-json", default="sprt_calibration.json")
    parser.add_argument(
        "--apply-out",
        default=None,
        help="Write calibrated suite to a new YAML path (safe default)",
    )
    parser.add_argument(
        "--apply-inplace",
        action="store_true",
        help="Apply calibrated params to --suite in place (creates .bak)",
    )
    parser.add_argument("--check", action="store_true", help="Enable calibration checks")
    parser.add_argument(
        "--fail-if-no-deltas",
        action="store_true",
        help="Fail checks when no paired deltas are collected",
    )
    parser.add_argument(
        "--min-total-deltas",
        type=int,
        default=None,
        help="Fail checks when total paired deltas are below this threshold",
    )
    parser.add_argument(
        "--max-mismatch-runs",
        type=int,
        default=None,
        help="Fail checks when mismatch run count exceeds this threshold",
    )
    parser.add_argument(
        "--max-recommended-sigma",
        type=float,
        default=None,
        help="Fail checks when recommended sigma exceeds this value",
    )
    parser.add_argument(
        "--min-recommended-sigma",
        type=float,
        default=None,
        help="Fail checks when recommended sigma is below this value",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        print("--runs must be > 0")
        return 1
    if args.min_total_deltas is not None and args.min_total_deltas < 0:
        print("--min-total-deltas must be >= 0")
        return 1
    if args.max_mismatch_runs is not None and args.max_mismatch_runs < 0:
        print("--max-mismatch-runs must be >= 0")
        return 1
    if args.max_recommended_sigma is not None and args.max_recommended_sigma <= 0:
        print("--max-recommended-sigma must be > 0")
        return 1
    if args.min_recommended_sigma is not None and args.min_recommended_sigma <= 0:
        print("--min-recommended-sigma must be > 0")
        return 1
    if args.apply_inplace and args.apply_out:
        print("--apply-inplace and --apply-out are mutually exclusive")
        return 1

    suite_path = Path(args.suite).resolve()
    suite = load_suite(suite_path)
    model = select_model(suite, args.model)
    sprt_gate = _resolve_sprt_gate(suite, args.gate_metric)
    sprt_params = _read_sprt_params(sprt_gate.params)

    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = artifacts_dir / output_json

    baseline_payload: dict[str, Any]
    if args.baseline_run_id:
        baseline_path = _resolve_run_json_path(
            artifacts_dir=artifacts_dir,
            suite=suite,
            model_name=model.name,
            run_id=str(args.baseline_run_id),
        )
        baseline_payload = _load_run_payload(baseline_path)
    else:
        baseline_payload = run_suite(
            suite=suite,
            model_name=model.name,
            artifacts_dir=artifacts_dir,
            baseline_mode="none",
            fail_on_regression=False,
        )

    baseline_metrics = baseline_payload.get("metrics")
    if not isinstance(baseline_metrics, dict):
        print("baseline run payload does not include metrics")
        return 1

    pairing_mode = str(sprt_params["pairing_mode"])
    all_deltas: list[float] = []
    run_summaries: list[dict[str, Any]] = []

    for _ in range(args.runs):
        candidate_payload = run_suite(
            suite=suite,
            model_name=model.name,
            artifacts_dir=artifacts_dir,
            baseline_mode="none",
            fail_on_regression=False,
        )
        candidate_metrics = candidate_payload.get("metrics")
        if not isinstance(candidate_metrics, dict):
            continue

        deltas, pairing = _paired_deltas_for_gate(
            metric_path=sprt_gate.metric,
            op=sprt_gate.op,
            current_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            require_sample_id=pairing_mode == "sample_id",
            allow_legacy_pairing=(
                pairing_mode == "legacy" or bool(sprt_params["allow_index_fallback"])
            ),
        )
        all_deltas.extend(deltas)
        run_summaries.append(
            {
                "run_id": candidate_payload.get("run_id"),
                "status": candidate_payload.get("status"),
                "paired_count": int(pairing.get("paired_count", 0)),
                "expected_pairs": int(pairing.get("expected_pairs", 0)),
                "paired_ratio": float(pairing.get("paired_ratio", 0.0)),
                "delta_count": len(deltas),
                "delta_mean": mean(deltas) if deltas else None,
                "delta_std": _sample_std(deltas) if deltas else None,
            }
        )

    summary_stats = _delta_summary(all_deltas)
    sigma_floor = float(sprt_params["sigma_floor"])
    sigma_std = float(summary_stats["std"] or 0.0)
    sigma_mad = float(summary_stats["mad_sigma"] or 0.0)
    recommended_sigma = max(sigma_floor, sigma_std, sigma_mad)

    required_pairs = _estimate_required_pairs(
        alpha=float(sprt_params["alpha"]),
        beta=float(sprt_params["beta"]),
        effect_size=float(sprt_params["effect_size"]),
        sigma=recommended_sigma,
    )

    required_min_pairs: int | None = None
    if required_pairs is not None:
        required_min_pairs = max(
            int(sprt_params["min_pairs"]),
            2,
            int(math.ceil(required_pairs["required_pairs_upper"])),
        )

    mismatch_runs = _count_mismatch_runs(
        run_summaries=run_summaries,
        min_paired_ratio=float(sprt_params["min_paired_ratio"]),
    )
    notes: list[str] = []
    if not all_deltas:
        notes.append("No paired deltas collected. Check pairing_mode and sample_id coverage.")
    if sigma_std == 0.0 and sigma_mad == 0.0:
        notes.append(
            "Observed deltas were zero-variance across calibration runs. "
            "Recommended sigma may be optimistic unless suite includes realistic noise."
        )
    if mismatch_runs:
        notes.append(
            f"{mismatch_runs} / {len(run_summaries)} runs were below min_paired_ratio="
            f"{sprt_params['min_paired_ratio']}."
        )
    if required_min_pairs is not None:
        median_expected = _quantile(
            [float(row.get("expected_pairs") or 0.0) for row in run_summaries], 0.5
        )
        if median_expected is not None and required_min_pairs > int(median_expected):
            notes.append(
                "Estimated required min_pairs exceeds median expected pairs per run; "
                "increase prompts/seeds or adjust effect_size."
            )

    summary: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "suite_path": str(Path(args.suite).resolve()),
        "gate_metric": sprt_gate.metric,
        "model_name": model.name,
        "artifacts_dir": str(artifacts_dir),
        "baseline_run_id": baseline_payload.get("run_id"),
        "runs_requested": int(args.runs),
        "runs_completed": len(run_summaries),
        "sprt_params": {
            "alpha": sprt_params["alpha"],
            "beta": sprt_params["beta"],
            "effect_size": sprt_params["effect_size"],
            "sigma_floor": sprt_params["sigma_floor"],
            "min_pairs": sprt_params["min_pairs"],
            "min_paired_ratio": sprt_params["min_paired_ratio"],
            "pairing_mode": sprt_params["pairing_mode"],
            "pairing_mismatch": sprt_params["pairing_mismatch"],
        },
        "delta_summary": summary_stats,
        "sigma_candidates": {
            "sigma_hat_std": sigma_std,
            "sigma_hat_mad": sigma_mad if summary_stats["mad_sigma"] is not None else None,
            "sigma_floor": sigma_floor,
            "recommended_sigma": recommended_sigma,
        },
        "required_pairs": required_pairs,
        "recommended_params": {
            "sigma_mode": "fixed",
            "sigma": round(recommended_sigma, 6),
            "min_pairs": required_min_pairs,
            "min_paired_ratio": sprt_params["min_paired_ratio"],
            "pairing_mismatch": sprt_params["pairing_mismatch"],
            "effect_size": sprt_params["effect_size"],
            "alpha": sprt_params["alpha"],
            "beta": sprt_params["beta"],
        },
        "run_summaries": run_summaries,
        "mismatch_runs": mismatch_runs,
        "notes": notes,
    }

    check_enabled = bool(
        args.check
        or args.fail_if_no_deltas
        or args.min_total_deltas is not None
        or args.max_mismatch_runs is not None
        or args.max_recommended_sigma is not None
        or args.min_recommended_sigma is not None
    )
    check_failures: list[str] = []
    if check_enabled:
        check_failures = _evaluate_checks(
            summary=summary,
            fail_if_no_deltas=bool(args.fail_if_no_deltas),
            min_total_deltas=args.min_total_deltas,
            max_mismatch_runs=args.max_mismatch_runs,
            max_recommended_sigma=args.max_recommended_sigma,
            min_recommended_sigma=args.min_recommended_sigma,
        )
        summary["check"] = {
            "enabled": True,
            "passed": len(check_failures) == 0,
            "failures": check_failures,
            "thresholds": {
                "fail_if_no_deltas": bool(args.fail_if_no_deltas),
                "min_total_deltas": args.min_total_deltas,
                "max_mismatch_runs": args.max_mismatch_runs,
                "max_recommended_sigma": args.max_recommended_sigma,
                "min_recommended_sigma": args.min_recommended_sigma,
            },
        }

    apply_requested = bool(args.apply_inplace or args.apply_out)
    apply_result: dict[str, Any] | None = None
    if apply_requested and not check_failures:
        apply_out = Path(args.apply_out).resolve() if args.apply_out else None
        applied_path, apply_diff = _apply_recommended_params_to_suite(
            suite_path=suite_path,
            gate_metric=sprt_gate.metric,
            recommended_params=summary["recommended_params"],
            apply_out=apply_out,
            apply_inplace=bool(args.apply_inplace),
        )
        apply_result = {
            "applied": True,
            "mode": "inplace" if args.apply_inplace else "output",
            "path": str(applied_path),
            "changes": apply_diff,
        }
        summary["apply"] = apply_result
        print(
            "applied_calibration "
            f"path={applied_path} "
            f"before={apply_diff['before']} "
            f"after={apply_diff['after']}"
        )
    elif apply_requested:
        summary["apply"] = {
            "applied": False,
            "reason": "check_failed",
        }

    atomic_write_json(output_json, summary)
    print(f"wrote_calibration {output_json}")
    print(
        "recommended_params "
        f"sigma={summary['recommended_params']['sigma']} "
        f"min_pairs={summary['recommended_params']['min_pairs']}"
    )
    if check_failures:
        for failure in check_failures:
            print(f"check_failed: {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
