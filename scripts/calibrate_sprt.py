from __future__ import annotations

import argparse
import json
import math
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
    args = parser.parse_args()

    if args.runs <= 0:
        print("--runs must be > 0")
        return 1

    suite = load_suite(Path(args.suite))
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

    mismatched_runs = [
        row
        for row in run_summaries
        if float(row.get("paired_ratio") or 0.0) < float(sprt_params["min_paired_ratio"])
    ]
    notes: list[str] = []
    if not all_deltas:
        notes.append("No paired deltas collected. Check pairing_mode and sample_id coverage.")
    if sigma_std == 0.0 and sigma_mad == 0.0:
        notes.append(
            "Observed deltas were zero-variance across calibration runs. "
            "Recommended sigma may be optimistic unless suite includes realistic noise."
        )
    if mismatched_runs:
        notes.append(
            f"{len(mismatched_runs)} / {len(run_summaries)} runs were below min_paired_ratio="
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
        "notes": notes,
    }

    atomic_write_json(output_json, summary)
    print(f"wrote_calibration {output_json}")
    print(
        "recommended_params "
        f"sigma={summary['recommended_params']['sigma']} "
        f"min_pairs={summary['recommended_params']['min_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
