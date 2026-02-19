"""Pure gate-evaluation logic extracted from ``engine.py``.

This module contains all functions concerned with evaluating quality gates,
running SPRT regression tests, and detecting metric regressions.  It has no
dependency on the adapter layer or I/O orchestration in ``engine.py``.

Public names (used by ``engine.py`` and ``sprt_calibration.py``):
    _resolve_metric_path, _compare, _split_metric_path,
    _resolve_sample_metric_value, _build_legacy_series_key,
    _extract_metric_series, _paired_deltas_for_gate, _read_sprt_params,
    _sample_std, _run_sprt, _evaluate_gates, _load_recent_runs,
    _apply_windowed_gates, _compute_regressions
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.constants import (
    DIRECTION_HIGHER_IS_BETTER,
    DIRECTION_LOWER_IS_BETTER,
    GATE_OPERATORS,
)
from temporalci.sprt import derive_sprt_metrics, sprt_thresholds
from temporalci.types import GateSpec
from temporalci.utils import as_bool, is_number, resolve_dotted_path, sample_std

# ---------------------------------------------------------------------------
# Aliases that keep the private names stable for callers that imported from
# engine.py (including tests and sprt_calibration.py).
# ---------------------------------------------------------------------------

_resolve_metric_path = resolve_dotted_path
_sample_std = sample_std


# ---------------------------------------------------------------------------
# Low-level comparison
# ---------------------------------------------------------------------------


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return bool(actual == expected)
    if op == "!=":
        return bool(actual != expected)
    if op == ">=":
        return float(actual) >= float(expected)
    if op == "<=":
        return float(actual) <= float(expected)
    if op == ">":
        return float(actual) > float(expected)
    if op == "<":
        return float(actual) < float(expected)
    raise ValueError(f"unsupported operator: {op}")


# ---------------------------------------------------------------------------
# Metric-path helpers
# ---------------------------------------------------------------------------


def _split_metric_path(dotted_path: str) -> tuple[str, str]:
    if "." not in dotted_path:
        return dotted_path.strip(), ""
    head, tail = dotted_path.split(".", 1)
    return head.strip(), tail.strip()


def _resolve_sample_metric_value(row: dict[str, Any], subpath: str) -> float | None:
    if not subpath:
        return None
    if subpath == "score":
        score = row.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        dims = row.get("dims")
        if isinstance(dims, dict):
            values: list[float] = []
            for value in dims.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            if values:
                return float(mean(values))
        return None
    try:
        value = resolve_dotted_path(row, subpath)
    except KeyError:
        return None
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return None
    return float(value)


def _build_legacy_series_key(row: dict[str, Any], *, fallback_index: int) -> str:
    key_parts: list[str] = []
    for field in ("test_id", "seed", "prompt"):
        field_value = row.get(field)
        if field_value is None:
            continue
        key_parts.append(str(field_value))
    if key_parts:
        return "|".join(key_parts)
    return f"idx:{fallback_index}"


def _extract_metric_series(
    metrics_payload: dict[str, Any],
    metric_path: str,
    *,
    require_sample_id: bool,
    allow_legacy_pairing: bool,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    metric_name, subpath = _split_metric_path(metric_path)
    metric_payload = metrics_payload.get(metric_name)
    if not isinstance(metric_payload, dict):
        return [], {
            "total_rows": 0,
            "usable_rows": 0,
            "missing_sample_id_count": 0,
        }
    per_sample = metric_payload.get("per_sample")
    if not isinstance(per_sample, list):
        return [], {
            "total_rows": 0,
            "usable_rows": 0,
            "missing_sample_id_count": 0,
        }

    rows: list[tuple[str, float]] = []
    missing_sample_id_count = 0
    for idx, raw_row in enumerate(per_sample):
        if not isinstance(raw_row, dict):
            continue
        value = _resolve_sample_metric_value(raw_row, subpath)
        if value is None:
            continue

        raw_sample_id = raw_row.get("sample_id")
        sample_id = str(raw_sample_id).strip() if raw_sample_id is not None else ""
        if sample_id:
            key = f"sid:{sample_id}"
        else:
            missing_sample_id_count += 1
            if require_sample_id and not allow_legacy_pairing:
                continue
            key = _build_legacy_series_key(raw_row, fallback_index=idx)
        rows.append((key, value))
    return rows, {
        "total_rows": len(per_sample),
        "usable_rows": len(rows),
        "missing_sample_id_count": missing_sample_id_count,
    }


# ---------------------------------------------------------------------------
# Paired-delta extraction (SPRT pairing)
# ---------------------------------------------------------------------------


def _paired_deltas_for_gate(
    *,
    metric_path: str,
    op: str,
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    require_sample_id: bool,
    allow_legacy_pairing: bool,
) -> tuple[list[float], dict[str, Any]]:
    current_rows, current_meta = _extract_metric_series(
        current_metrics,
        metric_path,
        require_sample_id=require_sample_id,
        allow_legacy_pairing=allow_legacy_pairing,
    )
    baseline_rows, baseline_meta = _extract_metric_series(
        baseline_metrics,
        metric_path,
        require_sample_id=require_sample_id,
        allow_legacy_pairing=allow_legacy_pairing,
    )
    expected_pairs = max(len(current_rows), len(baseline_rows))
    base_summary = {
        "current_series_count": len(current_rows),
        "baseline_series_count": len(baseline_rows),
        "expected_pairs": expected_pairs,
        "current_missing_sample_id_count": int(current_meta["missing_sample_id_count"]),
        "baseline_missing_sample_id_count": int(baseline_meta["missing_sample_id_count"]),
        "require_sample_id": require_sample_id,
        "allow_legacy_pairing": allow_legacy_pairing,
    }
    if not current_rows or not baseline_rows:
        return [], {
            "pairing": "unavailable",
            "paired_count": 0,
            "paired_ratio": 0.0,
            **base_summary,
        }

    baseline_by_key = {key: value for key, value in baseline_rows}
    deltas: list[float] = []
    matched_rows: list[dict[str, Any]] = []
    key_paired = 0
    for key, current_value in current_rows:
        baseline_value = baseline_by_key.get(key)
        if baseline_value is None:
            continue
        delta = (
            float(current_value) - float(baseline_value)
            if op in DIRECTION_HIGHER_IS_BETTER
            else float(baseline_value) - float(current_value)
        )
        deltas.append(delta)
        matched_rows.append(
            {
                "pair_key": key,
                "current": round(float(current_value), 6),
                "baseline": round(float(baseline_value), 6),
                "delta": round(float(delta), 6),
            }
        )
        key_paired += 1

    if deltas:
        worst_rows = sorted(matched_rows, key=lambda item: float(item["delta"]))[:5]
        return deltas, {
            "pairing": "key_match",
            "paired_count": key_paired,
            "paired_ratio": round(float(key_paired) / max(1, expected_pairs), 6),
            "worst_deltas": worst_rows,
            **base_summary,
        }

    if not allow_legacy_pairing:
        return [], {
            "pairing": "key_mismatch",
            "paired_count": 0,
            "paired_ratio": 0.0,
            **base_summary,
        }

    # Optional fallback for legacy metrics that do not expose stable sample identifiers.
    paired_count = min(len(current_rows), len(baseline_rows))
    for idx in range(paired_count):
        current_value = current_rows[idx][1]
        baseline_value = baseline_rows[idx][1]
        delta = (
            float(current_value) - float(baseline_value)
            if op in DIRECTION_HIGHER_IS_BETTER
            else float(baseline_value) - float(current_value)
        )
        deltas.append(delta)
        matched_rows.append(
            {
                "pair_key": f"idx:{idx}",
                "current": round(float(current_value), 6),
                "baseline": round(float(baseline_value), 6),
                "delta": round(float(delta), 6),
            }
        )
    worst_rows = sorted(matched_rows, key=lambda item: float(item["delta"]))[:5]
    return deltas, {
        "pairing": "index_fallback",
        "paired_count": paired_count,
        "paired_ratio": round(float(paired_count) / max(1, expected_pairs), 6),
        "worst_deltas": worst_rows,
        **base_summary,
    }


# ---------------------------------------------------------------------------
# SPRT parameter parsing
# ---------------------------------------------------------------------------


def _read_sprt_params(raw: dict[str, Any]) -> dict[str, Any]:
    alpha = float(raw.get("alpha", 0.05))
    beta = float(raw.get("beta", 0.1))
    effect_size = abs(float(raw.get("effect_size", 0.02)))
    sigma_floor = abs(float(raw.get("sigma_floor", 0.01)))
    sigma_mode = str(raw.get("sigma_mode", "estimate")).strip().lower()
    sigma_value_raw = raw.get("sigma")
    min_pairs = max(2, int(raw.get("min_pairs", 6)))
    min_paired_ratio = float(raw.get("min_paired_ratio", 1.0))
    pairing_mismatch = str(raw.get("pairing_mismatch", "fail")).strip().lower()
    inconclusive = str(raw.get("inconclusive", "fail")).strip().lower()
    require_baseline = as_bool(raw.get("require_baseline", True), default=True)
    baseline_missing = str(raw.get("baseline_missing", "fail")).strip().lower()
    pairing_mode = str(raw.get("pairing_mode", "sample_id")).strip().lower()
    allow_index_fallback = as_bool(raw.get("allow_index_fallback", False), default=False)

    if not (0.0 < alpha < 0.5):
        raise ValueError("sprt alpha must be in (0, 0.5)")
    if not (0.0 < beta < 0.5):
        raise ValueError("sprt beta must be in (0, 0.5)")
    if effect_size <= 0.0:
        raise ValueError("sprt effect_size must be > 0")
    if sigma_floor <= 0.0:
        raise ValueError("sprt sigma_floor must be > 0")
    if sigma_mode not in {"estimate", "fixed"}:
        raise ValueError("sprt sigma_mode must be one of: estimate, fixed")
    if not (0.0 < min_paired_ratio <= 1.0):
        raise ValueError("sprt min_paired_ratio must be in (0, 1]")
    if pairing_mismatch not in {"fail", "pass", "skip"}:
        raise ValueError("sprt pairing_mismatch must be one of: fail, pass, skip")
    if inconclusive not in {"fail", "pass"}:
        raise ValueError("sprt inconclusive must be one of: fail, pass")
    if baseline_missing not in {"fail", "pass", "skip"}:
        raise ValueError("sprt baseline_missing must be one of: fail, pass, skip")
    if pairing_mode not in {"sample_id", "legacy"}:
        raise ValueError("sprt pairing_mode must be one of: sample_id, legacy")

    sigma: float | None = None
    if sigma_value_raw is not None:
        sigma = abs(float(sigma_value_raw))
        if sigma <= 0.0:
            raise ValueError("sprt sigma must be > 0 when provided")
    if sigma_mode == "fixed" and sigma is None:
        raise ValueError("sprt sigma must be provided when sigma_mode=fixed")

    return {
        "alpha": alpha,
        "beta": beta,
        "effect_size": effect_size,
        "sigma_floor": sigma_floor,
        "sigma_mode": sigma_mode,
        "sigma": sigma,
        "min_pairs": min_pairs,
        "min_paired_ratio": min_paired_ratio,
        "pairing_mismatch": pairing_mismatch,
        "inconclusive": inconclusive,
        "require_baseline": require_baseline,
        "baseline_missing": baseline_missing,
        "pairing_mode": pairing_mode,
        "allow_index_fallback": allow_index_fallback,
    }


# ---------------------------------------------------------------------------
# SPRT sequential test
# ---------------------------------------------------------------------------


def _run_sprt(*, deltas: list[float], params: dict[str, Any]) -> dict[str, Any]:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    effect_size = float(params["effect_size"])
    sigma_floor = float(params["sigma_floor"])
    sigma_mode = str(params.get("sigma_mode", "estimate")).strip().lower()
    min_pairs = int(params["min_pairs"])
    min_paired_ratio = float(params.get("min_paired_ratio", 1.0))
    inconclusive = str(params["inconclusive"])
    fixed_sigma = params.get("sigma")
    thresholds = sprt_thresholds(alpha=alpha, beta=beta)
    upper: float | None = None
    lower: float | None = None
    if thresholds is not None:
        upper, lower = thresholds

    if len(deltas) < min_pairs:
        sigma_for_payload = None
        if sigma_mode == "fixed" and fixed_sigma is not None:
            sigma_for_payload = round(float(fixed_sigma), 8)
        derived = derive_sprt_metrics(
            effect_size=effect_size,
            sigma=sigma_for_payload,
            llr=0.0,
            paired_count=len(deltas),
            upper_threshold=upper,
            lower_threshold=lower,
            alpha=alpha,
            beta=beta,
        )
        return {
            "decision": "inconclusive",
            "decision_passed": inconclusive == "pass",
            "inconclusive_policy": inconclusive,
            "reason": "insufficient_pairs",
            "paired_count": len(deltas),
            "min_pairs": min_pairs,
            "min_paired_ratio": round(float(min_paired_ratio), 6),
            "alpha": alpha,
            "beta": beta,
            "effect_size": effect_size,
            "sigma_mode": sigma_mode,
            "sigma": sigma_for_payload,
            "llr": 0.0,
            "upper_threshold": round(float(upper), 8) if upper is not None else None,
            "lower_threshold": round(float(lower), 8) if lower is not None else None,
            "drift_per_pair": derived["drift_per_pair"],
            "required_pairs_upper": derived["required_pairs_upper"],
            "required_pairs_lower": derived["required_pairs_lower"],
            "llr_per_pair": derived["llr_per_pair"],
        }

    if sigma_mode == "fixed":
        if fixed_sigma is None:
            raise ValueError("sprt sigma is required when sigma_mode=fixed")
        sigma = float(fixed_sigma)
    else:
        sigma = max(sample_std(deltas), sigma_floor)
    if sigma <= 0.0:
        raise ValueError("sprt sigma must be > 0")

    sigma_sq = sigma * sigma
    mu0 = -effect_size
    mu1 = 0.0
    if upper is None or lower is None:
        raise ValueError("sprt thresholds must be finite")

    llr = 0.0
    llr_history: list[float] = []
    crossed_at: int | None = None
    decision = "inconclusive"
    for idx, delta in enumerate(deltas, start=1):
        llr += (((delta - mu0) ** 2) - ((delta - mu1) ** 2)) / (2.0 * sigma_sq)
        llr_history.append(round(llr, 8))
        if idx < min_pairs:
            continue
        if llr >= upper:
            decision = "accept_h1_no_regression"
            crossed_at = idx
            break
        if llr <= lower:
            decision = "accept_h0_regression"
            crossed_at = idx
            break

    if decision == "accept_h1_no_regression":
        decision_passed = True
    elif decision == "accept_h0_regression":
        decision_passed = False
    else:
        decision_passed = inconclusive == "pass"

    derived = derive_sprt_metrics(
        effect_size=effect_size,
        sigma=sigma,
        llr=llr,
        paired_count=len(deltas),
        upper_threshold=upper,
        lower_threshold=lower,
    )

    return {
        "decision": decision,
        "decision_passed": decision_passed,
        "inconclusive_policy": inconclusive,
        "paired_count": len(deltas),
        "min_pairs": min_pairs,
        "min_paired_ratio": round(float(min_paired_ratio), 6),
        "alpha": alpha,
        "beta": beta,
        "effect_size": effect_size,
        "sigma_mode": sigma_mode,
        "sigma": round(float(sigma), 8),
        "llr": round(float(llr), 8),
        "upper_threshold": round(float(upper), 8),
        "lower_threshold": round(float(lower), 8),
        "drift_per_pair": derived["drift_per_pair"],
        "required_pairs_upper": derived["required_pairs_upper"],
        "required_pairs_lower": derived["required_pairs_lower"],
        "llr_per_pair": derived["llr_per_pair"],
        "crossed_at": crossed_at,
        "llr_history": llr_history,
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_gates(
    gates: list[GateSpec],
    metrics_payload: dict[str, Any],
    *,
    baseline_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for gate in gates:
        result: dict[str, Any] = {
            "metric": gate.metric,
            "op": gate.op,
            "value": gate.value,
            "method": gate.method,
            "actual": None,
            "passed": False,
        }
        if gate.op not in GATE_OPERATORS:
            result["error"] = f"unsupported operator: {gate.op}"
            results.append(result)
            continue

        try:
            actual = resolve_dotted_path(metrics_payload, gate.metric)
            result["actual"] = actual
            threshold_passed = _compare(actual, gate.op, gate.value)
            result["threshold_passed"] = threshold_passed

            if gate.method != "sprt_regression":
                result["passed"] = threshold_passed
                results.append(result)
                continue

            params = _read_sprt_params(gate.params)
            if baseline_metrics is None:
                baseline_missing = str(params["baseline_missing"])
                require_baseline = bool(params["require_baseline"])
                if require_baseline and baseline_missing == "skip":
                    baseline_missing = "fail"
                decision_passed = baseline_missing in {"pass", "skip"}
                result["sprt"] = {
                    "decision": "baseline_missing",
                    "decision_passed": decision_passed,
                    "reason": "baseline_missing",
                    "baseline_missing_policy": baseline_missing,
                    "require_baseline": require_baseline,
                }
                result["passed"] = threshold_passed and decision_passed
                results.append(result)
                continue

            pairing_mode = str(params["pairing_mode"])
            deltas, pairing_summary = _paired_deltas_for_gate(
                metric_path=gate.metric,
                op=gate.op,
                current_metrics=metrics_payload,
                baseline_metrics=baseline_metrics,
                require_sample_id=pairing_mode == "sample_id",
                allow_legacy_pairing=(
                    pairing_mode == "legacy" or bool(params["allow_index_fallback"])
                ),
            )
            min_paired_ratio = float(params["min_paired_ratio"])
            pairing_mismatch = str(params["pairing_mismatch"])
            paired_ratio = float(pairing_summary.get("paired_ratio", 0.0))
            if paired_ratio < min_paired_ratio:
                decision_passed = pairing_mismatch in {"pass", "skip"}
                result["sprt"] = {
                    "decision": "pairing_mismatch",
                    "decision_passed": decision_passed,
                    "reason": "paired_ratio_below_min",
                    "pairing_mismatch_policy": pairing_mismatch,
                    "paired_count": int(pairing_summary.get("paired_count", 0)),
                    "expected_pairs": int(pairing_summary.get("expected_pairs", 0)),
                    "paired_ratio": round(paired_ratio, 6),
                    "min_paired_ratio": round(min_paired_ratio, 6),
                    "pairing": pairing_summary,
                }
                result["passed"] = threshold_passed and decision_passed
                results.append(result)
                continue
            sprt_payload = _run_sprt(deltas=deltas, params=params)
            sprt_payload["pairing"] = pairing_summary
            result["sprt"] = sprt_payload
            result["passed"] = threshold_passed and bool(sprt_payload["decision_passed"])
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            result["passed"] = False
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Windowed gate helpers
# ---------------------------------------------------------------------------


def _load_recent_runs(
    model_root: Path,
    *,
    current_run_id: str,
    n: int,
) -> list[dict[str, Any]]:
    """Return up to *n* most-recent run payloads (excluding current run)."""
    if not model_root.exists() or n <= 0:
        return []
    candidates: list[tuple[str, dict[str, Any]]] = []
    for child in model_root.iterdir():
        if not child.is_dir() or child.name == current_run_id:
            continue
        run_json = child / "run.json"
        if run_json.exists():
            try:
                payload = json.loads(run_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    candidates.append((child.name, payload))
            except Exception:  # noqa: BLE001
                pass
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [payload for _, payload in candidates[:n]]


def _apply_windowed_gates(
    gate_specs: list[Any],
    gate_results: list[dict[str, Any]],
    recent_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Override gate results for windowed gates that haven't hit min_failures yet.

    For a gate with ``window > 0``: count threshold failures in the last
    ``window - 1`` historical runs plus the current run.  If the total is
    below ``min_failures``, the gate is marked as passed (transient issue
    policy).

    Historical lookup uses ``threshold_passed`` (the raw metric comparison)
    rather than ``passed``, so previously windowed-passed runs are correctly
    counted as threshold failures.
    """
    out = list(gate_results)
    for i, (spec, result) in enumerate(zip(gate_specs, gate_results)):
        if spec.window <= 0:
            continue
        if result.get("passed", True):
            continue  # already passing — no override needed

        # Count historical threshold failures for this gate (metric + op match).
        # Use threshold_passed when available (present for all gates evaluated by
        # _evaluate_gates); fall back to passed for legacy payloads.
        hist_failures = 0
        for run in recent_runs:
            for g in run.get("gates") or []:
                if not isinstance(g, dict):
                    continue
                if g.get("metric") == spec.metric and g.get("op") == spec.op:
                    # threshold_passed=False means the raw metric check failed,
                    # even if the gate was subsequently windowed-passed.
                    raw_fail = not g.get("threshold_passed", g.get("passed", True))
                    if raw_fail:
                        hist_failures += 1
                    break

        total_failures = hist_failures + 1  # +1 for current run failure
        if total_failures < spec.min_failures:
            out[i] = {
                **result,
                "passed": True,
                "windowed_pass": True,
                "window_failures": total_failures,
                "window_size": spec.window,
                "min_failures": spec.min_failures,
            }
    return out


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def _compute_regressions(
    gates: list[dict[str, Any]],
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if baseline_metrics is None:
        return []

    regressions: list[dict[str, Any]] = []
    for gate in gates:
        metric = str(gate["metric"])
        op = str(gate["op"])
        if op not in DIRECTION_HIGHER_IS_BETTER and op not in DIRECTION_LOWER_IS_BETTER:
            continue
        try:
            current_value = resolve_dotted_path(current_metrics, metric)
            baseline_value = resolve_dotted_path(baseline_metrics, metric)
        except KeyError:
            continue
        if not is_number(current_value) or not is_number(baseline_value):
            continue

        if op in DIRECTION_HIGHER_IS_BETTER:
            regressed = float(current_value) < float(baseline_value)
            direction = "higher_is_better"
        else:
            regressed = float(current_value) > float(baseline_value)
            direction = "lower_is_better"
        delta = round(float(current_value) - float(baseline_value), 6)
        regressions.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "direction": direction,
                "regressed": regressed,
            }
        )
    return regressions
