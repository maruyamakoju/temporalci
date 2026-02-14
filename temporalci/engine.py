from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.adapters import build_adapter
from temporalci.config import select_model
from temporalci.constants import (
    BASELINE_MODES,
    DIRECTION_HIGHER_IS_BETTER,
    DIRECTION_LOWER_IS_BETTER,
    GATE_OPERATORS,
)
from temporalci.metrics import run_metric
from temporalci.report import write_html_report
from temporalci.types import GateSpec, GeneratedSample, SuiteSpec
from temporalci.utils import as_bool, is_number, normalize_prompt, utc_now

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_DIR_RETRY_LIMIT = 20

# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%S%fZ")


def _create_run_dir(model_root: Path) -> tuple[str, Path]:
    """Allocate a unique timestamp-based run directory under *model_root*."""
    for _ in range(_RUN_DIR_RETRY_LIMIT):
        run_id = _new_run_id()
        run_dir = model_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("failed to allocate unique run directory after retries")


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _resolve_metric_path(payload: dict[str, Any], dotted_path: str) -> Any:
    """Walk *payload* along a dotted key path (e.g. ``'vbench.score'``)."""
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


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
        value = _resolve_metric_path(row, subpath)
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
    expected_pairs = min(len(current_rows), len(baseline_rows))
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


def _read_sprt_params(raw: dict[str, Any]) -> dict[str, Any]:
    alpha = float(raw.get("alpha", 0.05))
    beta = float(raw.get("beta", 0.1))
    effect_size = abs(float(raw.get("effect_size", 0.02)))
    sigma_floor = abs(float(raw.get("sigma_floor", 0.01)))
    sigma_mode = str(raw.get("sigma_mode", "estimate")).strip().lower()
    sigma_value_raw = raw.get("sigma")
    min_pairs = max(2, int(raw.get("min_pairs", 6)))
    min_paired_ratio = float(raw.get("min_paired_ratio", 1.0))
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
        "inconclusive": inconclusive,
        "require_baseline": require_baseline,
        "baseline_missing": baseline_missing,
        "pairing_mode": pairing_mode,
        "allow_index_fallback": allow_index_fallback,
    }


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)
    return math.sqrt(max(0.0, variance))


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

    if len(deltas) < min_pairs:
        sigma_for_payload = None
        if sigma_mode == "fixed" and fixed_sigma is not None:
            sigma_for_payload = round(float(fixed_sigma), 8)
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
        }

    if sigma_mode == "fixed":
        if fixed_sigma is None:
            raise ValueError("sprt sigma is required when sigma_mode=fixed")
        sigma = float(fixed_sigma)
    else:
        sigma = max(_sample_std(deltas), sigma_floor)
    if sigma <= 0.0:
        raise ValueError("sprt sigma must be > 0")

    sigma_sq = sigma * sigma
    mu0 = -effect_size
    mu1 = 0.0
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))

    llr = 0.0
    crossed_at: int | None = None
    decision = "inconclusive"
    for idx, delta in enumerate(deltas, start=1):
        llr += (((delta - mu0) ** 2) - ((delta - mu1) ** 2)) / (2.0 * sigma_sq)
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
        "crossed_at": crossed_at,
    }


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
            actual = _resolve_metric_path(metrics_payload, gate.metric)
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
            paired_ratio = float(pairing_summary.get("paired_ratio", 0.0))
            if paired_ratio < min_paired_ratio:
                result["sprt"] = {
                    "decision": "pairing_insufficient",
                    "decision_passed": False,
                    "reason": "paired_ratio_below_min",
                    "paired_count": int(pairing_summary.get("paired_count", 0)),
                    "expected_pairs": int(pairing_summary.get("expected_pairs", 0)),
                    "paired_ratio": round(paired_ratio, 6),
                    "min_paired_ratio": round(min_paired_ratio, 6),
                    "pairing": pairing_summary,
                }
                result["passed"] = False
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
# Baseline loading & regression detection
# ---------------------------------------------------------------------------


def _load_previous_run(
    model_root: Path,
    current_run_id: str,
    *,
    baseline_mode: str,
) -> dict[str, Any] | None:
    if baseline_mode == "none":
        return None
    if not model_root.exists():
        return None

    candidates: list[tuple[str, dict[str, Any]]] = []
    for child in model_root.iterdir():
        if not child.is_dir() or child.name == current_run_id:
            continue
        run_json = child / "run.json"
        if run_json.exists():
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                candidates.append((child.name, payload))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    if baseline_mode == "latest":
        return candidates[0][1]
    if baseline_mode == "latest_pass":
        for _, payload in candidates:
            if payload.get("status") == "PASS":
                return payload
        return None

    raise ValueError(
        f"unsupported baseline_mode '{baseline_mode}'. "
        f"supported: {sorted(BASELINE_MODES)}"
    )


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
            current_value = _resolve_metric_path(current_metrics, metric)
            baseline_value = _resolve_metric_path(baseline_metrics, metric)
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


# ---------------------------------------------------------------------------
# Artifact retention
# ---------------------------------------------------------------------------


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


def _build_sample_rows_with_retention(
    *,
    samples: list[GeneratedSample],
    status: str,
    artifacts_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    video_policy = str(artifacts_cfg.get("video", "all")).strip().lower()
    max_samples_raw = artifacts_cfg.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None

    if video_policy == "none":
        retain_limit = 0
    elif video_policy == "failures_only" and status == "PASS":
        retain_limit = 0
    else:
        retain_limit = len(samples)

    if max_samples is not None:
        retain_limit = min(retain_limit, max_samples)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        keep = idx < retain_limit
        local_path = Path(sample.video_path)
        local_exists = local_path.exists() and local_path.is_file()
        deleted = False
        if local_exists and not keep:
            deleted = _safe_unlink(local_path)

        rows.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "video_path": sample.video_path if keep else None,
                "artifact_retained": keep,
                "artifact_deleted": deleted,
                "metadata": sample.metadata,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def _build_sample_id(
    *,
    test_id: str,
    prompt: str,
    seed: int,
    video_cfg: dict[str, Any],
) -> str:
    payload = {
        "test_id": test_id,
        "prompt": normalize_prompt(prompt),
        "seed": int(seed),
        "video_cfg": video_cfg,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _generate_samples(
    *,
    suite: SuiteSpec,
    adapter: Any,
    videos_dir: Path,
) -> list[GeneratedSample]:
    """Run the adapter for every (test, prompt, seed) combination."""
    samples: list[GeneratedSample] = []
    for test in suite.tests:
        for prompt in test.prompts:
            for seed in test.seeds:
                effective_video_cfg = dict(test.video)
                if "encode" in suite.artifacts and "encode" not in effective_video_cfg:
                    effective_video_cfg["encode"] = suite.artifacts["encode"]
                sample = adapter.generate(
                    test_id=test.id,
                    prompt=prompt,
                    seed=seed,
                    video_cfg=effective_video_cfg,
                    output_dir=videos_dir,
                )
                sample_id = _build_sample_id(
                    test_id=test.id,
                    prompt=prompt,
                    seed=int(seed),
                    video_cfg=effective_video_cfg,
                )
                metadata = dict(sample.metadata)
                metadata.setdefault("sample_id", sample_id)
                sample.metadata = metadata
                samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Run artifact writing
# ---------------------------------------------------------------------------


def _write_run_artifacts(
    *,
    run_dir: Path,
    model_root: Path,
    run_id: str,
    timestamp: str,
    status: str,
    samples: list[GeneratedSample],
    payload: dict[str, Any],
) -> None:
    """Persist run.json, report.html, latest_run.txt, and runs.jsonl."""
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html_report(run_dir / "report.html", payload)

    latest_file = model_root / "latest_run.txt"
    latest_file.write_text(run_id, encoding="utf-8")

    index_entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "status": status,
        "sample_count": len(samples),
    }
    with (model_root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_entry) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_suite(
    *,
    suite: SuiteSpec,
    model_name: str | None = None,
    artifacts_dir: str | Path = "artifacts",
    fail_on_regression: bool = True,
    baseline_mode: str = "latest_pass",
) -> dict[str, Any]:
    """Execute a full suite run and return the result payload."""
    if baseline_mode not in BASELINE_MODES:
        available = ", ".join(sorted(BASELINE_MODES))
        raise ValueError(f"invalid baseline_mode '{baseline_mode}'. choose: {available}")

    model = select_model(suite, model_name)
    adapter = build_adapter(model)

    timestamp = utc_now().isoformat()
    model_root = Path(artifacts_dir) / suite.project / suite.suite_name / model.name
    run_id, run_dir = _create_run_dir(model_root=model_root)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate samples
    samples = _generate_samples(suite=suite, adapter=adapter, videos_dir=videos_dir)

    # 2. Evaluate metrics
    metrics_payload: dict[str, Any] = {}
    for metric in suite.metrics:
        metric_params = dict(metric.params)
        if "keep_workdir" in suite.artifacts and "keep_workdir" not in metric_params:
            metric_params["keep_workdir"] = suite.artifacts["keep_workdir"]
        metrics_payload[metric.name] = run_metric(
            name=metric.name,
            samples=samples,
            params=metric_params,
        )

    # 3. Load baseline
    baseline = _load_previous_run(
        model_root=model_root, current_run_id=run_id, baseline_mode=baseline_mode
    )
    baseline_metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
    baseline_run_id = baseline.get("run_id") if isinstance(baseline, dict) else None

    # 4. Evaluate gates
    gates = _evaluate_gates(
        suite.gates,
        metrics_payload,
        baseline_metrics=baseline_metrics if isinstance(baseline_metrics, dict) else None,
    )
    gate_failed = any(not gate.get("passed", False) for gate in gates)

    # 5. Regression comparison
    regressions = _compute_regressions(
        gates=gates, current_metrics=metrics_payload, baseline_metrics=baseline_metrics
    )
    regression_failed = any(item["regressed"] for item in regressions)

    # 6. Determine final status
    should_fail = gate_failed or (fail_on_regression and regression_failed)
    status = "FAIL" if should_fail else "PASS"

    # 7. Apply retention policy
    sample_rows = _build_sample_rows_with_retention(
        samples=samples, status=status, artifacts_cfg=suite.artifacts
    )

    # 8. Build result payload
    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "project": suite.project,
        "suite_name": suite.suite_name,
        "model_name": model.name,
        "status": status,
        "sample_count": len(samples),
        "metrics": metrics_payload,
        "gates": gates,
        "gate_failed": gate_failed,
        "regressions": regressions,
        "regression_failed": regression_failed,
        "baseline_run_id": baseline_run_id,
        "baseline_mode": baseline_mode,
        "artifacts_policy": suite.artifacts,
        "samples": sample_rows,
    }

    # 9. Write artifacts
    _write_run_artifacts(
        run_dir=run_dir,
        model_root=model_root,
        run_id=run_id,
        timestamp=timestamp,
        status=status,
        samples=samples,
        payload=payload,
    )

    payload["run_dir"] = str(run_dir)
    return payload
