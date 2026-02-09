from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from temporalci.adapters import build_adapter
from temporalci.constants import BASELINE_MODES
from temporalci.constants import DIRECTION_HIGHER_IS_BETTER
from temporalci.constants import DIRECTION_LOWER_IS_BETTER
from temporalci.constants import GATE_OPERATORS
from temporalci.config import select_model
from temporalci.metrics import run_metric
from temporalci.report import write_html_report
from temporalci.types import GateSpec
from temporalci.types import GeneratedSample
from temporalci.types import SuiteSpec


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%S%fZ")


def _create_run_dir(model_root: Path) -> tuple[str, Path]:
    # Timestamp collisions are unlikely but possible on fast successive runs.
    for _ in range(20):
        run_id = _new_run_id()
        run_dir = model_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("failed to allocate unique run directory after retries")


def _resolve_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == ">=":
        return float(actual) >= float(expected)
    if op == "<=":
        return float(actual) <= float(expected)
    if op == ">":
        return float(actual) > float(expected)
    if op == "<":
        return float(actual) < float(expected)
    raise ValueError(f"unsupported operator: {op}")


def _evaluate_gates(gates: list[GateSpec], metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for gate in gates:
        result: dict[str, Any] = {
            "metric": gate.metric,
            "op": gate.op,
            "value": gate.value,
            "actual": None,
            "passed": False,
        }
        if gate.op not in GATE_OPERATORS:
            result["error"] = f"unsupported operator: {gate.op}"
            results.append(result)
            continue

        try:
            actual = _resolve_path(metrics_payload, gate.metric)
            result["actual"] = actual
            result["passed"] = _compare(actual, gate.op, gate.value)
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            result["passed"] = False
        results.append(result)
    return results


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
        if not child.is_dir():
            continue
        if child.name == current_run_id:
            continue
        run_json = child / "run.json"
        if run_json.exists():
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
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
            current_value = _resolve_path(current_metrics, metric)
            baseline_value = _resolve_path(baseline_metrics, metric)
        except KeyError:
            continue
        if not _is_number(current_value) or not _is_number(baseline_value):
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


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
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


def run_suite(
    *,
    suite: SuiteSpec,
    model_name: str | None = None,
    artifacts_dir: str | Path = "artifacts",
    fail_on_regression: bool = True,
    baseline_mode: str = "latest_pass",
) -> dict[str, Any]:
    if baseline_mode not in BASELINE_MODES:
        available = ", ".join(sorted(BASELINE_MODES))
        raise ValueError(f"invalid baseline_mode '{baseline_mode}'. choose: {available}")

    model = select_model(suite, model_name)
    adapter = build_adapter(model)

    timestamp = _utc_now().isoformat()
    model_root = Path(artifacts_dir) / suite.project / suite.suite_name / model.name
    run_id, run_dir = _create_run_dir(model_root=model_root)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

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
                samples.append(sample)

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

    gates = _evaluate_gates(suite.gates, metrics_payload)
    gate_failed = any(not gate.get("passed", False) for gate in gates)

    baseline = _load_previous_run(
        model_root=model_root,
        current_run_id=run_id,
        baseline_mode=baseline_mode,
    )
    baseline_metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
    baseline_run_id = baseline.get("run_id") if isinstance(baseline, dict) else None

    regressions = _compute_regressions(
        gates=gates, current_metrics=metrics_payload, baseline_metrics=baseline_metrics
    )
    regression_failed = any(item["regressed"] for item in regressions)

    should_fail = gate_failed or (fail_on_regression and regression_failed)
    status = "FAIL" if should_fail else "PASS"

    sample_rows = _build_sample_rows_with_retention(
        samples=samples,
        status=status,
        artifacts_cfg=suite.artifacts,
    )

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

    payload["run_dir"] = str(run_dir)
    return payload
