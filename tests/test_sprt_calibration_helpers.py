"""Extended coverage tests for temporalci.sprt_calibration.

Covers branches not exercised by tests/test_calibrate_sprt.py:

  _quantile           — empty list, q<=0, q>=1
  _mad_sigma          — empty list
  _resolve_sprt_gate  — no SPRT gates, metric not found, metric found
  _load_run_payload   — missing file, non-dict JSON, valid payload
  _load_calibration_summary — missing file, non-dict JSON
  _validate_calibration_schema — non-integer schema_version
  _validate_suite_hash — missing hash field, hash match (True,None)
  _load_suite_yaml    — non-mapping YAML root
  _apply_recommended_params_to_suite — no gates list, non-dict gate,
                        wrong method, target not found, auto-create params,
                        apply_inplace (creates .bak), apply_out=None+!inplace
  _evaluate_checks    — min_total_deltas failure, max/min_recommended_sigma failures
  _validate_check_threshold_args — max_mismatch<0, sigma<=0 branches
  run_calibration     — runs<=0, threshold error, inplace+out conflict,
                        baseline_run_id path, missing baseline metrics,
                        candidate metrics skipped
  run_apply_from_calibration — inplace+out conflict, missing recommended_params,
                        empty gate_metric, default output path, inplace path
  run_check_from_calibration — passing path (return 0), failing path (return 2)
  main()              — dispatches to calibrate_main
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from temporalci.sprt_calibration import (
    _apply_recommended_params_to_suite,
    _evaluate_checks,
    _load_calibration_summary,
    _load_run_payload,
    _load_suite_yaml,
    _mad_sigma,
    _quantile,
    _resolve_sprt_gate,
    _validate_calibration_schema,
    _validate_check_threshold_args,
    _validate_suite_hash,
    main,
    run_apply_from_calibration,
    run_calibration,
    run_check_from_calibration,
)
from temporalci.types import GateSpec, MetricSpec, ModelSpec, SuiteSpec
from temporalci.types import TestSpec as SuiteTestSpec


# ---------------------------------------------------------------------------
# Shared suite-YAML helpers (reused from test_calibrate_sprt.py pattern)
# ---------------------------------------------------------------------------


def _write_suite(path: Path, *, gate_metric: str = "vbench_temporal.dims.motion_smoothness") -> Path:
    payload = {
        "version": 1,
        "project": "demo",
        "suite_name": "sprt-calibrate",
        "models": [{"name": "mock-model", "adapter": "mock"}],
        "tests": [
            {
                "id": "t1",
                "type": "generation",
                "prompts": ["a calm city timelapse"],
                "seeds": [0, 1],
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [
            {
                "metric": gate_metric,
                "op": ">=",
                "value": 0.0,
                "method": "sprt_regression",
                "params": {
                    "alpha": 0.05,
                    "beta": 0.1,
                    "effect_size": 0.03,
                    "sigma_floor": 0.01,
                    "min_pairs": 4,
                    "min_paired_ratio": 1.0,
                    "pairing_mismatch": "fail",
                    "pairing_mode": "sample_id",
                },
            }
        ],
    }
    suite_path = path / "suite_sprt_calibrate.yaml"
    suite_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return suite_path


def _write_calibration_json(
    path: Path,
    *,
    suite_path: Path,
    schema_version: int = 1,
    suite_hash_sha1: str | None = None,
    recommended_sigma: float = 0.04,
    delta_count: int = 4,
) -> Path:
    if suite_hash_sha1 is None:
        suite_hash_sha1 = hashlib.sha1(suite_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": schema_version,
        "suite_hash_sha1": suite_hash_sha1,
        "gate_metric": "vbench_temporal.dims.motion_smoothness",
        "recommended_params": {
            "sigma_mode": "fixed",
            "sigma": recommended_sigma,
            "min_pairs": 4,
        },
        "sprt_params": {"min_paired_ratio": 1.0},
        "delta_summary": {"count": delta_count},
        "run_summaries": [{"paired_ratio": 1.0}],
    }
    calibration_path = path / "calibration.json"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    return calibration_path


def _make_sprt_suite(*, extra_gates: list[GateSpec] | None = None) -> SuiteSpec:
    gates: list[GateSpec] = [
        GateSpec(
            metric="score",
            op=">=",
            value=0.0,
            method="sprt_regression",
        )
    ]
    if extra_gates:
        gates.extend(extra_gates)
    return SuiteSpec(
        version=1,
        project="demo",
        suite_name="core",
        models=[ModelSpec(name="m", adapter="mock")],
        tests=[SuiteTestSpec(id="t1", type="generation", prompts=["p"], seeds=[0])],
        metrics=[MetricSpec(name="vbench_temporal")],
        gates=gates,
    )


# ---------------------------------------------------------------------------
# _quantile
# ---------------------------------------------------------------------------


def test_quantile_empty_returns_none() -> None:
    assert _quantile([], 0.5) is None


def test_quantile_q_zero_returns_min() -> None:
    assert _quantile([3.0, 1.0, 2.0], 0.0) == pytest.approx(1.0)


def test_quantile_q_negative_returns_min() -> None:
    assert _quantile([3.0, 1.0, 2.0], -0.5) == pytest.approx(1.0)


def test_quantile_q_one_returns_max() -> None:
    assert _quantile([3.0, 1.0, 2.0], 1.0) == pytest.approx(3.0)


def test_quantile_q_above_one_returns_max() -> None:
    assert _quantile([3.0, 1.0, 2.0], 1.5) == pytest.approx(3.0)


def test_quantile_median_odd() -> None:
    assert _quantile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)


def test_quantile_median_even_interpolates() -> None:
    result = _quantile([1.0, 2.0, 3.0, 4.0], 0.5)
    assert result == pytest.approx(2.5)


def test_quantile_single_value() -> None:
    assert _quantile([7.0], 0.5) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# _mad_sigma
# ---------------------------------------------------------------------------


def test_mad_sigma_empty_returns_none() -> None:
    assert _mad_sigma([]) is None


def test_mad_sigma_single_value_returns_zero() -> None:
    result = _mad_sigma([5.0])
    assert result == pytest.approx(0.0)


def test_mad_sigma_symmetric_distribution() -> None:
    # [1,2,3,4,5]: median=3, deviations=[2,1,0,1,2], mad=1.0, result=1.4826
    result = _mad_sigma([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(1.4826, rel=1e-3)


def test_mad_sigma_constant_values_returns_zero() -> None:
    result = _mad_sigma([4.0, 4.0, 4.0])
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _resolve_sprt_gate
# ---------------------------------------------------------------------------


def test_resolve_sprt_gate_no_sprt_gates_raises() -> None:
    suite = SuiteSpec(
        version=1,
        project="demo",
        suite_name="core",
        models=[ModelSpec(name="m", adapter="mock")],
        tests=[SuiteTestSpec(id="t1", type="generation", prompts=["p"], seeds=[0])],
        metrics=[MetricSpec(name="vbench_temporal")],
        gates=[GateSpec(metric="score", op=">=", value=0.5, method="threshold")],
    )
    with pytest.raises(ValueError, match="does not define any"):
        _resolve_sprt_gate(suite, None)


def test_resolve_sprt_gate_metric_not_found_raises() -> None:
    suite = _make_sprt_suite()
    with pytest.raises(ValueError, match="not found in sprt gates"):
        _resolve_sprt_gate(suite, "missing.metric")


def test_resolve_sprt_gate_finds_specific_metric() -> None:
    extra = GateSpec(metric="other.score", op=">=", value=0.1, method="sprt_regression")
    suite = _make_sprt_suite(extra_gates=[extra])
    gate = _resolve_sprt_gate(suite, "other.score")
    assert gate.metric == "other.score"


def test_resolve_sprt_gate_single_gate_no_metric_required() -> None:
    suite = _make_sprt_suite()
    gate = _resolve_sprt_gate(suite, None)
    assert gate.metric == "score"


# ---------------------------------------------------------------------------
# _load_run_payload
# ---------------------------------------------------------------------------


def test_load_run_payload_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_run_payload(tmp_path / "missing.json")


def test_load_run_payload_non_dict_raises(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be object"):
        _load_run_payload(path)


def test_load_run_payload_valid_returns_dict(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"status": "PASS", "run_id": "r1"}', encoding="utf-8")
    payload = _load_run_payload(path)
    assert payload == {"status": "PASS", "run_id": "r1"}


# ---------------------------------------------------------------------------
# _load_calibration_summary
# ---------------------------------------------------------------------------


def test_load_calibration_summary_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_calibration_summary(tmp_path / "nope.json")


def test_load_calibration_summary_non_dict_raises(tmp_path: Path) -> None:
    path = tmp_path / "calib.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        _load_calibration_summary(path)


def test_load_calibration_summary_valid_returns_dict(tmp_path: Path) -> None:
    path = tmp_path / "calib.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    result = _load_calibration_summary(path)
    assert result["schema_version"] == 1


# ---------------------------------------------------------------------------
# _validate_calibration_schema
# ---------------------------------------------------------------------------


def test_validate_schema_missing_key_returns_false() -> None:
    ok, err = _validate_calibration_schema({})
    assert not ok
    assert err is not None and "schema_version" in err


def test_validate_schema_non_integer_returns_false() -> None:
    ok, err = _validate_calibration_schema({"schema_version": "1"})
    assert not ok
    assert err is not None and "integer" in err


def test_validate_schema_unknown_version_returns_false() -> None:
    ok, err = _validate_calibration_schema({"schema_version": 999})
    assert not ok
    assert err is not None and "unsupported" in err


def test_validate_schema_valid_returns_true() -> None:
    ok, err = _validate_calibration_schema({"schema_version": 1})
    assert ok
    assert err is None


# ---------------------------------------------------------------------------
# _validate_suite_hash
# ---------------------------------------------------------------------------


def test_validate_suite_hash_missing_field_returns_false(tmp_path: Path) -> None:
    ok, err = _validate_suite_hash({}, tmp_path / "suite.yaml")
    assert not ok
    assert err is not None and "missing suite_hash_sha1" in err


def test_validate_suite_hash_empty_string_returns_false(tmp_path: Path) -> None:
    ok, err = _validate_suite_hash({"suite_hash_sha1": "   "}, tmp_path / "suite.yaml")
    assert not ok


def test_validate_suite_hash_mismatch_returns_false(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("content", encoding="utf-8")
    ok, err = _validate_suite_hash({"suite_hash_sha1": "0" * 40}, suite_path)
    assert not ok
    assert err is not None and "mismatch" in err


def test_validate_suite_hash_match_returns_true(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    data = b"suite_content"
    suite_path.write_bytes(data)
    real_hash = hashlib.sha1(data).hexdigest()
    ok, err = _validate_suite_hash({"suite_hash_sha1": real_hash}, suite_path)
    assert ok
    assert err is None


# ---------------------------------------------------------------------------
# _load_suite_yaml
# ---------------------------------------------------------------------------


def test_load_suite_yaml_non_mapping_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        _load_suite_yaml(path)


def test_load_suite_yaml_valid_mapping(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text("project: demo\n", encoding="utf-8")
    result = _load_suite_yaml(path)
    assert result["project"] == "demo"


# ---------------------------------------------------------------------------
# _apply_recommended_params_to_suite
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


_RECOMMENDED = {"sigma": 0.05, "min_pairs": 6}
_TARGET_METRIC = "vbench_temporal.dims.motion_smoothness"


def _base_gate_yaml(
    *,
    metric: str = _TARGET_METRIC,
    method: str = "sprt_regression",
    include_params: bool = True,
) -> dict:
    gate: dict = {"metric": metric, "method": method, "op": ">=", "value": 0.0}
    if include_params:
        gate["params"] = {"alpha": 0.05, "effect_size": 0.03}
    return gate


def test_apply_params_no_gates_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"project": "demo"})
    with pytest.raises(ValueError, match="gates"):
        _apply_recommended_params_to_suite(
            suite_path=path,
            gate_metric=_TARGET_METRIC,
            recommended_params=_RECOMMENDED,
            apply_out=tmp_path / "out.yaml",
            apply_inplace=False,
        )


def test_apply_params_non_list_gates_raises(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": "not_a_list"})
    with pytest.raises(ValueError, match="gates"):
        _apply_recommended_params_to_suite(
            suite_path=path,
            gate_metric=_TARGET_METRIC,
            recommended_params=_RECOMMENDED,
            apply_out=tmp_path / "out.yaml",
            apply_inplace=False,
        )


def test_apply_params_target_gate_not_found_raises(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": [_base_gate_yaml(metric="other.metric")]})
    with pytest.raises(ValueError, match="target gate not found"):
        _apply_recommended_params_to_suite(
            suite_path=path,
            gate_metric=_TARGET_METRIC,
            recommended_params=_RECOMMENDED,
            apply_out=tmp_path / "out.yaml",
            apply_inplace=False,
        )


def test_apply_params_skips_non_dict_gate_entry(tmp_path: Path) -> None:
    # A string in the gates list should be skipped; the valid dict gate after it is found
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": ["not_a_dict", _base_gate_yaml()]})
    out = tmp_path / "out.yaml"
    applied_path, diff = _apply_recommended_params_to_suite(
        suite_path=path,
        gate_metric=_TARGET_METRIC,
        recommended_params=_RECOMMENDED,
        apply_out=out,
        apply_inplace=False,
    )
    assert out.exists()


def test_apply_params_skips_wrong_method_gate(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": [_base_gate_yaml(method="threshold")]})
    with pytest.raises(ValueError, match="target gate not found"):
        _apply_recommended_params_to_suite(
            suite_path=path,
            gate_metric=_TARGET_METRIC,
            recommended_params=_RECOMMENDED,
            apply_out=tmp_path / "out.yaml",
            apply_inplace=False,
        )


def test_apply_params_auto_creates_params_dict(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    gate = _base_gate_yaml(include_params=False)  # no "params" key
    _write_yaml(path, {"gates": [gate]})
    out = tmp_path / "out.yaml"
    applied_path, diff = _apply_recommended_params_to_suite(
        suite_path=path,
        gate_metric=_TARGET_METRIC,
        recommended_params=_RECOMMENDED,
        apply_out=out,
        apply_inplace=False,
    )
    calibrated = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert calibrated["gates"][0]["params"]["sigma_mode"] == "fixed"
    assert calibrated["gates"][0]["params"]["sigma"] == pytest.approx(0.05)


def test_apply_params_apply_inplace_creates_bak(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": [_base_gate_yaml()]})
    bak_path = path.with_suffix(path.suffix + ".bak")
    applied_path, diff = _apply_recommended_params_to_suite(
        suite_path=path,
        gate_metric=_TARGET_METRIC,
        recommended_params=_RECOMMENDED,
        apply_out=None,
        apply_inplace=True,
    )
    assert bak_path.exists()
    assert applied_path == path


def test_apply_params_apply_out_none_not_inplace_raises(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": [_base_gate_yaml()]})
    with pytest.raises(ValueError, match="apply_out must be set"):
        _apply_recommended_params_to_suite(
            suite_path=path,
            gate_metric=_TARGET_METRIC,
            recommended_params=_RECOMMENDED,
            apply_out=None,
            apply_inplace=False,
        )


def test_apply_params_before_after_diff_recorded(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    _write_yaml(path, {"gates": [_base_gate_yaml()]})
    out = tmp_path / "out.yaml"
    _, diff = _apply_recommended_params_to_suite(
        suite_path=path,
        gate_metric=_TARGET_METRIC,
        recommended_params=_RECOMMENDED,
        apply_out=out,
        apply_inplace=False,
    )
    assert "before" in diff
    assert "after" in diff
    assert diff["after"]["sigma_mode"] == "fixed"
    assert diff["after"]["sigma"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _evaluate_checks
# ---------------------------------------------------------------------------


def _make_check_summary(
    *,
    delta_count: int = 10,
    sigma: float = 0.05,
    mismatch_run_paired_ratio: float = 1.0,
) -> dict[str, Any]:
    return {
        "delta_summary": {"count": delta_count},
        "recommended_params": {"sigma": sigma},
        "run_summaries": [{"paired_ratio": mismatch_run_paired_ratio}],
        "sprt_params": {"min_paired_ratio": 1.0},
    }


def _call_evaluate_checks(summary: dict, **overrides) -> list[str]:
    defaults: dict[str, Any] = {
        "fail_if_no_deltas": False,
        "min_total_deltas": None,
        "max_mismatch_runs": None,
        "max_recommended_sigma": None,
        "min_recommended_sigma": None,
    }
    defaults.update(overrides)
    return _evaluate_checks(summary=summary, **defaults)


def test_evaluate_checks_min_total_deltas_failure() -> None:
    summary = _make_check_summary(delta_count=2)
    failures = _call_evaluate_checks(summary, min_total_deltas=5)
    assert any("min-total-deltas" in f for f in failures)


def test_evaluate_checks_max_recommended_sigma_failure() -> None:
    summary = _make_check_summary(sigma=0.2)
    failures = _call_evaluate_checks(summary, max_recommended_sigma=0.1)
    assert any("max-recommended-sigma" in f for f in failures)


def test_evaluate_checks_min_recommended_sigma_failure() -> None:
    summary = _make_check_summary(sigma=0.01)
    failures = _call_evaluate_checks(summary, min_recommended_sigma=0.05)
    assert any("min-recommended-sigma" in f for f in failures)


def test_evaluate_checks_all_pass() -> None:
    summary = _make_check_summary(delta_count=10, sigma=0.05)
    failures = _call_evaluate_checks(
        summary,
        min_total_deltas=5,
        max_recommended_sigma=0.1,
        min_recommended_sigma=0.01,
        max_mismatch_runs=1,
    )
    assert failures == []


def test_evaluate_checks_fail_if_no_deltas() -> None:
    summary = _make_check_summary(delta_count=0)
    failures = _call_evaluate_checks(summary, fail_if_no_deltas=True)
    assert any("fail-if-no-deltas" in f for f in failures)


def test_evaluate_checks_max_mismatch_runs_failure() -> None:
    # Run with paired_ratio < min_paired_ratio=1.0 triggers mismatch_runs count
    summary = _make_check_summary(mismatch_run_paired_ratio=0.5)
    failures = _call_evaluate_checks(summary, max_mismatch_runs=0)
    assert any("mismatch_runs" in f for f in failures)


# ---------------------------------------------------------------------------
# _validate_check_threshold_args
# ---------------------------------------------------------------------------


def _call_validate(**kwargs) -> str | None:
    defaults: dict[str, Any] = {
        "min_total_deltas": None,
        "max_mismatch_runs": None,
        "max_recommended_sigma": None,
        "min_recommended_sigma": None,
    }
    defaults.update(kwargs)
    return _validate_check_threshold_args(**defaults)


def test_validate_check_max_mismatch_runs_negative() -> None:
    err = _call_validate(max_mismatch_runs=-1)
    assert err is not None
    assert "--max-mismatch-runs" in err


def test_validate_check_max_recommended_sigma_zero() -> None:
    err = _call_validate(max_recommended_sigma=0.0)
    assert err is not None
    assert "--max-recommended-sigma" in err


def test_validate_check_max_recommended_sigma_negative() -> None:
    err = _call_validate(max_recommended_sigma=-0.1)
    assert err is not None


def test_validate_check_min_recommended_sigma_zero() -> None:
    err = _call_validate(min_recommended_sigma=0.0)
    assert err is not None
    assert "--min-recommended-sigma" in err


def test_validate_check_min_recommended_sigma_negative() -> None:
    err = _call_validate(min_recommended_sigma=-1.0)
    assert err is not None


def test_validate_check_all_valid_returns_none() -> None:
    err = _call_validate(
        min_total_deltas=5,
        max_mismatch_runs=2,
        max_recommended_sigma=0.2,
        min_recommended_sigma=0.01,
    )
    assert err is None


# ---------------------------------------------------------------------------
# run_calibration — error-path early exits
# ---------------------------------------------------------------------------


def test_run_calibration_runs_zero_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    code = run_calibration(
        suite=str(suite_path),
        runs=0,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 1


def test_run_calibration_runs_negative_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    code = run_calibration(
        suite=str(suite_path),
        runs=-5,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 1


def test_run_calibration_invalid_threshold_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    code = run_calibration(
        suite=str(suite_path),
        min_total_deltas=-1,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 1


def test_run_calibration_inplace_and_out_conflict_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    code = run_calibration(
        suite=str(suite_path),
        apply_inplace=True,
        apply_out=str(tmp_path / "out.yaml"),
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 1


def test_run_calibration_with_existing_baseline_run_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    suite_path = _write_suite(tmp_path)
    baseline_run_id = "run_baseline"
    # Create baseline run.json at the path _resolve_run_json_path builds
    baseline_dir = tmp_path / "demo" / "sprt-calibrate" / "mock-model" / baseline_run_id
    baseline_dir.mkdir(parents=True)
    baseline_payload = {
        "run_id": baseline_run_id,
        "status": "PASS",
        "metrics": {"vbench_temporal": {"per_sample": []}},
    }
    (baseline_dir / "run.json").write_text(json.dumps(baseline_payload), encoding="utf-8")

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        return {
            "run_id": "run_cand",
            "status": "PASS",
            "metrics": {"vbench_temporal": {"per_sample": []}},
        }

    def _fake_pairing(**_: Any) -> tuple[list[float], dict[str, Any]]:
        return [0.01, -0.01], {"paired_count": 2, "expected_pairs": 2, "paired_ratio": 1.0}

    monkeypatch.setattr("temporalci.sprt_calibration.run_suite", _fake_run_suite)
    monkeypatch.setattr("temporalci.sprt_calibration._paired_deltas_for_gate", _fake_pairing)

    code = run_calibration(
        suite=str(suite_path),
        runs=1,
        baseline_run_id=baseline_run_id,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["baseline_run_id"] == baseline_run_id


def test_run_calibration_baseline_missing_metrics_returns_1(
    tmp_path: Path, monkeypatch: Any
) -> None:
    suite_path = _write_suite(tmp_path)
    call_count = {"n": 0}

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        call_count["n"] += 1
        # First call is baseline — return non-dict metrics
        return {"run_id": f"run-{call_count['n']}", "status": "PASS", "metrics": None}

    monkeypatch.setattr("temporalci.sprt_calibration.run_suite", _fake_run_suite)

    code = run_calibration(
        suite=str(suite_path),
        runs=1,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 1


def test_run_calibration_candidate_missing_metrics_skipped(
    tmp_path: Path, monkeypatch: Any
) -> None:
    suite_path = _write_suite(tmp_path)
    call_count = {"n": 0}

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "run_id": "baseline",
                "status": "PASS",
                "metrics": {"vbench_temporal": {"per_sample": []}},
            }
        # Candidate has no metrics dict → should be skipped (line 445: continue)
        return {"run_id": "cand", "status": "PASS", "metrics": None}

    def _fake_pairing(**_: Any) -> tuple[list[float], dict[str, Any]]:
        return [], {"paired_count": 0, "expected_pairs": 0, "paired_ratio": 0.0}

    monkeypatch.setattr("temporalci.sprt_calibration.run_suite", _fake_run_suite)
    monkeypatch.setattr("temporalci.sprt_calibration._paired_deltas_for_gate", _fake_pairing)

    code = run_calibration(
        suite=str(suite_path),
        runs=2,
        artifacts_dir=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
    )
    assert code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    # Both candidates had None metrics → run_summaries is empty (skipped)
    assert payload["runs_completed"] == 0


# ---------------------------------------------------------------------------
# run_apply_from_calibration — missing branches
# ---------------------------------------------------------------------------


def test_run_apply_inplace_and_out_conflict_returns_1(tmp_path: Path) -> None:
    code = run_apply_from_calibration(
        suite=tmp_path / "suite.yaml",
        calibration_json=tmp_path / "calib.json",
        out=tmp_path / "out.yaml",
        inplace=True,
    )
    assert code == 1


def test_run_apply_missing_recommended_params_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    real_hash = hashlib.sha1(suite_path.read_bytes()).hexdigest()
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps({
            "schema_version": 1,
            "suite_hash_sha1": real_hash,
            "gate_metric": "score",
            # no recommended_params key
        }),
        encoding="utf-8",
    )
    code = run_apply_from_calibration(suite=suite_path, calibration_json=calib_path)
    assert code == 1


def test_run_apply_empty_gate_metric_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    real_hash = hashlib.sha1(suite_path.read_bytes()).hexdigest()
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps({
            "schema_version": 1,
            "suite_hash_sha1": real_hash,
            "gate_metric": "",  # empty → no target_metric
            "recommended_params": {"sigma": 0.05},
        }),
        encoding="utf-8",
    )
    # No gate_metric arg either → should return 1
    code = run_apply_from_calibration(suite=suite_path, calibration_json=calib_path)
    assert code == 1


def test_run_apply_default_output_path_created(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(tmp_path, suite_path=suite_path)
    # out=None, inplace=False → default output: suite_path.with_suffix(".calibrated.yaml")
    code = run_apply_from_calibration(suite=suite_path, calibration_json=calib_path)
    assert code == 0
    expected = suite_path.with_suffix(".calibrated.yaml")
    assert expected.exists()


def test_run_apply_inplace_modifies_suite_in_place(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(tmp_path, suite_path=suite_path)
    bak = suite_path.with_suffix(suite_path.suffix + ".bak")
    code = run_apply_from_calibration(
        suite=suite_path,
        calibration_json=calib_path,
        inplace=True,
    )
    assert code == 0
    assert bak.exists()
    calibrated = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    assert calibrated["gates"][0]["params"]["sigma_mode"] == "fixed"


def test_run_apply_explicit_out_path(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(tmp_path, suite_path=suite_path)
    out_path = tmp_path / "suite_calibrated_explicit.yaml"
    code = run_apply_from_calibration(
        suite=suite_path,
        calibration_json=calib_path,
        out=out_path,
    )
    assert code == 0
    assert out_path.exists()


# ---------------------------------------------------------------------------
# run_check_from_calibration — success and failure paths
# ---------------------------------------------------------------------------


def test_run_check_from_calibration_passes_with_no_thresholds(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(tmp_path, suite_path=suite_path)
    code = run_check_from_calibration(calibration_json=calib_path)
    assert code == 0


def test_run_check_from_calibration_passes_within_thresholds(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(
        tmp_path, suite_path=suite_path, recommended_sigma=0.04, delta_count=10
    )
    code = run_check_from_calibration(
        calibration_json=calib_path,
        max_recommended_sigma=0.1,
        min_total_deltas=5,
    )
    assert code == 0


def test_run_check_from_calibration_fails_when_sigma_too_high(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(
        tmp_path, suite_path=suite_path, recommended_sigma=0.5
    )
    code = run_check_from_calibration(
        calibration_json=calib_path,
        max_recommended_sigma=0.1,
    )
    assert code == 2


def test_run_check_from_calibration_fails_when_delta_count_too_low(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(
        tmp_path, suite_path=suite_path, delta_count=2
    )
    code = run_check_from_calibration(
        calibration_json=calib_path,
        min_total_deltas=10,
    )
    assert code == 2


def test_run_check_from_calibration_invalid_threshold_returns_1(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    calib_path = _write_calibration_json(tmp_path, suite_path=suite_path)
    code = run_check_from_calibration(
        calibration_json=calib_path,
        max_recommended_sigma=-0.1,  # invalid: must be > 0
    )
    assert code == 1


# ---------------------------------------------------------------------------
# main() — dispatches to calibrate_main
# ---------------------------------------------------------------------------


def test_main_dispatches_to_run_calibration(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_calibration(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("temporalci.sprt_calibration.run_calibration", _fake_run_calibration)
    code = main(
        [
            "--suite",
            str(tmp_path / "suite.yaml"),
            "--runs",
            "3",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-json",
            "calib.json",
        ]
    )
    assert code == 0
    assert captured["runs"] == 3
