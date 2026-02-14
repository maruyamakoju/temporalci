from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.calibrate_sprt import _estimate_required_pairs
from scripts.calibrate_sprt import _resolve_sprt_gate
from scripts.calibrate_sprt import main as calibrate_main
from temporalci.types import GateSpec
from temporalci.types import MetricSpec
from temporalci.types import ModelSpec
from temporalci.types import SuiteSpec
from temporalci.types import TestSpec as SuiteTestSpec


def _write_suite(path: Path) -> Path:
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
                "metric": "vbench_temporal.dims.motion_smoothness",
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


def test_estimate_required_pairs_matches_formula() -> None:
    payload = _estimate_required_pairs(alpha=0.05, beta=0.1, effect_size=0.03, sigma=0.04)
    assert payload is not None
    drift = (0.03 * 0.03) / (2.0 * 0.04 * 0.04)
    upper = math.log((1.0 - 0.1) / 0.05)
    lower = math.log(0.1 / (1.0 - 0.05))
    assert payload["drift_per_pair"] == pytest.approx(drift)
    assert payload["upper_threshold"] == pytest.approx(upper)
    assert payload["lower_threshold"] == pytest.approx(lower)
    assert payload["required_pairs_upper"] == pytest.approx(upper / drift)
    assert payload["required_pairs_lower"] == pytest.approx(abs(lower) / drift)


def test_resolve_sprt_gate_requires_metric_when_multiple() -> None:
    suite = SuiteSpec(
        version=1,
        project="demo",
        suite_name="core",
        models=[ModelSpec(name="m", adapter="mock")],
        tests=[SuiteTestSpec(id="t1", type="generation", prompts=["p"], seeds=[0])],
        metrics=[MetricSpec(name="vbench_temporal")],
        gates=[
            GateSpec(
                metric="vbench_temporal.score",
                op=">=",
                value=0.1,
                method="sprt_regression",
            ),
            GateSpec(
                metric="vbench_temporal.dims.motion_smoothness",
                op=">=",
                value=0.1,
                method="sprt_regression",
            ),
        ],
    )
    with pytest.raises(ValueError, match="multiple sprt gates"):
        _resolve_sprt_gate(suite, None)


def test_calibrate_main_writes_summary_json(tmp_path: Path, monkeypatch: Any) -> None:
    suite_path = _write_suite(tmp_path)

    calls = {"run_suite": 0, "pairing": 0}

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        calls["run_suite"] += 1
        return {
            "run_id": f"run-{calls['run_suite']}",
            "status": "PASS",
            "metrics": {"vbench_temporal": {"score": 0.5, "per_sample": []}},
        }

    def _fake_pairing(**_: Any) -> tuple[list[float], dict[str, Any]]:
        calls["pairing"] += 1
        if calls["pairing"] == 1:
            return [0.02, -0.01], {"paired_count": 2, "expected_pairs": 2, "paired_ratio": 1.0}
        return [0.01, 0.0], {"paired_count": 2, "expected_pairs": 2, "paired_ratio": 1.0}

    monkeypatch.setattr("scripts.calibrate_sprt.run_suite", _fake_run_suite)
    monkeypatch.setattr("scripts.calibrate_sprt._paired_deltas_for_gate", _fake_pairing)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_sprt.py",
            "--suite",
            str(suite_path),
            "--runs",
            "2",
            "--artifacts-dir",
            str(tmp_path),
            "--output-json",
            "calibration.json",
        ],
    )
    code = calibrate_main()
    assert code == 0

    output_path = tmp_path / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["runs_requested"] == 2
    assert payload["runs_completed"] == 2
    assert payload["baseline_run_id"] == "run-1"
    assert payload["delta_summary"]["count"] == 4
    assert payload["required_pairs"] is not None
    assert payload["recommended_params"]["sigma_mode"] == "fixed"
    assert payload["recommended_params"]["min_pairs"] >= 4


def test_calibrate_main_apply_out_writes_calibrated_suite(
    tmp_path: Path, monkeypatch: Any
) -> None:
    suite_path = _write_suite(tmp_path)
    apply_out = tmp_path / "suite_calibrated.yaml"

    calls = {"run_suite": 0}

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        calls["run_suite"] += 1
        return {
            "run_id": f"run-{calls['run_suite']}",
            "status": "PASS",
            "metrics": {"vbench_temporal": {"score": 0.5, "per_sample": []}},
        }

    def _fake_pairing(**_: Any) -> tuple[list[float], dict[str, Any]]:
        return [0.02, -0.01], {"paired_count": 2, "expected_pairs": 2, "paired_ratio": 1.0}

    monkeypatch.setattr("scripts.calibrate_sprt.run_suite", _fake_run_suite)
    monkeypatch.setattr("scripts.calibrate_sprt._paired_deltas_for_gate", _fake_pairing)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_sprt.py",
            "--suite",
            str(suite_path),
            "--runs",
            "1",
            "--artifacts-dir",
            str(tmp_path),
            "--output-json",
            "calibration.json",
            "--apply-out",
            str(apply_out),
        ],
    )
    code = calibrate_main()
    assert code == 0
    assert apply_out.exists()

    original = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    calibrated = yaml.safe_load(apply_out.read_text(encoding="utf-8"))
    original_params = original["gates"][0]["params"]
    calibrated_params = calibrated["gates"][0]["params"]
    assert "sigma_mode" not in original_params
    assert "sigma" not in original_params
    assert calibrated_params["sigma_mode"] == "fixed"
    assert isinstance(calibrated_params["sigma"], float)
    assert calibrated_params["min_pairs"] >= int(original_params["min_pairs"])
    assert calibrated_params["min_paired_ratio"] == original_params["min_paired_ratio"]
    assert calibrated_params["pairing_mismatch"] == original_params["pairing_mismatch"]

    payload = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert payload["apply"]["applied"] is True
    assert payload["apply"]["mode"] == "output"


def test_calibrate_main_check_failure_returns_nonzero_and_skips_apply(
    tmp_path: Path, monkeypatch: Any
) -> None:
    suite_path = _write_suite(tmp_path)
    apply_out = tmp_path / "suite_calibrated.yaml"

    calls = {"run_suite": 0}

    def _fake_run_suite(**_: Any) -> dict[str, Any]:
        calls["run_suite"] += 1
        return {
            "run_id": f"run-{calls['run_suite']}",
            "status": "PASS",
            "metrics": {"vbench_temporal": {"score": 0.5, "per_sample": []}},
        }

    def _fake_pairing(**_: Any) -> tuple[list[float], dict[str, Any]]:
        return [], {"paired_count": 0, "expected_pairs": 2, "paired_ratio": 0.0}

    monkeypatch.setattr("scripts.calibrate_sprt.run_suite", _fake_run_suite)
    monkeypatch.setattr("scripts.calibrate_sprt._paired_deltas_for_gate", _fake_pairing)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_sprt.py",
            "--suite",
            str(suite_path),
            "--runs",
            "1",
            "--artifacts-dir",
            str(tmp_path),
            "--output-json",
            "calibration.json",
            "--check",
            "--fail-if-no-deltas",
            "--max-mismatch-runs",
            "0",
            "--apply-out",
            str(apply_out),
        ],
    )
    code = calibrate_main()
    assert code == 2
    assert not apply_out.exists()

    payload = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert payload["check"]["enabled"] is True
    assert payload["check"]["passed"] is False
    assert payload["apply"]["applied"] is False
    assert payload["apply"]["reason"] == "check_failed"
