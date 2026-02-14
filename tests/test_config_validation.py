from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from temporalci.config import SuiteValidationError
from temporalci.config import load_suite


def _base_payload() -> dict[str, object]:
    return {
        "version": 1,
        "project": "demo",
        "suite_name": "core",
        "models": [{"name": "m1", "adapter": "mock"}],
        "tests": [{"id": "t1", "type": "generation", "prompts": ["hello"], "seeds": [0]}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.1}],
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    suite_path = path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return suite_path


def test_empty_models_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["models"] = []
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_invalid_gate_operator_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [{"metric": "vbench_temporal.score", "op": "~=", "value": 0.1}]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_unknown_test_type_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["tests"] = [{"id": "t1", "type": "classification", "prompts": ["hello"], "seeds": [0]}]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_artifacts_keep_workdir_parses_string_false(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["artifacts"] = {"keep_workdir": "false"}
    suite = load_suite(_write_yaml(tmp_path, payload))
    assert suite.artifacts["keep_workdir"] is False


def test_artifacts_keep_workdir_parses_string_true(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["artifacts"] = {"keep_workdir": "true"}
    suite = load_suite(_write_yaml(tmp_path, payload))
    assert suite.artifacts["keep_workdir"] is True


def test_gate_method_sprt_regression_is_parsed(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.dims.motion_smoothness",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"effect_size": 0.05, "min_pairs": 6},
        }
    ]
    suite = load_suite(_write_yaml(tmp_path, payload))
    assert suite.gates[0].method == "sprt_regression"
    assert suite.gates[0].params["effect_size"] == 0.05


def test_gate_method_invalid_is_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "unknown_method",
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_non_directional_operator(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": "==",
            "value": 0.1,
            "method": "sprt_regression",
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_invalid_baseline_missing_policy(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"baseline_missing": "unknown"},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_invalid_pairing_mode(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"pairing_mode": "bad_mode"},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_skip_policy_when_require_baseline_true(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"require_baseline": True, "baseline_missing": "skip"},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_allows_skip_policy_with_require_baseline_false(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"require_baseline": False, "baseline_missing": "skip"},
        }
    ]
    suite = load_suite(_write_yaml(tmp_path, payload))
    assert suite.gates[0].params["require_baseline"] is False
    assert suite.gates[0].params["baseline_missing"] == "skip"


def test_gate_sprt_rejects_invalid_sigma_mode(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"sigma_mode": "bad_mode"},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_fixed_sigma_mode_without_sigma(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"sigma_mode": "fixed"},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_rejects_non_positive_sigma(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"sigma_mode": "fixed", "sigma": 0},
        }
    ]
    with pytest.raises(SuiteValidationError):
        load_suite(_write_yaml(tmp_path, payload))


def test_gate_sprt_allows_fixed_sigma_mode_with_positive_sigma(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.1,
            "method": "sprt_regression",
            "params": {"sigma_mode": "fixed", "sigma": 0.05},
        }
    ]
    suite = load_suite(_write_yaml(tmp_path, payload))
    assert suite.gates[0].params["sigma_mode"] == "fixed"
    assert suite.gates[0].params["sigma"] == 0.05
