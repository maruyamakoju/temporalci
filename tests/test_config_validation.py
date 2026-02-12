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
