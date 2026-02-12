from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporalci.cli import main


def _write_minimal_suite(path: Path) -> None:
    suite = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [{"name": "mock1", "adapter": "mock", "params": {}}],
        "tests": [
            {
                "id": "t1",
                "prompts": ["a safe prompt"],
                "seeds": [0],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    import yaml

    path.write_text(yaml.dump(suite), encoding="utf-8")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["list"])
    assert result == 0
    output = capsys.readouterr().out
    assert "mock" in output
    assert "vbench_temporal" in output


def test_cli_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["list", "--json"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert "adapters" in payload
    assert "metrics" in payload


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


def test_cli_validate_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    result = main(["validate", str(suite_file)])
    assert result == 0
    assert "valid suite" in capsys.readouterr().out


def test_cli_validate_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["validate", "/nonexistent/suite.yaml"])
    assert result == 1
    assert "config error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_cli_run_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts = tmp_path / "artifacts"
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts),
        "--baseline-mode", "none",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "status=PASS" in output


def test_cli_run_print_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts = tmp_path / "artifacts"
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts),
        "--baseline-mode", "none",
        "--print-json",
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"


def test_cli_run_missing_suite(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["run", "/nonexistent/suite.yaml"])
    assert result == 1
    assert "config error" in capsys.readouterr().out


def test_cli_run_gate_fail(tmp_path: Path) -> None:
    import yaml

    suite = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [{"name": "mock1", "adapter": "mock", "params": {}}],
        "tests": [
            {
                "id": "t1",
                "prompts": ["test"],
                "seeds": [0],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 99.0}],
    }
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(suite), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts),
        "--baseline-mode", "none",
    ])
    assert result == 2  # FAIL exit code
