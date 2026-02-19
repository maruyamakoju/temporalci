from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# sprt command
# ---------------------------------------------------------------------------


def test_cli_sprt_requires_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["sprt"])
    assert result == 1
    assert "usage: temporalci sprt" in capsys.readouterr().out


def test_cli_sprt_dispatches_to_sprt_main(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_sprt_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("temporalci.cli.sprt_main", _fake_sprt_main)
    result = main(["sprt", "check", "--calibration-json", "out.json"])
    assert result == 0
    assert captured["argv"] == ["check", "--calibration-json", "out.json"]


# ---------------------------------------------------------------------------
# webhook flag
# ---------------------------------------------------------------------------


def test_cli_run_webhook_url_passed_to_run_suite(tmp_path: Path) -> None:
    """--webhook-url is forwarded to run_suite as webhook_url kwarg."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts = tmp_path / "artifacts"

    captured: dict = {}

    def _fake_run_suite(**kwargs):
        captured.update(kwargs)
        return {
            "status": "PASS",
            "run_id": "fake",
            "run_dir": str(tmp_path),
            "model_name": "mock1",
            "sample_count": 0,
            "gate_failed": False,
            "regression_failed": False,
            "gates": [],
            "regressions": [],
        }

    with patch("temporalci.cli.run_suite", side_effect=_fake_run_suite):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(artifacts),
            "--baseline-mode", "none",
            "--webhook-url", "http://test.example/hook",
        ])

    assert result == 0
    assert captured.get("webhook_url") == "http://test.example/hook"


def test_cli_run_no_webhook_url_by_default(tmp_path: Path) -> None:
    """webhook_url defaults to None when --webhook-url is omitted."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts = tmp_path / "artifacts"

    captured: dict = {}

    def _fake_run_suite(**kwargs):
        captured.update(kwargs)
        return {
            "status": "PASS",
            "run_id": "fake",
            "run_dir": str(tmp_path),
            "model_name": "mock1",
            "sample_count": 0,
            "gate_failed": False,
            "regression_failed": False,
            "gates": [],
            "regressions": [],
        }

    with patch("temporalci.cli.run_suite", side_effect=_fake_run_suite):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(artifacts),
            "--baseline-mode", "none",
        ])

    assert result == 0
    assert captured.get("webhook_url") is None


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------


def _make_run_json(tmp_path: Path, run_id: str, score: float, status: str = "PASS") -> Path:
    payload = {
        "run_id": run_id,
        "status": status,
        "timestamp_utc": "2026-02-12T00:00:00+00:00",
        "project": "p",
        "suite_name": "s",
        "model_name": "m",
        "metrics": {"vbench_temporal": {"score": score}},
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.5, "actual": score, "passed": score >= 0.5}],
        "samples": [],
    }
    path = tmp_path / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_compare_creates_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline_json = _make_run_json(tmp_path, "base001", score=0.8)
    candidate_json = _make_run_json(tmp_path, "cand002", score=0.6)
    out = tmp_path / "compare.html"

    result = main([
        "compare",
        "--baseline", str(baseline_json),
        "--candidate", str(candidate_json),
        "--output", str(out),
    ])

    # candidate (0.6) < baseline (0.8) → regression → exit 1
    assert result == 1
    assert out.exists()
    output = capsys.readouterr().out
    assert "compare report" in output


def test_cli_compare_missing_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    candidate_json = _make_run_json(tmp_path, "cand", score=0.6)
    result = main([
        "compare",
        "--baseline", str(tmp_path / "nonexistent.json"),
        "--candidate", str(candidate_json),
    ])
    assert result == 1
    assert "config error" in capsys.readouterr().out


def test_cli_compare_missing_candidate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline_json = _make_run_json(tmp_path, "base", score=0.8)
    result = main([
        "compare",
        "--baseline", str(baseline_json),
        "--candidate", str(tmp_path / "nonexistent.json"),
    ])
    assert result == 1
    assert "config error" in capsys.readouterr().out


def test_cli_compare_report_contains_run_ids(tmp_path: Path) -> None:
    baseline_json = _make_run_json(tmp_path, "base-xyz", score=0.8)
    candidate_json = _make_run_json(tmp_path, "cand-abc", score=0.6)
    out = tmp_path / "compare.html"

    result = main([
        "compare",
        "--baseline", str(baseline_json),
        "--candidate", str(candidate_json),
        "--output", str(out),
    ])

    # Report is always written; exit code depends on whether regression detected
    assert result in (0, 1)
    content = out.read_text(encoding="utf-8")
    assert "base-xyz" in content
    assert "cand-abc" in content


# ---------------------------------------------------------------------------
# trend command
# ---------------------------------------------------------------------------


def _populate_model_root(tmp_path: Path, n_runs: int = 3) -> Path:
    """Write n_runs minimal run artifacts into a model_root directory."""
    model_root = tmp_path / "artifacts" / "proj" / "suite" / "model"
    model_root.mkdir(parents=True)
    import datetime

    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "status": "PASS",
            "timestamp_utc": ts.isoformat(),
            "project": "proj",
            "suite_name": "suite",
            "model_name": "model",
            "metrics": {"vbench_temporal": {"score": 0.7 + i * 0.05}},
            "gates": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "status": "PASS", "timestamp_utc": ts.isoformat(), "sample_count": 0}) + "\n")
    return model_root


def test_cli_trend_creates_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root(tmp_path, n_runs=3)
    out = tmp_path / "trend.html"

    result = main([
        "trend",
        "--model-root", str(model_root),
        "--output", str(out),
    ])

    assert result == 0
    assert out.exists()
    output = capsys.readouterr().out
    assert "trend report" in output
    assert "3 runs" in output


def test_cli_trend_no_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty_root = tmp_path / "empty_model"
    empty_root.mkdir()
    result = main([
        "trend",
        "--model-root", str(empty_root),
    ])
    assert result == 1
    assert "no runs found" in capsys.readouterr().out


def test_cli_trend_last_n_limits_output(tmp_path: Path) -> None:
    model_root = _populate_model_root(tmp_path, n_runs=5)
    out = tmp_path / "trend2.html"

    result = main([
        "trend",
        "--model-root", str(model_root),
        "--output", str(out),
        "--last-n", "2",
    ])

    assert result == 0
    output_text = out.read_text(encoding="utf-8")
    # last-n=2 means only 2 runs shown; --last-n limits runs.jsonl entries read
    assert "<!doctype html>" in output_text


def test_cli_trend_custom_title(tmp_path: Path) -> None:
    model_root = _populate_model_root(tmp_path, n_runs=2)
    out = tmp_path / "trend_titled.html"

    result = main([
        "trend",
        "--model-root", str(model_root),
        "--output", str(out),
        "--title", "My Custom Trend",
    ])

    assert result == 0
    assert "My Custom Trend" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_cli_status_shows_run_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root(tmp_path, n_runs=3)

    result = main([
        "status",
        "--model-root", str(model_root),
        "--last-n", "3",
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert "PASS" in output
    assert "Model:" in output
    assert "Runs:" in output


def test_cli_status_shows_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root(tmp_path, n_runs=2)

    result = main([
        "status",
        "--model-root", str(model_root),
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert "vbench_temporal" in output


def test_cli_status_no_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty_model"
    empty.mkdir()

    result = main([
        "status",
        "--model-root", str(empty),
    ])

    assert result == 1
    assert "no runs found" in capsys.readouterr().out


def test_cli_status_sparkline_shows_p_and_f(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Sparkline contains 'P' for PASS runs."""
    model_root = _populate_model_root(tmp_path, n_runs=3)

    main(["status", "--model-root", str(model_root), "--last-n", "3"])
    output = capsys.readouterr().out
    # sparkline shows P for each pass run
    assert "P P P" in output


# ---------------------------------------------------------------------------
# Priority 1: artifact paths in run output
# ---------------------------------------------------------------------------


def test_cli_run_output_shows_report_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a run, stdout includes '→ report:' pointing to report.html."""
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
    assert "→ report:" in output
    assert "report.html" in output


def test_cli_run_output_shows_index_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a run, stdout includes '→ index:' pointing to the suite index."""
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
    assert "→ index:" in output
    assert "index.html" in output


def test_cli_run_output_no_paths_with_print_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When --print-json is used, stdout is pure JSON (no '→ report:' lines)."""
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
    raw = capsys.readouterr().out
    # Must be valid JSON — not contaminated with → report: lines
    payload = json.loads(raw)
    assert payload["status"] == "PASS"


# ---------------------------------------------------------------------------
# compare command — auto mode (--model-root)
# ---------------------------------------------------------------------------


def _populate_model_root_with_runs(
    tmp_path: Path,
    *,
    n_runs: int = 3,
) -> Path:
    """Create a model_root with n_runs, all PASS, each with a run.json."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir(parents=True)
    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "status": "PASS",
            "timestamp_utc": ts.isoformat(),
            "project": "p",
            "suite_name": "s",
            "model_name": "m",
            "metrics": {"vbench_temporal": {"score": 0.7 + i * 0.05}},
            "gates": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "status": "PASS", "timestamp_utc": ts.isoformat()}) + "\n")
    return model_root


def test_cli_compare_auto_creates_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare --model-root auto-selects runs and writes compare report."""
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    out = tmp_path / "auto_compare.html"

    result = main([
        "compare",
        "--model-root", str(model_root),
        "--output", str(out),
    ])

    assert result == 0
    assert out.exists()
    output = capsys.readouterr().out
    assert "compare report" in output
    assert "auto-selected" in output


def test_cli_compare_auto_shows_run_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    out = tmp_path / "auto_compare.html"
    main(["compare", "--model-root", str(model_root), "--output", str(out)])
    content = out.read_text(encoding="utf-8")
    # Should contain some run IDs from the model
    assert "20260201T" in content or "20260202T" in content or "20260203T" in content


def test_cli_compare_auto_no_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    result = main(["compare", "--model-root", str(empty_root)])
    assert result == 1
    assert "no runs" in capsys.readouterr().out


def test_cli_compare_auto_only_one_run_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=1)
    result = main(["compare", "--model-root", str(model_root)])
    assert result == 1
    assert "at least 2" in capsys.readouterr().out


def test_cli_compare_no_args_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare with no --model-root and no --baseline/--candidate fails."""
    result = main(["compare"])
    assert result == 1
    assert "--model-root" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# prune command
# ---------------------------------------------------------------------------


def _make_prune_model_root(tmp_path: Path, n_runs: int = 5) -> Path:
    import datetime

    model_root = tmp_path / "prune_model"
    model_root.mkdir()
    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "status": "PASS"}) + "\n")
    return model_root


def test_cli_prune_deletes_old_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _make_prune_model_root(tmp_path, n_runs=5)
    result = main([
        "prune",
        "--model-root", str(model_root),
        "--keep-last", "3",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "deleted=2" in output
    assert "kept=3" in output


def test_cli_prune_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _make_prune_model_root(tmp_path, n_runs=4)
    dirs_before = sorted(d.name for d in model_root.iterdir() if d.is_dir())
    result = main([
        "prune",
        "--model-root", str(model_root),
        "--keep-last", "2",
        "--dry-run",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "[dry-run]" in output
    # Nothing actually deleted
    dirs_after = sorted(d.name for d in model_root.iterdir() if d.is_dir())
    assert dirs_before == dirs_after


def test_cli_prune_missing_model_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["prune", "--model-root", str(tmp_path / "nonexistent")])
    assert result == 1
    assert "config error" in capsys.readouterr().out


def test_cli_prune_keep_last_default_keeps_20(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Default keep-last=20 keeps all runs when total is less than 20."""
    model_root = _make_prune_model_root(tmp_path, n_runs=5)
    result = main(["prune", "--model-root", str(model_root)])
    assert result == 0
    output = capsys.readouterr().out
    assert "deleted=0" in output
    assert "kept=5" in output


# ---------------------------------------------------------------------------
# status --verbose
# ---------------------------------------------------------------------------


def test_cli_status_verbose_shows_dims(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--verbose shows nested metric paths like vbench_temporal.dims.motion_smoothness."""
    model_root = tmp_path / "model"
    model_root.mkdir()
    import datetime

    ts = datetime.datetime(2026, 2, 1)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "status": "PASS",
        "timestamp_utc": ts.isoformat(),
        "project": "p",
        "suite_name": "s",
        "model_name": "m",
        "metrics": {
            "vbench_temporal": {
                "score": 0.75,
                "dims": {"motion_smoothness": 0.80, "temporal_flicker": 0.70},
            }
        },
        "gates": [],
        "samples": [],
        "sample_count": 0,
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "status": "PASS", "timestamp_utc": ts.isoformat(), "sample_count": 0}) + "\n")

    result = main(["status", "--model-root", str(model_root), "--verbose"])
    assert result == 0
    output = capsys.readouterr().out
    assert "motion_smoothness" in output
    assert "temporal_flicker" in output


def test_cli_status_no_verbose_hides_dims(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without --verbose, nested dim paths are hidden."""
    model_root = tmp_path / "model"
    model_root.mkdir()
    import datetime

    ts = datetime.datetime(2026, 2, 1)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "status": "PASS",
        "timestamp_utc": ts.isoformat(),
        "project": "p",
        "suite_name": "s",
        "model_name": "m",
        "metrics": {
            "vbench_temporal": {
                "score": 0.75,
                "dims": {"motion_smoothness": 0.80, "temporal_flicker": 0.70},
            }
        },
        "gates": [],
        "samples": [],
        "sample_count": 0,
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "status": "PASS", "timestamp_utc": ts.isoformat(), "sample_count": 0}) + "\n")

    result = main(["status", "--model-root", str(model_root)])
    assert result == 0
    output = capsys.readouterr().out
    # score shown, dims hidden
    assert "vbench_temporal.score" in output
    assert "motion_smoothness" not in output
    # Hint about hidden paths
    assert "hidden" in output or "--verbose" in output


# ---------------------------------------------------------------------------
# compare text summary output (P2)
# ---------------------------------------------------------------------------


def test_cli_compare_auto_prints_text_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare --model-root prints format_compare_text summary before HTML path."""
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    out = tmp_path / "cmp.html"
    result = main(["compare", "--model-root", str(model_root), "--output", str(out)])
    assert result == 0
    output = capsys.readouterr().out
    # Text summary shows baseline/candidate line
    assert "baseline" in output
    assert "candidate" in output
    # HTML path still shown
    assert "compare report" in output


def test_cli_compare_explicit_prints_text_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare --baseline --candidate also prints text summary."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir()
    for i, run_id in enumerate(["run_001", "run_002"]):
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "status": "PASS",
            "timestamp_utc": datetime.datetime(2026, 2, i + 1).isoformat(),
            "metrics": {},
            "gates": [],
            "regressions": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    b_path = model_root / "run_001" / "run.json"
    c_path = model_root / "run_002" / "run.json"
    out = tmp_path / "cmp.html"
    result = main(["compare", "--baseline", str(b_path), "--candidate", str(c_path), "--output", str(out)])
    assert result == 0
    output = capsys.readouterr().out
    assert "baseline" in output
    assert "compare report" in output


# ---------------------------------------------------------------------------
# run --prune-keep-last (P1)
# ---------------------------------------------------------------------------


def test_cli_run_prune_keep_last_removes_old_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run --prune-keep-last N prunes old runs after successful run."""
    suite_path = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_path)
    artifacts_dir = str(tmp_path / "artifacts")

    # Run twice to create history, then third run with prune
    main(["run", str(suite_path), "--artifacts-dir", artifacts_dir])
    main(["run", str(suite_path), "--artifacts-dir", artifacts_dir])
    result = main([
        "run", str(suite_path),
        "--artifacts-dir", artifacts_dir,
        "--prune-keep-last", "2",
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert "prune:" in output
    assert "kept=2" in output


def test_cli_run_no_prune_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run without --prune-keep-last does not print prune line."""
    suite_path = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_path)
    artifacts_dir = str(tmp_path / "artifacts")
    main(["run", str(suite_path), "--artifacts-dir", artifacts_dir])
    output = capsys.readouterr().out
    assert "prune:" not in output


# ---------------------------------------------------------------------------
# status --suite-root (P4)
# ---------------------------------------------------------------------------


def _make_suite_root(tmp_path: Path, model_names: list[str], n_runs: int = 3) -> Path:
    """Build a suite_root with multiple model subdirectories."""
    import datetime

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    for model_name in model_names:
        model_root = suite_root / model_name
        model_root.mkdir()
        for i in range(n_runs):
            ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
            run_id = ts.strftime("%Y%m%dT%H%M%SZ")
            run_dir = model_root / run_id
            run_dir.mkdir()
            payload = {
                "run_id": run_id,
                "status": "PASS",
                "timestamp_utc": ts.isoformat(),
                "metrics": {},
                "gates": [],
                "samples": [],
            }
            (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
            with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": run_id,
                    "status": "PASS",
                    "timestamp_utc": ts.isoformat(),
                    "sample_count": 0,
                }) + "\n")
    return suite_root


def test_cli_status_suite_root_lists_all_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_root = _make_suite_root(tmp_path, ["model_a", "model_b"])
    result = main(["status", "--suite-root", str(suite_root)])
    assert result == 0
    output = capsys.readouterr().out
    assert "model_a" in output
    assert "model_b" in output


def test_cli_status_suite_root_shows_pass_fail_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_root = _make_suite_root(tmp_path, ["model_x"], n_runs=3)
    result = main(["status", "--suite-root", str(suite_root)])
    assert result == 0
    output = capsys.readouterr().out
    assert "3" in output  # run count


def test_cli_status_suite_root_nonexistent_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["status", "--suite-root", str(tmp_path / "no_such_dir")])
    assert result != 0
    assert "not found" in capsys.readouterr().out


def test_cli_status_suite_root_empty_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_root = tmp_path / "empty_suite"
    suite_root.mkdir()
    result = main(["status", "--suite-root", str(suite_root)])
    assert result != 0
    assert "no models" in capsys.readouterr().out


def test_cli_status_suite_root_failing_model_returns_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """If any model's latest run is FAIL, suite-status returns non-zero."""
    import datetime

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    model_root = suite_root / "bad_model"
    model_root.mkdir()
    ts = datetime.datetime(2026, 2, 1, 0, 0, 0)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {"run_id": run_id, "status": "FAIL", "metrics": {}, "gates": [], "samples": []}
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "status": "FAIL", "timestamp_utc": ts.isoformat(), "sample_count": 0}) + "\n")

    result = main(["status", "--suite-root", str(suite_root)])
    assert result != 0
    output = capsys.readouterr().out
    assert "FAIL" in output


# ---------------------------------------------------------------------------
# prune --suite-root (P1)
# ---------------------------------------------------------------------------


def test_cli_prune_suite_root_prunes_all_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_root = _make_suite_root(tmp_path, ["model_a", "model_b"], n_runs=4)
    result = main([
        "prune",
        "--suite-root", str(suite_root),
        "--keep-last", "2",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "model_a" in output
    assert "model_b" in output
    assert "total freed" in output


def test_cli_prune_suite_root_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_root = _make_suite_root(tmp_path, ["model_x"], n_runs=5)
    # Count dirs before
    dirs_before = list((suite_root / "model_x").iterdir())
    result = main([
        "prune",
        "--suite-root", str(suite_root),
        "--keep-last", "2",
        "--dry-run",
    ])
    assert result == 0
    dirs_after = list((suite_root / "model_x").iterdir())
    # Nothing actually deleted
    assert sorted(d.name for d in dirs_before) == sorted(d.name for d in dirs_after)
    output = capsys.readouterr().out
    assert "[dry-run]" in output


def test_cli_prune_suite_root_nonexistent_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["prune", "--suite-root", str(tmp_path / "no_such"), "--keep-last", "5"])
    assert result != 0
    assert "not found" in capsys.readouterr().out


def test_cli_prune_suite_root_empty_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty_suite"
    empty.mkdir()
    result = main(["prune", "--suite-root", str(empty), "--keep-last", "5"])
    assert result != 0
    assert "no models" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# export command (P2)
# ---------------------------------------------------------------------------


def test_cli_export_csv_creates_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    out = tmp_path / "metrics.csv"
    result = main(["export", "--model-root", str(model_root), "--output", str(out)])
    assert result == 0
    assert out.exists()
    output = capsys.readouterr().out
    assert "exported 3 runs" in output


def test_cli_export_csv_has_header(tmp_path: Path) -> None:
    import csv as csv_mod

    model_root = _populate_model_root_with_runs(tmp_path, n_runs=2)
    out = tmp_path / "out.csv"
    main(["export", "--model-root", str(model_root), "--output", str(out)])
    with out.open(encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        fieldnames = reader.fieldnames or []
    assert "run_id" in fieldnames
    assert "status" in fieldnames


def test_cli_export_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=2)
    out = tmp_path / "runs.jsonl"
    result = main([
        "export",
        "--model-root", str(model_root),
        "--output", str(out),
        "--format", "jsonl",
    ])
    assert result == 0
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_cli_export_last_n(tmp_path: Path) -> None:
    import csv as csv_mod

    model_root = _populate_model_root_with_runs(tmp_path, n_runs=5)
    out = tmp_path / "out.csv"
    main(["export", "--model-root", str(model_root), "--output", str(out), "--last-n", "3"])
    with out.open(encoding="utf-8") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 3


def test_cli_export_missing_model_root_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main([
        "export",
        "--model-root", str(tmp_path / "no_such"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result != 0
    assert "not found" in capsys.readouterr().out


def test_cli_export_empty_history_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "runs.jsonl").write_text("", encoding="utf-8")
    result = main([
        "export",
        "--model-root", str(model_root),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result != 0
    assert "no runs" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run --sample N (P4)
# ---------------------------------------------------------------------------


def test_cli_run_sample_limits_sample_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run --sample 1 produces a run with only 1 sample even if suite has more."""
    suite_path = tmp_path / "suite.yaml"
    # Suite has 3 prompts × 2 seeds = 6 samples total
    suite = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [{"name": "mock1", "adapter": "mock", "params": {}}],
        "tests": [
            {
                "id": "t1",
                "prompts": ["prompt a", "prompt b", "prompt c"],
                "seeds": [0, 1],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    suite_path.write_text(yaml.dump(suite), encoding="utf-8")
    artifacts_dir = str(tmp_path / "artifacts")

    result = main(["run", str(suite_path), "--artifacts-dir", artifacts_dir, "--sample", "1"])
    assert result == 0
    output = capsys.readouterr().out
    # Summary shows sample count = 1
    assert "samples=1" in output


def test_cli_run_sample_zero_same_as_full(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run without --sample runs all samples."""
    suite_path = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_path)
    artifacts_dir = str(tmp_path / "artifacts")
    result = main(["run", str(suite_path), "--artifacts-dir", artifacts_dir])
    assert result == 0
    output = capsys.readouterr().out
    # Should have at least 1 sample (suite has 1 prompt × 1 seed)
    assert "samples=" in output


# ---------------------------------------------------------------------------
# run (multi-model, P1)
# ---------------------------------------------------------------------------


def _write_multi_model_suite(path: Path, *, fail_gate: bool = False) -> None:
    """Write a suite YAML with two mock models."""
    gate_value = 99.0 if fail_gate else 0.0
    suite = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [
            {"name": "model_a", "adapter": "mock", "params": {}},
            {"name": "model_b", "adapter": "mock", "params": {}},
        ],
        "tests": [
            {
                "id": "t1",
                "prompts": ["a safe prompt"],
                "seeds": [0],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": gate_value}],
    }
    path.write_text(yaml.dump(suite), encoding="utf-8")


def test_cli_run_all_models_output_contains_both_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When --model is omitted, both models run and appear in output."""
    suite_file = tmp_path / "suite.yaml"
    _write_multi_model_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    assert result == 0
    output = capsys.readouterr().out
    assert "model_a" in output
    assert "model_b" in output


def test_cli_run_all_models_summary_table_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Summary table with '── summary ──' is printed for multi-model runs."""
    suite_file = tmp_path / "suite.yaml"
    _write_multi_model_suite(suite_file)
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    output = capsys.readouterr().out
    assert "── summary ──" in output
    assert "models passed" in output


def test_cli_run_all_models_exit_zero_when_all_pass(tmp_path: Path) -> None:
    """All models PASS gate → exit 0."""
    suite_file = tmp_path / "suite.yaml"
    _write_multi_model_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    assert result == 0


def test_cli_run_all_models_exit_2_when_any_fail(tmp_path: Path) -> None:
    """Gate failure for models → exit 2."""
    suite_file = tmp_path / "suite.yaml"
    _write_multi_model_suite(suite_file, fail_gate=True)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    assert result == 2


# ---------------------------------------------------------------------------
# run (progress callback, P2)
# ---------------------------------------------------------------------------


def test_cli_run_progress_shown_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Progress '[1/1]' line is printed when --no-progress is not given."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
    ])
    output = capsys.readouterr().out
    assert "[1/1]" in output


def test_cli_run_no_progress_suppresses_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-progress suppresses '[N/M]' progress lines."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    output = capsys.readouterr().out
    assert "[1/1]" not in output


def test_cli_run_print_json_suppresses_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--print-json implicitly disables progress output."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--print-json",
    ])
    output = capsys.readouterr().out
    assert "[1/1]" not in output


# ---------------------------------------------------------------------------
# run (--tag / --baseline-mode tag:<name>, P3)
# ---------------------------------------------------------------------------


def test_cli_run_tag_writes_tags_json(tmp_path: Path) -> None:
    """--tag gold writes tags.json in the model artifact directory."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "none",
        "--no-progress",
        "--tag", "gold",
    ])
    assert result == 0
    tags_files = list(artifacts_dir.rglob("tags.json"))
    assert tags_files, "tags.json not found under artifacts"
    tags = json.loads(tags_files[0].read_text(encoding="utf-8"))
    assert "gold" in tags


def test_cli_run_baseline_mode_tag(tmp_path: Path) -> None:
    """--baseline-mode tag:gold uses tagged run as baseline on second run."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"

    # First run: tag as 'gold'
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "none",
        "--no-progress",
        "--tag", "gold",
    ])
    # Capture the first run_id from tags.json
    tags_file = next(artifacts_dir.rglob("tags.json"))
    first_run_id = json.loads(tags_file.read_text(encoding="utf-8"))["gold"]

    # Second run: use tag:gold as baseline
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "tag:gold",
        "--no-progress",
    ])
    assert result == 0

    # Verify second run.json has baseline_run_id pointing to the tagged run
    model_root = tags_file.parent
    run_jsons = sorted(model_root.glob("*/run.json"))
    assert len(run_jsons) == 2
    second_run = json.loads(run_jsons[-1].read_text(encoding="utf-8"))
    assert second_run.get("baseline_run_id") == first_run_id


def test_cli_run_invalid_baseline_mode_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown --baseline-mode value (not built-in, not tag:) causes error exit."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "bogus_mode",
        "--no-progress",
    ])
    assert result != 0
    assert "error" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# export (--suite-root, P4)
# ---------------------------------------------------------------------------


def _populate_cli_suite_root(
    tmp_path: Path, model_names: list[str], n_runs: int = 2
) -> Path:
    """Create suite_root/model_name/... for each model with runs.jsonl + run.json."""
    import datetime

    suite_root = tmp_path / "suite_export"
    suite_root.mkdir()
    for model_name in model_names:
        model_root = suite_root / model_name
        model_root.mkdir()
        for i in range(n_runs):
            ts = datetime.datetime(2026, 3, i + 1, 0, 0, 0)
            run_id = ts.strftime("%Y%m%dT%H%M%SZ")
            run_dir = model_root / run_id
            run_dir.mkdir()
            payload = {
                "run_id": run_id,
                "status": "PASS",
                "timestamp_utc": ts.isoformat(),
                "model_name": model_name,
                "sample_count": 1,
                "metrics": {"vbench_temporal": {"score": 0.8}},
                "gates": [],
                "samples": [],
            }
            (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
            with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": run_id, "status": "PASS"}) + "\n")
    return suite_root


def test_cli_export_suite_root_creates_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """export --suite-root creates a CSV with rows from all models."""
    suite_root = _populate_cli_suite_root(tmp_path, ["model_a", "model_b"], n_runs=2)
    out = tmp_path / "all.csv"
    result = main(["export", "--suite-root", str(suite_root), "--output", str(out)])
    assert result == 0
    assert out.exists()
    output = capsys.readouterr().out
    assert "exported 4 runs" in output
    assert "all models" in output


def test_cli_export_suite_root_has_model_name_column(tmp_path: Path) -> None:
    """Suite export CSV includes a 'model_name' column with all model names."""
    import csv as csv_mod

    suite_root = _populate_cli_suite_root(tmp_path, ["model_a", "model_b"])
    out = tmp_path / "all.csv"
    main(["export", "--suite-root", str(suite_root), "--output", str(out)])
    with out.open(encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    assert "model_name" in fieldnames
    names_in_csv = {r["model_name"] for r in rows}
    assert "model_a" in names_in_csv
    assert "model_b" in names_in_csv


def test_cli_export_suite_root_not_found_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """export --suite-root with nonexistent path returns non-zero."""
    result = main([
        "export",
        "--suite-root", str(tmp_path / "no_such"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result != 0
    assert "not found" in capsys.readouterr().out


def test_cli_export_suite_root_empty_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """export --suite-root with no model data returns non-zero."""
    empty_suite = tmp_path / "empty"
    empty_suite.mkdir()
    result = main([
        "export",
        "--suite-root", str(empty_suite),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert result != 0
    output = capsys.readouterr().out
    assert "no runs" in output or "not found" in output


# ---------------------------------------------------------------------------
# run --workers N  (P1 CLI)
# ---------------------------------------------------------------------------


def test_cli_run_workers_2_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--workers 2 completes without error and reports correct sample count."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "none",
        "--no-progress",
        "--workers", "2",
    ])
    assert result == 0
    assert "samples=1" in capsys.readouterr().out


def test_cli_run_workers_skipped_count_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """skipped= token appears in output when samples are skipped via retry exhaustion."""
    from unittest.mock import patch
    from temporalci.adapters.mock import MockAdapter

    # Use a gate threshold > 0 so that 0 samples → score=0.0 → gate FAIL
    suite_file = tmp_path / "suite.yaml"
    suite_yaml = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [{"name": "mock1", "adapter": "mock", "params": {}}],
        "tests": [{"id": "t1", "prompts": ["p"], "seeds": [0], "video": {"num_frames": 5}}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.5}],
    }
    suite_file.write_text(yaml.dump(suite_yaml), encoding="utf-8")

    with patch.object(MockAdapter, "generate", side_effect=RuntimeError("fail")):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--baseline-mode", "none",
            "--no-progress",
            "--retry", "2",
        ])
    # Gate fails (score 0.0 < 0.5) so exit 2; skipped= in output
    assert result == 2
    assert "skipped=" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run --retry N  (P2 CLI)
# ---------------------------------------------------------------------------


def test_cli_run_retry_succeeds_after_transient_error(tmp_path: Path) -> None:
    """--retry 2: adapter failing once then succeeding keeps sample_count correct."""
    from unittest.mock import patch
    from temporalci.adapters.mock import MockAdapter

    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)  # 1 prompt × 1 seed = 1 sample

    call_count = [0]
    _unbound = MockAdapter.generate

    def _flaky(self, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("transient")
        return _unbound(self, *args, **kwargs)

    with patch.object(MockAdapter, "generate", _flaky):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--baseline-mode", "none",
            "--no-progress",
            "--retry", "2",
        ])
    assert result == 0
    assert call_count[0] == 2  # 1 fail + 1 success


# ---------------------------------------------------------------------------
# init command (P3)
# ---------------------------------------------------------------------------


def test_cli_init_creates_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """init creates a suite.yaml with valid YAML content."""
    out = tmp_path / "suite.yaml"
    result = main(["init", "--output", str(out)])
    assert result == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "version: 1" in text
    assert "my-project" in text
    output = capsys.readouterr().out
    assert "created" in output


def test_cli_init_custom_project(tmp_path: Path) -> None:
    """init --project sets the project name in the generated file."""
    out = tmp_path / "suite.yaml"
    main(["init", "--project", "video-ci", "--output", str(out)])
    assert "video-ci" in out.read_text(encoding="utf-8")


def test_cli_init_output_is_valid_yaml(tmp_path: Path) -> None:
    """init output can be parsed by PyYAML."""
    out = tmp_path / "suite.yaml"
    main(["init", "--output", str(out)])
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    assert parsed.get("version") == 1
    assert "models" in parsed
    assert "tests" in parsed
    assert "gates" in parsed


def test_cli_init_refuses_existing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """init fails if output already exists without --force."""
    out = tmp_path / "suite.yaml"
    out.write_text("existing", encoding="utf-8")
    result = main(["init", "--output", str(out)])
    assert result != 0
    assert "already exists" in capsys.readouterr().out


def test_cli_init_force_overwrites(tmp_path: Path) -> None:
    """init --force overwrites an existing file."""
    out = tmp_path / "suite.yaml"
    out.write_text("old content", encoding="utf-8")
    result = main(["init", "--output", str(out), "--force"])
    assert result == 0
    assert "version: 1" in out.read_text(encoding="utf-8")


def test_cli_init_generated_suite_validates(tmp_path: Path) -> None:
    """The generated suite.yaml passes 'temporalci validate'."""
    out = tmp_path / "suite.yaml"
    main(["init", "--output", str(out)])
    result = main(["validate", str(out)])
    assert result == 0


# ---------------------------------------------------------------------------
# annotate command (P4)
# ---------------------------------------------------------------------------


def test_cli_annotate_adds_note_to_run_json(tmp_path: Path) -> None:
    """annotate writes the note field into run.json."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    # Find the run.json
    run_jsons = list(artifacts_dir.rglob("run.json"))
    assert len(run_jsons) == 1
    run_json = run_jsons[0]
    run_id = run_json.parent.name
    model_root = run_json.parent.parent

    result = main([
        "annotate",
        "--model-root", str(model_root),
        "--run-id", run_id,
        "--note", "first gold run",
    ])
    assert result == 0
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload.get("note") == "first gold run"


def test_cli_annotate_shows_note_in_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status output includes the note text when a run is annotated."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    run_jsons = list(artifacts_dir.rglob("run.json"))
    run_id = run_jsons[0].parent.name
    model_root = run_jsons[0].parent.parent

    main([
        "annotate",
        "--model-root", str(model_root),
        "--run-id", run_id,
        "--note", "special release",
    ])
    capsys.readouterr()  # clear

    main(["status", "--model-root", str(model_root)])
    output = capsys.readouterr().out
    assert "special release" in output


def test_cli_annotate_missing_model_root_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """annotate with nonexistent --model-root returns error."""
    result = main([
        "annotate",
        "--model-root", str(tmp_path / "no_such"),
        "--run-id", "20260101T000000Z",
        "--note", "test",
    ])
    assert result != 0
    assert "not found" in capsys.readouterr().out


def test_cli_annotate_missing_run_id_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """annotate with nonexistent run-id returns error."""
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = main([
        "annotate",
        "--model-root", str(model_root),
        "--run-id", "20260101T000000Z",
        "--note", "test",
    ])
    assert result != 0
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run --fail-on-skip  (P2 CLI)
# ---------------------------------------------------------------------------


def test_cli_run_fail_on_skip_exits_2_when_skips_occur(tmp_path: Path) -> None:
    """--fail-on-skip causes exit 2 when any sample is skipped."""
    from unittest.mock import patch
    from temporalci.adapters.mock import MockAdapter

    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)

    with patch.object(MockAdapter, "generate", side_effect=RuntimeError("fail")):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--baseline-mode", "none",
            "--no-progress",
            "--retry", "1",
            "--fail-on-skip",
        ])
    assert result == 2


def test_cli_run_fail_on_skip_not_set_is_lenient(tmp_path: Path) -> None:
    """Without --fail-on-skip, skipped samples with passing gate → exit 0."""
    from unittest.mock import patch
    from temporalci.adapters.mock import MockAdapter

    suite_yaml = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [{"name": "mock1", "adapter": "mock", "params": {}}],
        "tests": [{"id": "t1", "prompts": ["p"], "seeds": [0], "video": {"num_frames": 5}}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(suite_yaml), encoding="utf-8")

    with patch.object(MockAdapter, "generate", side_effect=RuntimeError("fail")):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--baseline-mode", "none",
            "--no-progress",
            "--retry", "1",
        ])
    assert result == 0


# ---------------------------------------------------------------------------
# run --baseline-mode rolling:N  (P1 CLI)
# ---------------------------------------------------------------------------


def test_cli_run_rolling_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--baseline-mode rolling:2 completes successfully after enough prior runs."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"

    for _ in range(2):
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(artifacts_dir),
            "--baseline-mode", "none",
            "--no-progress",
        ])
    capsys.readouterr()

    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "rolling:2",
        "--no-progress",
    ])
    assert result == 0


def test_cli_run_rolling_baseline_invalid_raises_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """rolling:0 is invalid and causes an error exit."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--baseline-mode", "rolling:0",
        "--no-progress",
    ])
    assert result != 0
    assert "error" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# report command  (P3 CLI)
# ---------------------------------------------------------------------------


def _run_once_and_get_run_dir(tmp_path: Path) -> Path:
    """Run the minimal suite once and return the run directory path."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(artifacts_dir),
        "--baseline-mode", "none",
        "--no-progress",
    ])
    return next(artifacts_dir.rglob("run.json")).parent


def test_cli_report_run_dir_creates_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """report --run-dir regenerates report.html."""
    run_dir = _run_once_and_get_run_dir(tmp_path)
    (run_dir / "report.html").unlink()
    result = main(["report", "--run-dir", str(run_dir)])
    assert result == 0
    assert (run_dir / "report.html").exists()
    assert "report:" in capsys.readouterr().out


def test_cli_report_run_dir_custom_output(tmp_path: Path) -> None:
    """report --run-dir --output PATH writes to the specified path."""
    run_dir = _run_once_and_get_run_dir(tmp_path)
    custom_out = tmp_path / "my_report.html"
    result = main(["report", "--run-dir", str(run_dir), "--output", str(custom_out)])
    assert result == 0
    assert custom_out.exists()


def test_cli_report_model_root_regenerates_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """report --model-root regenerates reports for every run under the model."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    artifacts_dir = tmp_path / "artifacts"
    for _ in range(2):
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(artifacts_dir),
            "--baseline-mode", "none",
            "--no-progress",
        ])
    model_root = next(artifacts_dir.rglob("runs.jsonl")).parent
    capsys.readouterr()

    result = main(["report", "--model-root", str(model_root)])
    assert result == 0
    assert "regenerated 2" in capsys.readouterr().out


def test_cli_report_missing_run_dir_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """report --run-dir with nonexistent path returns error."""
    result = main(["report", "--run-dir", str(tmp_path / "no_such")])
    assert result != 0
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# status --output-format json/csv  (P4 CLI)
# ---------------------------------------------------------------------------


def test_cli_status_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status --output-format json prints a JSON array of run dicts."""
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    result = main(["status", "--model-root", str(model_root), "--output-format", "json"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert "run_id" in payload[0]


def test_cli_status_json_includes_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status JSON output includes the metrics dict."""
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=2)
    main(["status", "--model-root", str(model_root), "--output-format", "json"])
    runs = json.loads(capsys.readouterr().out)
    assert "metrics" in runs[0]


def test_cli_status_csv_creates_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status --output-format csv writes a CSV to --output PATH."""
    import csv as csv_mod

    model_root = _populate_model_root_with_runs(tmp_path, n_runs=3)
    out = tmp_path / "status.csv"
    result = main([
        "status", "--model-root", str(model_root),
        "--output-format", "csv",
        "--output", str(out),
    ])
    assert result == 0
    assert out.exists()
    with out.open(encoding="utf-8") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 3
    assert "run_id" in rows[0]
    assert "csv:" in capsys.readouterr().out


def test_cli_status_csv_requires_output_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status --output-format csv without --output returns error."""
    model_root = _populate_model_root_with_runs(tmp_path, n_runs=1)
    result = main([
        "status", "--model-root", str(model_root),
        "--output-format", "csv",
    ])
    assert result != 0
    assert "required" in capsys.readouterr().out


def test_cli_status_suite_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status --suite-root --output-format json prints dict of model→runs."""
    suite_root = _populate_cli_suite_root(tmp_path, ["model_a", "model_b"], n_runs=2)
    result = main([
        "status", "--suite-root", str(suite_root),
        "--output-format", "json",
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    assert "model_a" in payload
    assert "model_b" in payload


# ---------------------------------------------------------------------------
# inter_sample_delay CLI wiring
# ---------------------------------------------------------------------------


def _mock_run_suite_result(tmp_path: Path) -> dict:
    return {
        "status": "PASS",
        "run_id": "testid",
        "run_dir": str(tmp_path),
        "model_name": "mock1",
        "sample_count": 1,
        "skipped_count": 0,
        "gate_failed": False,
        "regression_failed": False,
        "gates": [],
        "regressions": [],
    }


def test_cli_run_inter_sample_delay_wired(tmp_path: Path) -> None:
    """--inter-sample-delay is forwarded to run_suite as inter_sample_delay kwarg."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--workers", "2",
            "--inter-sample-delay", "1.5",
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["inter_sample_delay"] == 1.5


def test_cli_run_inter_sample_delay_default_zero(tmp_path: Path) -> None:
    """--inter-sample-delay defaults to 0.0 when not supplied."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["inter_sample_delay"] == 0.0


# ---------------------------------------------------------------------------
# P1: --dry-run
# ---------------------------------------------------------------------------


def test_cli_dry_run_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run prints summary and returns 0 without creating artifacts."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    art_dir = tmp_path / "art"
    result = main(["run", str(suite_file), "--artifacts-dir", str(art_dir), "--dry-run"])
    assert result == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "mock1" in out
    assert "adapter" in out
    assert "samples" in out


def test_cli_dry_run_does_not_create_artifacts(tmp_path: Path) -> None:
    """--dry-run must not create any artifact directories."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    art_dir = tmp_path / "art"
    main(["run", str(suite_file), "--artifacts-dir", str(art_dir), "--dry-run"])
    assert not art_dir.exists()


def test_cli_dry_run_with_sample_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run respects --sample N and shows capped count."""
    suite = {
        "version": 1,
        "project": "p",
        "suite_name": "s",
        "models": [{"name": "m", "adapter": "mock", "params": {}}],
        "tests": [
            {
                "id": "t",
                "prompts": ["a", "b", "c"],
                "seeds": [0, 1, 2],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    import yaml

    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(suite), encoding="utf-8")
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--dry-run",
        "--sample", "4",
    ])
    assert result == 0
    out = capsys.readouterr().out
    # 9 total, capped to 4
    assert "4" in out


def test_cli_dry_run_multi_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run with multiple models reports each model."""
    suite = {
        "version": 1,
        "project": "p",
        "suite_name": "s",
        "models": [
            {"name": "model_a", "adapter": "mock", "params": {}},
            {"name": "model_b", "adapter": "mock", "params": {}},
        ],
        "tests": [{"id": "t", "prompts": ["x"], "seeds": [0], "video": {"num_frames": 5}}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    import yaml

    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(suite), encoding="utf-8")
    result = main(["run", str(suite_file), "--artifacts-dir", str(tmp_path / "art"), "--dry-run"])
    assert result == 0
    out = capsys.readouterr().out
    assert "model_a" in out
    assert "model_b" in out


# ---------------------------------------------------------------------------
# P2: repair-index
# ---------------------------------------------------------------------------


def _populate_model_root_run_jsons(tmp_path: Path, n_runs: int = 3) -> Path:
    """Create model_root with run.json files but NO runs.jsonl."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir()
    for i in range(n_runs):
        ts = datetime.datetime(2026, 4, i + 1)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps({
                "run_id": run_id,
                "timestamp_utc": ts.isoformat(),
                "status": "PASS" if i % 2 == 0 else "FAIL",
                "sample_count": 2,
            }),
            encoding="utf-8",
        )
    return model_root


def test_cli_repair_index_rebuilds_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """repair-index writes a runs.jsonl with one entry per run.json."""
    model_root = _populate_model_root_run_jsons(tmp_path, n_runs=3)
    result = main(["repair-index", "--model-root", str(model_root)])
    assert result == 0
    index = model_root / "runs.jsonl"
    assert index.exists()
    lines = [ln for ln in index.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "run_id" in obj
        assert "status" in obj


def test_cli_repair_index_overwrites_existing_jsonl(
    tmp_path: Path,
) -> None:
    """repair-index replaces a stale/corrupt runs.jsonl."""
    model_root = _populate_model_root_run_jsons(tmp_path, n_runs=2)
    stale = model_root / "runs.jsonl"
    stale.write_text('{"run_id": "stale"}\n' * 10, encoding="utf-8")
    result = main(["repair-index", "--model-root", str(model_root)])
    assert result == 0
    lines = [ln for ln in stale.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_cli_repair_index_dry_run_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """repair-index --dry-run prints what would be written but leaves the file unchanged."""
    model_root = _populate_model_root_run_jsons(tmp_path, n_runs=2)
    result = main(["repair-index", "--model-root", str(model_root), "--dry-run"])
    assert result == 0
    assert "dry-run" in capsys.readouterr().out
    assert not (model_root / "runs.jsonl").exists()


def test_cli_repair_index_missing_model_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["repair-index", "--model-root", str(tmp_path / "no_such_dir")])
    assert result == 1
    assert "config error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# P3: --output-json
# ---------------------------------------------------------------------------


def test_cli_run_output_json_single_model(tmp_path: Path) -> None:
    """--output-json writes a single-model payload dict to the given file."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    out_json = tmp_path / "result.json"
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--no-progress",
        "--output-json", str(out_json),
    ])
    assert result == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["status"] == "PASS"
    assert "run_id" in payload


def test_cli_run_output_json_creates_parent_dirs(tmp_path: Path) -> None:
    """--output-json creates missing parent directories automatically."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    out_json = tmp_path / "reports" / "sub" / "result.json"
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--no-progress",
        "--output-json", str(out_json),
    ])
    assert out_json.exists()


def test_cli_run_output_json_multi_model_is_list(tmp_path: Path) -> None:
    """--output-json writes a JSON array when multiple models are run."""
    suite = {
        "version": 1,
        "project": "p",
        "suite_name": "s",
        "models": [
            {"name": "m1", "adapter": "mock", "params": {}},
            {"name": "m2", "adapter": "mock", "params": {}},
        ],
        "tests": [{"id": "t", "prompts": ["x"], "seeds": [0], "video": {"num_frames": 5}}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    import yaml

    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(suite), encoding="utf-8")
    out_json = tmp_path / "all.json"
    main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--no-progress",
        "--output-json", str(out_json),
    ])
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    assert all("status" in item for item in data)


# ---------------------------------------------------------------------------
# P4: trend arrows in status text
# ---------------------------------------------------------------------------


def _populate_model_root_with_two_runs(
    tmp_path: Path,
    *,
    prev_score: float,
    latest_score: float,
) -> Path:
    """Create model_root with two run.json files and runs.jsonl."""
    import datetime

    model_root = tmp_path / "model_trend"
    model_root.mkdir()
    for i, score in enumerate([prev_score, latest_score]):
        ts = datetime.datetime(2026, 5, i + 1)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "project": "p",
            "suite_name": "s",
            "model_name": "m",
            "status": "PASS",
            "sample_count": 1,
            "metrics": {"vbench_temporal": {"score": score}},
            "gates": [],
            "regressions": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_id": run_id,
                "timestamp_utc": ts.isoformat(),
                "status": "PASS",
                "sample_count": 1,
                "metrics": payload["metrics"],
            }) + "\n")
    return model_root


def test_cli_status_trend_arrow_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """↑ is shown when latest metric improved vs previous run."""
    model_root = _populate_model_root_with_two_runs(
        tmp_path, prev_score=0.70, latest_score=0.82
    )
    result = main(["status", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "↑" in out


def test_cli_status_trend_arrow_down(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """↓ is shown when latest metric declined vs previous run."""
    model_root = _populate_model_root_with_two_runs(
        tmp_path, prev_score=0.85, latest_score=0.72
    )
    result = main(["status", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "↓" in out


def test_cli_status_trend_arrow_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """→ is shown when latest metric is unchanged vs previous run."""
    model_root = _populate_model_root_with_two_runs(
        tmp_path, prev_score=0.75, latest_score=0.75
    )
    result = main(["status", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "→" in out


def test_cli_status_no_trend_arrows_for_first_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No trend arrows when there is only one run (no previous to compare)."""
    import datetime

    model_root = tmp_path / "model_single"
    model_root.mkdir()
    ts = datetime.datetime(2026, 5, 1)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "timestamp_utc": ts.isoformat(),
        "project": "p",
        "suite_name": "s",
        "model_name": "m",
        "status": "PASS",
        "sample_count": 1,
        "metrics": {"vbench_temporal": {"score": 0.80}},
        "gates": [],
        "regressions": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "status": "PASS",
            "sample_count": 1,
            "metrics": payload["metrics"],
        }) + "\n")

    result = main(["status", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "↑" not in out
    assert "↓" not in out
    assert "→" not in out


# ---------------------------------------------------------------------------
# P1: .temporalci.yaml project config file
# ---------------------------------------------------------------------------


def test_config_file_sets_workers_default(tmp_path: Path) -> None:
    """Workers value from .temporalci.yaml is forwarded to run_suite."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    cfg_file = tmp_path / ".temporalci.yaml"
    cfg_file.write_text("run:\n  workers: 3\n", encoding="utf-8")

    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "--config", str(cfg_file),
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["workers"] == 3


def test_config_file_cli_overrides_config(tmp_path: Path) -> None:
    """Explicit CLI flag overrides the config file default."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    cfg_file = tmp_path / ".temporalci.yaml"
    cfg_file.write_text("run:\n  workers: 3\n", encoding="utf-8")

    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "--config", str(cfg_file),
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
            "--workers", "7",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["workers"] == 7


def test_config_file_missing_uses_parser_defaults(tmp_path: Path) -> None:
    """When no config file exists, parser defaults (workers=1) are used."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)

    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "--config", str(tmp_path / "nonexistent.yaml"),
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["workers"] == 1


def test_config_file_multiple_keys(tmp_path: Path) -> None:
    """Multiple config keys (workers, retry, inter_sample_delay) are all applied."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    cfg_file = tmp_path / ".temporalci.yaml"
    cfg_file.write_text(
        "run:\n  workers: 2\n  retry: 3\n  inter_sample_delay: 0.5\n",
        encoding="utf-8",
    )

    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "--config", str(cfg_file),
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["workers"] == 2
    assert kwargs["retry"] == 3
    assert kwargs["inter_sample_delay"] == 0.5


# ---------------------------------------------------------------------------
# P3: temporalci alert
# ---------------------------------------------------------------------------


def _write_alert_state(model_root: Path, state: str, last_run_id: str = "run001") -> None:
    import json as _json

    (model_root / "alert_state.json").write_text(
        _json.dumps({
            "state": state,
            "last_run_id": last_run_id,
            "last_change_run_id": last_run_id,
        }),
        encoding="utf-8",
    )


def test_cli_alert_passing_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    _write_alert_state(model_root, "passing")
    result = main(["alert", "--model-root", str(model_root)])
    assert result == 0
    assert "ok" in capsys.readouterr().out


def test_cli_alert_failing_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    _write_alert_state(model_root, "failing", last_run_id="run_bad")
    result = main(["alert", "--model-root", str(model_root)])
    assert result == 1
    assert "FAILING" in capsys.readouterr().out


def test_cli_alert_no_state_file_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing alert_state.json → no previous runs → not failing."""
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = main(["alert", "--model-root", str(model_root)])
    assert result == 0
    assert "ok" in capsys.readouterr().out


def test_cli_alert_missing_model_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["alert", "--model-root", str(tmp_path / "nope")])
    assert result == 1
    assert "config error" in capsys.readouterr().out


def test_cli_alert_suite_root_any_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """alert --suite-root exits 1 if any model is failing."""
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    for name, state in [("m1", "passing"), ("m2", "failing"), ("m3", "passing")]:
        mr = suite_root / name
        mr.mkdir()
        _write_alert_state(mr, state)
        # write a minimal runs.jsonl so discover_models finds the model
        (mr / "runs.jsonl").write_text('{"run_id":"r1","status":"PASS","sample_count":1}\n', encoding="utf-8")
    result = main(["alert", "--suite-root", str(suite_root)])
    assert result == 1
    out = capsys.readouterr().out
    assert "FAILING" in out
    assert "m2" in out


def test_cli_alert_suite_root_all_passing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    for name in ["m1", "m2"]:
        mr = suite_root / name
        mr.mkdir()
        _write_alert_state(mr, "passing")
        (mr / "runs.jsonl").write_text('{"run_id":"r1","status":"PASS","sample_count":1}\n', encoding="utf-8")
    result = main(["alert", "--suite-root", str(suite_root)])
    assert result == 0


# ---------------------------------------------------------------------------
# P4: --fail-fast
# ---------------------------------------------------------------------------


def _write_named_model_suite(path: Path, model_names: list[str]) -> None:
    suite = {
        "version": 1,
        "project": "p",
        "suite_name": "s",
        "models": [{"name": n, "adapter": "mock", "params": {}} for n in model_names],
        "tests": [{"id": "t", "prompts": ["x"], "seeds": [0], "video": {"num_frames": 5}}],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0}],
    }
    import yaml

    path.write_text(yaml.dump(suite), encoding="utf-8")


def test_cli_run_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    """--fail-fast: only 1 run_suite call when first model fails."""
    suite_file = tmp_path / "suite.yaml"
    _write_named_model_suite(suite_file, ["m1", "m2", "m3"])

    call_count = 0

    def _failing_run_suite(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "status": "FAIL",
            "run_id": f"r{call_count}",
            "run_dir": str(tmp_path),
            "model_name": kwargs.get("model_name", "m1"),
            "sample_count": 1,
            "skipped_count": 0,
            "gate_failed": True,
            "regression_failed": False,
            "gates": [],
            "regressions": [],
        }

    with patch("temporalci.cli.run_suite", side_effect=_failing_run_suite):
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
            "--fail-fast",
        ])
    assert call_count == 1


def test_cli_run_fail_fast_false_runs_all_models(tmp_path: Path) -> None:
    """Without --fail-fast, all models run even when first fails."""
    suite_file = tmp_path / "suite.yaml"
    _write_named_model_suite(suite_file, ["m1", "m2", "m3"])

    call_count = 0

    def _failing_run_suite(**kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "status": "FAIL",
            "run_id": f"r{call_count}",
            "run_dir": str(tmp_path),
            "model_name": kwargs.get("model_name", "m1"),
            "sample_count": 1,
            "skipped_count": 0,
            "gate_failed": True,
            "regression_failed": False,
            "gates": [],
            "regressions": [],
        }

    with patch("temporalci.cli.run_suite", side_effect=_failing_run_suite):
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    assert call_count == 3


# ---------------------------------------------------------------------------
# P1: temporalci doctor
# ---------------------------------------------------------------------------


def test_cli_doctor_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """doctor always exits 0 (informational, not a gate)."""
    result = main(["doctor"])
    assert result == 0


def test_cli_doctor_output_has_key_sections(capsys: pytest.CaptureFixture[str]) -> None:
    """doctor output includes Python, adapters, and metrics sections."""
    main(["doctor"])
    out = capsys.readouterr().out
    assert "Python" in out
    assert "adapters" in out
    assert "metrics" in out


# ---------------------------------------------------------------------------
# P2: temporalci tag
# ---------------------------------------------------------------------------


def _make_tagged_model_root(tmp_path: Path, tags: dict) -> Path:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    (model_root / "tags.json").write_text(
        json.dumps(tags, indent=2), encoding="utf-8"
    )
    return model_root


def test_cli_tag_list_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = main(["tag", "list", "--model-root", str(model_root)])
    assert result == 0
    assert "no tags" in capsys.readouterr().out


def test_cli_tag_list_shows_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = _make_tagged_model_root(tmp_path, {"stable": "run001", "dev": "run002"})
    result = main(["tag", "list", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "stable" in out
    assert "run001" in out
    assert "dev" in out


def test_cli_tag_show_existing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = _make_tagged_model_root(tmp_path, {"stable": "run001"})
    result = main(["tag", "show", "--model-root", str(model_root), "--name", "stable"])
    assert result == 0
    assert "run001" in capsys.readouterr().out


def test_cli_tag_show_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_root = _make_tagged_model_root(tmp_path, {})
    result = main(["tag", "show", "--model-root", str(model_root), "--name", "ghost"])
    assert result == 1
    assert "not found" in capsys.readouterr().out


def test_cli_tag_set_and_read_back(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = main([
        "tag", "set",
        "--model-root", str(model_root),
        "--name", "v1",
        "--run-id", "runABC",
    ])
    assert result == 0
    tags = json.loads((model_root / "tags.json").read_text(encoding="utf-8"))
    assert tags["v1"] == "runABC"


def test_cli_tag_delete(tmp_path: Path) -> None:
    model_root = _make_tagged_model_root(tmp_path, {"old": "run000", "keep": "run001"})
    result = main([
        "tag", "delete",
        "--model-root", str(model_root),
        "--name", "old",
    ])
    assert result == 0
    tags = json.loads((model_root / "tags.json").read_text(encoding="utf-8"))
    assert "old" not in tags
    assert "keep" in tags


def test_cli_tag_missing_model_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["tag", "list", "--model-root", str(tmp_path / "nope")])
    assert result == 1
    assert "config error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# P3: --model-workers
# ---------------------------------------------------------------------------


def test_cli_run_model_workers_2_runs_all_models(tmp_path: Path) -> None:
    """--model-workers 2 completes all models via the parallel path."""
    suite_file = tmp_path / "suite.yaml"
    _write_named_model_suite(suite_file, ["m1", "m2", "m3"])

    called: list[str] = []

    def _rs(**kwargs):
        called.append(kwargs["model_name"])
        return {
            "status": "PASS",
            "run_id": "r1",
            "run_dir": str(tmp_path),
            "model_name": kwargs["model_name"],
            "sample_count": 1,
            "skipped_count": 0,
            "gate_failed": False,
            "regression_failed": False,
            "gates": [],
            "regressions": [],
        }

    with patch("temporalci.cli.run_suite", side_effect=_rs):
        rc = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
            "--model-workers", "2",
        ])
    assert set(called) == {"m1", "m2", "m3"}
    assert rc == 0


def test_cli_run_model_workers_sequential_default(tmp_path: Path) -> None:
    """Without --model-workers, models run sequentially (model_workers=1)."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    assert mock_rs.call_count == 1


# ---------------------------------------------------------------------------
# P4: --notify-on
# ---------------------------------------------------------------------------


def test_cli_run_notify_on_always_wired(tmp_path: Path) -> None:
    """--notify-on always is forwarded to run_suite as notify_on kwarg."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
            "--notify-on", "always",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["notify_on"] == "always"


def test_cli_run_notify_on_default_is_change(tmp_path: Path) -> None:
    """notify_on defaults to 'change' when --notify-on is omitted."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["notify_on"] == "change"


# ---------------------------------------------------------------------------
# P1: compare run-id mode
# ---------------------------------------------------------------------------


def _make_compare_run(model_root: Path, run_id: str, score: float) -> None:
    """Write a minimal run.json under model_root/run_id/."""
    run_dir = model_root / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "timestamp_utc": "2026-02-01T00:00:00",
        "status": "PASS",
        "project": "p",
        "suite_name": "s",
        "model_name": "m",
        "sample_count": 1,
        "gate_failed": False,
        "regression_failed": False,
        "metrics": {"vbench_temporal": {"score": score, "dims": {}}},
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.0,
                   "passed": True, "threshold_passed": True, "actual": score}],
        "regressions": [],
        "samples": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cli_compare_run_id_mode_zero_on_no_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """RUN_A RUN_B --model-root returns 0 when no metric regressions."""
    model_root = tmp_path / "model"
    _make_compare_run(model_root, "run_a", 0.8)
    _make_compare_run(model_root, "run_b", 0.85)  # improved
    out_html = tmp_path / "cmp.html"
    result = main([
        "compare", "run_a", "run_b",
        "--model-root", str(model_root),
        "--output", str(out_html),
    ])
    captured = capsys.readouterr().out
    assert "run_a" in captured
    assert "run_b" in captured
    assert result == 0


def test_cli_compare_run_id_mode_returns_one_on_regression(
    tmp_path: Path,
) -> None:
    """RUN_A RUN_B returns 1 when candidate has a lower metric (regression)."""
    model_root = tmp_path / "model"
    _make_compare_run(model_root, "run_a", 0.9)
    _make_compare_run(model_root, "run_b", 0.6)  # regression
    out_html = tmp_path / "cmp.html"
    result = main([
        "compare", "run_a", "run_b",
        "--model-root", str(model_root),
        "--output", str(out_html),
    ])
    assert result == 1


def test_cli_compare_run_id_missing_run_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run-ID mode returns 1 with informative message when run.json is missing."""
    model_root = tmp_path / "model"
    _make_compare_run(model_root, "run_a", 0.8)
    # run_b is NOT created
    out_html = tmp_path / "cmp.html"
    result = main([
        "compare", "run_a", "run_b",
        "--model-root", str(model_root),
        "--output", str(out_html),
    ])
    assert result == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_cli_compare_run_id_requires_model_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run-ID mode without --model-root returns 1 with clear error."""
    out_html = tmp_path / "cmp.html"
    result = main([
        "compare", "run_a", "run_b",
        "--output", str(out_html),
    ])
    assert result == 1
    assert "model-root" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# P2: history command
# ---------------------------------------------------------------------------


def _populate_history_model_root(model_root: Path, n: int = 4) -> None:
    """Create n run.json + runs.jsonl entries in model_root."""
    import datetime

    model_root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        status = "PASS" if i % 2 == 0 else "FAIL"
        payload = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "project": "proj",
            "suite_name": "suite",
            "model_name": "m",
            "status": status,
            "sample_count": 2,
            "gate_failed": status == "FAIL",
            "regression_failed": False,
            "metrics": {"vbench_temporal": {"score": 0.7 + i * 0.05}},
            "gates": [],
            "regressions": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run_id": run_id,
                "timestamp_utc": ts.isoformat(),
                "status": status,
                "sample_count": 2,
            }) + "\n")


def test_cli_history_shows_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """history command prints run table for a model_root."""
    model_root = tmp_path / "model"
    _populate_history_model_root(model_root, n=3)
    result = main(["history", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert "Showing 3 run(s)" in out
    assert "PASS" in out


def test_cli_history_filters_by_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--status PASS hides FAIL runs from output."""
    model_root = tmp_path / "model"
    _populate_history_model_root(model_root, n=4)  # 2 PASS, 2 FAIL
    result = main(["history", "--model-root", str(model_root), "--status", "PASS"])
    assert result == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "PASS" in out


def test_cli_history_filters_by_since(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--since filters out runs before the given date."""
    model_root = tmp_path / "model"
    _populate_history_model_root(model_root, n=4)  # runs on 2026-02-01 to 2026-02-04
    # Only runs on 2026-02-03 or later (2 runs)
    result = main([
        "history", "--model-root", str(model_root), "--since", "2026-02-03",
    ])
    assert result == 0
    out = capsys.readouterr().out
    assert "Showing 2 run(s)" in out


def test_cli_history_output_format_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--output-format json emits valid JSON list of runs."""
    model_root = tmp_path / "model"
    _populate_history_model_root(model_root, n=2)
    result = main([
        "history", "--model-root", str(model_root), "--output-format", "json",
    ])
    assert result == 0
    runs = json.loads(capsys.readouterr().out)
    assert isinstance(runs, list)
    assert len(runs) == 2
    assert all("run_id" in r for r in runs)


def test_cli_history_empty_model_root_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """history returns 1 when model_root has no runs."""
    model_root = tmp_path / "empty"
    model_root.mkdir()
    result = main(["history", "--model-root", str(model_root)])
    assert result == 1
    assert "no runs" in capsys.readouterr().out.lower()


def test_cli_history_last_n_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--last-n limits the number of runs shown."""
    model_root = tmp_path / "model"
    _populate_history_model_root(model_root, n=5)
    result = main([
        "history", "--model-root", str(model_root), "--last-n", "2",
    ])
    assert result == 0
    out = capsys.readouterr().out
    assert "Showing 2 run(s)" in out


# ---------------------------------------------------------------------------
# P4: CI mode auto-detection
# ---------------------------------------------------------------------------


def test_cli_ci_flag_suppresses_progress(tmp_path: Path) -> None:
    """--ci flag suppresses progress callback (use_progress=False)."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--ci",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["progress_callback"] is None


def test_cli_ci_env_auto_detected(tmp_path: Path) -> None:
    """CI=true environment variable auto-enables CI mode (suppresses progress)."""
    import os

    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        with patch.dict(os.environ, {"CI": "true"}, clear=False):
            main([
                "run", str(suite_file),
                "--artifacts-dir", str(tmp_path / "art"),
            ])
    _, kwargs = mock_rs.call_args
    assert kwargs["progress_callback"] is None


def test_cli_github_actions_annotations_on_gate_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """GITHUB_ACTIONS=true emits ::error:: annotations for failed gates."""
    import os

    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    failing_result = {
        **_mock_run_suite_result(tmp_path),
        "status": "FAIL",
        "gate_failed": True,
        "gates": [{
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.9,
            "actual": 0.5,
            "passed": False,
        }],
    }
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = failing_result
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=False):
            main([
                "run", str(suite_file),
                "--artifacts-dir", str(tmp_path / "art"),
            ])
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "vbench_temporal.score" in out


# ---------------------------------------------------------------------------
# P1: --watch mode
# ---------------------------------------------------------------------------


def test_cli_watch_runs_multiple_iterations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--watch N runs suite repeatedly; KeyboardInterrupt stops the loop gracefully."""

    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)

    call_count = 0

    def _fake_run_suite(**kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt  # stop after 2nd call
        return _mock_run_suite_result(tmp_path)

    with patch("temporalci.cli.run_suite", side_effect=_fake_run_suite):
        with patch("time.sleep"):  # don't actually sleep
            result = main([
                "run", str(suite_file),
                "--artifacts-dir", str(tmp_path / "art"),
                "--watch", "5",
                "--no-progress",
            ])

    assert result == 0  # KeyboardInterrupt → graceful exit 0
    assert call_count >= 1
    out = capsys.readouterr().out
    assert "watch" in out.lower()


def test_cli_watch_stopped_message_on_interrupt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--watch prints 'watch: stopped.' on KeyboardInterrupt."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)

    def _fake(*args: object, **kwargs: object) -> dict:
        raise KeyboardInterrupt

    with patch("temporalci.cli.run_suite", side_effect=_fake):
        with patch("time.sleep"):
            result = main([
                "run", str(suite_file),
                "--artifacts-dir", str(tmp_path / "art"),
                "--watch", "10",
                "--no-progress",
            ])

    assert result == 0
    assert "stopped" in capsys.readouterr().out.lower()


def test_cli_watch_not_active_without_flag(tmp_path: Path) -> None:
    """Without --watch, run executes exactly once and returns normally."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    call_count = 0

    def _fake(**kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return _mock_run_suite_result(tmp_path)

    with patch("temporalci.cli.run_suite", side_effect=_fake):
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])

    assert call_count == 1


# ---------------------------------------------------------------------------
# P2: tune-gates command
# ---------------------------------------------------------------------------


def _populate_model_root_for_tuning(model_root: Path, n_pass: int = 5) -> None:
    """Write n_pass PASS run.json files with vbench_temporal metrics."""
    import datetime

    model_root.mkdir(parents=True, exist_ok=True)
    for i in range(n_pass):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "project": "p",
            "suite_name": "s",
            "model_name": "m",
            "status": "PASS",
            "sample_count": 2,
            "gate_failed": False,
            "regression_failed": False,
            "metrics": {
                "vbench_temporal": {
                    "score": 0.70 + i * 0.01,
                    "dims": {"motion_smoothness": 0.75 + i * 0.01},
                }
            },
            "gates": [],
            "regressions": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run_id": run_id,
                "status": "PASS",
                "timestamp_utc": ts.isoformat(),
                "sample_count": 2,
            }) + "\n")


def test_cli_tune_gates_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """tune-gates exits 0 when PASS runs are found."""
    model_root = tmp_path / "model"
    _populate_model_root_for_tuning(model_root, n_pass=5)
    result = main(["tune-gates", "--model-root", str(model_root)])
    assert result == 0


def test_cli_tune_gates_output_has_metric_suggestions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """tune-gates output contains metric paths and operators."""
    model_root = tmp_path / "model"
    _populate_model_root_for_tuning(model_root, n_pass=5)
    main(["tune-gates", "--model-root", str(model_root)])
    out = capsys.readouterr().out
    assert "vbench_temporal" in out
    assert ">=" in out
    assert "value:" in out


def test_cli_tune_gates_missing_model_root_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """tune-gates returns 1 when model-root does not exist."""
    result = main(["tune-gates", "--model-root", str(tmp_path / "nope")])
    assert result == 1


def test_cli_tune_gates_no_pass_runs_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """tune-gates returns 1 when there are no PASS runs."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir(parents=True)
    ts = datetime.datetime(2026, 2, 1)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id, "status": "FAIL", "timestamp_utc": ts.isoformat(),
        "sample_count": 1, "gate_failed": True, "regression_failed": False,
        "metrics": {}, "gates": [], "regressions": [], "samples": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": run_id, "status": "FAIL",
                              "timestamp_utc": ts.isoformat(), "sample_count": 1}) + "\n")
    result = main(["tune-gates", "--model-root", str(model_root)])
    assert result == 1


def test_cli_tune_gates_metric_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--metric filters output to only the specified metric prefix."""
    model_root = tmp_path / "model"
    _populate_model_root_for_tuning(model_root, n_pass=3)
    main(["tune-gates", "--model-root", str(model_root), "--metric", "vbench_temporal.score"])
    out = capsys.readouterr().out
    assert "vbench_temporal.score" in out


# ---------------------------------------------------------------------------
# P3: --env flag
# ---------------------------------------------------------------------------


def test_cli_run_env_wired(tmp_path: Path) -> None:
    """--env NAME is forwarded to run_suite as env kwarg."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--env", "prod",
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["env"] == "prod"


def test_cli_run_env_default_none(tmp_path: Path) -> None:
    """env defaults to None when --env is omitted."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs.get("env") is None


# ---------------------------------------------------------------------------
# P4: summary command
# ---------------------------------------------------------------------------


def _populate_artifacts_dir(tmp_path: Path) -> Path:
    """Create artifacts/proj/suite/model/run structure."""
    import datetime

    artifacts = tmp_path / "artifacts"
    for proj in ("proj_a", "proj_b"):
        for suite in ("suite1",):
            for model in ("model_x", "model_y"):
                model_root = artifacts / proj / suite / model
                model_root.mkdir(parents=True)
                ts = datetime.datetime(2026, 2, 1)
                run_id = ts.strftime("%Y%m%dT%H%M%SZ")
                run_dir = model_root / run_id
                run_dir.mkdir()
                status = "PASS" if model == "model_x" else "FAIL"
                payload = {
                    "run_id": run_id, "timestamp_utc": ts.isoformat(),
                    "project": proj, "suite_name": suite, "model_name": model,
                    "status": status, "sample_count": 1,
                    "gate_failed": status == "FAIL", "regression_failed": False,
                    "metrics": {}, "gates": [], "regressions": [], "samples": [],
                }
                (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
                with (model_root / "runs.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "run_id": run_id, "status": status,
                        "timestamp_utc": ts.isoformat(), "sample_count": 1,
                    }) + "\n")
    return artifacts


def test_cli_summary_shows_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """summary lists all models from all projects/suites."""
    artifacts = _populate_artifacts_dir(tmp_path)
    result = main(["summary", "--artifacts-dir", str(artifacts)])
    # Some models fail → exit 1
    assert result in (0, 1)
    out = capsys.readouterr().out
    assert "proj_a" in out
    assert "model_x" in out
    assert "model_y" in out


def test_cli_summary_exit_one_when_any_fail(tmp_path: Path) -> None:
    """summary returns 1 when any model is not PASS."""
    artifacts = _populate_artifacts_dir(tmp_path)
    # model_y is FAIL
    result = main(["summary", "--artifacts-dir", str(artifacts)])
    assert result == 1


def test_cli_summary_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--output-format json emits valid JSON tree."""
    artifacts = _populate_artifacts_dir(tmp_path)
    main(["summary", "--artifacts-dir", str(artifacts), "--output-format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, dict)
    assert "proj_a" in data


def test_cli_summary_empty_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """summary returns 1 when artifacts-dir has no runs."""
    empty = tmp_path / "empty_arts"
    empty.mkdir()
    result = main(["summary", "--artifacts-dir", str(empty)])
    assert result == 1
    assert "no runs" in capsys.readouterr().out.lower()


def test_cli_summary_missing_dir_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """summary returns 1 when artifacts-dir does not exist."""
    result = main(["summary", "--artifacts-dir", str(tmp_path / "nope")])
    assert result == 1


# ---------------------------------------------------------------------------
# Helpers shared by P1-P4 CLI tests
# ---------------------------------------------------------------------------


def _write_two_model_suite(path: Path) -> None:
    """Write a minimal suite YAML with two models: 'alpha' and 'beta'."""
    suite = {
        "version": 1,
        "project": "test-proj",
        "suite_name": "test-suite",
        "models": [
            {"name": "alpha", "adapter": "mock", "params": {}},
            {"name": "beta", "adapter": "mock", "params": {}},
        ],
        "tests": [
            {
                "id": "t1",
                "prompts": ["a safe prompt"],
                "seeds": [0],
                "video": {"num_frames": 5},
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.5}],
    }
    path.write_text(yaml.dump(suite), encoding="utf-8")


def _make_mock_run_result(tmp_path: Path, model_name: str = "mock1") -> dict:
    return {
        "status": "PASS",
        "run_id": "testid",
        "run_dir": str(tmp_path),
        "model_name": model_name,
        "sample_count": 1,
        "skipped_count": 0,
        "gate_failed": False,
        "regression_failed": False,
        "gates": [],
        "regressions": [],
    }


# ---------------------------------------------------------------------------
# P1: --adapter-timeout wiring
# ---------------------------------------------------------------------------


def test_cli_run_adapter_timeout_wired(tmp_path: Path) -> None:
    """--adapter-timeout is forwarded to run_suite as adapter_timeout kwarg."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--adapter-timeout", "5.0",
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["adapter_timeout"] == 5.0


def test_cli_run_adapter_timeout_default_none(tmp_path: Path) -> None:
    """adapter_timeout defaults to None when --adapter-timeout is omitted."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    with patch("temporalci.cli.run_suite") as mock_rs:
        mock_rs.return_value = _mock_run_suite_result(tmp_path)
        main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--no-progress",
        ])
    _, kwargs = mock_rs.call_args
    assert kwargs["adapter_timeout"] is None


# ---------------------------------------------------------------------------
# P2: --include-model / --exclude-model
# ---------------------------------------------------------------------------


def test_cli_run_include_model_filters(tmp_path: Path) -> None:
    """--include-model restricts execution to the specified model."""
    suite_file = tmp_path / "suite.yaml"
    _write_two_model_suite(suite_file)
    called_with: list[str] = []

    def _fake_run_suite(**kwargs: Any) -> dict:
        called_with.append(kwargs["model_name"])
        return _make_mock_run_result(tmp_path, kwargs["model_name"])

    with patch("temporalci.cli.run_suite", side_effect=_fake_run_suite):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--include-model", "alpha",
            "--no-progress",
        ])
    assert result == 0
    assert called_with == ["alpha"]


def test_cli_run_exclude_model_skips(tmp_path: Path) -> None:
    """--exclude-model prevents the specified model from running."""
    suite_file = tmp_path / "suite.yaml"
    _write_two_model_suite(suite_file)
    called_with: list[str] = []

    def _fake_run_suite(**kwargs: Any) -> dict:
        called_with.append(kwargs["model_name"])
        return _make_mock_run_result(tmp_path, kwargs["model_name"])

    with patch("temporalci.cli.run_suite", side_effect=_fake_run_suite):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--exclude-model", "beta",
            "--no-progress",
        ])
    assert result == 0
    assert "beta" not in called_with
    assert "alpha" in called_with


def test_cli_run_no_model_match_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All models excluded → exits 1 with config error message."""
    suite_file = tmp_path / "suite.yaml"
    _write_two_model_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--include-model", "gamma",  # not in suite
        "--no-progress",
    ])
    assert result == 1
    out = capsys.readouterr().out
    assert "no models match" in out.lower()


# ---------------------------------------------------------------------------
# P3: metrics-show command
# ---------------------------------------------------------------------------


def _populate_model_root_for_metrics(model_root: Path) -> str:
    """Create a single run.json in model_root; return the run_id."""
    import datetime

    model_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime(2026, 2, 10, 12, 0, 0)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    run_dir = model_root / run_id
    run_dir.mkdir()
    payload = {
        "run_id": run_id,
        "timestamp_utc": ts.isoformat(),
        "status": "PASS",
        "model_name": "mock1",
        "sample_count": 2,
        "skipped_count": 0,
        "gate_failed": False,
        "regression_failed": False,
        "metrics": {"vbench_temporal": {"score": 0.75}},
        "gates": [
            {
                "metric": "vbench_temporal.score",
                "op": ">=",
                "value": 0.5,
                "actual": 0.75,
                "passed": True,
            }
        ],
        "regressions": [],
        "samples": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "run_id": run_id,
            "status": "PASS",
            "timestamp_utc": ts.isoformat(),
            "sample_count": 2,
        }) + "\n")
    return run_id


def test_cli_metrics_show_latest_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metrics-show displays run info for the latest run when no --run-id given."""
    model_root = tmp_path / "model"
    run_id = _populate_model_root_for_metrics(model_root)
    result = main(["metrics-show", "--model-root", str(model_root)])
    assert result == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "PASS" in out
    assert "vbench_temporal.score" in out


def test_cli_metrics_show_specific_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metrics-show --run-id shows the specified run."""
    model_root = tmp_path / "model"
    run_id = _populate_model_root_for_metrics(model_root)
    result = main(["metrics-show", "--model-root", str(model_root), "--run-id", run_id])
    assert result == 0
    out = capsys.readouterr().out
    assert run_id in out


def test_cli_metrics_show_missing_model_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metrics-show returns 1 when model-root does not exist."""
    result = main(["metrics-show", "--model-root", str(tmp_path / "nope")])
    assert result == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_cli_metrics_show_missing_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metrics-show returns 1 when the specified run-id has no run.json."""
    model_root = tmp_path / "model"
    _populate_model_root_for_metrics(model_root)
    result = main(["metrics-show", "--model-root", str(model_root), "--run-id", "20990101T000000Z"])
    assert result == 1
    out = capsys.readouterr().out
    assert "not found" in out.lower()


def test_cli_metrics_show_no_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metrics-show returns 1 when there are no runs in model-root."""
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = main(["metrics-show", "--model-root", str(model_root)])
    assert result == 1
    out = capsys.readouterr().out
    assert "no runs" in out.lower()


# ---------------------------------------------------------------------------
# P4: --gate-override
# ---------------------------------------------------------------------------


def test_cli_gate_override_replaces_value(tmp_path: Path) -> None:
    """--gate-override patches an existing gate's value before run_suite is called."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    received_suite: list[Any] = []

    def _capture(**kwargs: Any) -> dict:
        received_suite.append(kwargs["suite"])
        return _mock_run_suite_result(tmp_path)

    with patch("temporalci.cli.run_suite", side_effect=_capture):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--gate-override", "vbench_temporal.score >= 0.99",
            "--no-progress",
        ])
    assert result == 0
    called_suite = received_suite[0]
    gate = next(g for g in called_suite.gates if g.metric == "vbench_temporal.score" and g.op == ">=")
    assert gate.value == pytest.approx(0.99)


def test_cli_gate_override_adds_new_gate(tmp_path: Path) -> None:
    """--gate-override with a non-matching spec appends a new gate."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    received_suite: list[Any] = []

    def _capture(**kwargs: Any) -> dict:
        received_suite.append(kwargs["suite"])
        return _mock_run_suite_result(tmp_path)

    with patch("temporalci.cli.run_suite", side_effect=_capture):
        result = main([
            "run", str(suite_file),
            "--artifacts-dir", str(tmp_path / "art"),
            "--gate-override", "new_metric.score >= 0.80",
            "--no-progress",
        ])
    assert result == 0
    called_suite = received_suite[0]
    metrics = [g.metric for g in called_suite.gates]
    assert "new_metric.score" in metrics


def test_cli_gate_override_invalid_spec_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed --gate-override value causes exit 1 with config error."""
    suite_file = tmp_path / "suite.yaml"
    _write_minimal_suite(suite_file)
    result = main([
        "run", str(suite_file),
        "--artifacts-dir", str(tmp_path / "art"),
        "--gate-override", "bad-spec",  # missing OP and VALUE
        "--no-progress",
    ])
    assert result == 1
    out = capsys.readouterr().out
    assert "config error" in out.lower()
