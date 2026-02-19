from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from temporalci.export import export_runs, export_suite_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_root(tmp_path: Path, n_runs: int = 3) -> Path:
    """Create a model_root with n run.json files and runs.jsonl."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir()
    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        payload = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(),
            "status": "PASS" if i % 2 == 0 else "FAIL",
            "sample_count": 3,
            "gate_failed": i % 2 != 0,
            "regression_failed": False,
            "baseline_run_id": None,
            "metrics": {
                "vbench_temporal": {
                    "score": 0.7 + i * 0.05,
                    "dims": {
                        "motion_smoothness": 0.75 + i * 0.02,
                        "temporal_flicker": 0.65 + i * 0.03,
                    },
                }
            },
            "gates": [],
            "regressions": [],
            "samples": [],
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "timestamp_utc": ts.isoformat(),
                        "status": payload["status"],
                        "sample_count": 3,
                    }
                )
                + "\n"
            )
    return model_root


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_export_csv_creates_file(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path)
    out = tmp_path / "metrics.csv"
    n = export_runs(model_root, out)
    assert out.exists()
    assert n == 3


def test_export_csv_has_correct_row_count(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=5)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5


def test_export_csv_base_columns_present(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
    assert "run_id" in fieldnames
    assert "timestamp_utc" in fieldnames
    assert "status" in fieldnames
    assert "sample_count" in fieldnames


def test_export_csv_metric_columns_present(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
    assert "vbench_temporal.score" in fieldnames
    assert "vbench_temporal.dims.motion_smoothness" in fieldnames


def test_export_csv_metric_values_correct(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=1)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    score = float(rows[0]["vbench_temporal.score"])
    assert abs(score - 0.7) < 1e-9


def test_export_csv_last_n_limits_rows(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=5)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out, last_n=2)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


def test_export_empty_model_root_returns_zero(tmp_path: Path) -> None:
    model_root = tmp_path / "empty"
    model_root.mkdir()
    (model_root / "runs.jsonl").write_text("", encoding="utf-8")
    out = tmp_path / "out.csv"
    n = export_runs(model_root, out)
    assert n == 0
    assert not out.exists()


def test_export_missing_model_root_returns_zero(tmp_path: Path) -> None:
    model_root = tmp_path / "no_model"
    model_root.mkdir()
    out = tmp_path / "out.csv"
    n = export_runs(model_root, out)
    assert n == 0


def test_export_status_values_correct(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=2)
    out = tmp_path / "metrics.csv"
    export_runs(model_root, out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    statuses = [r["status"] for r in rows]
    assert "PASS" in statuses
    assert "FAIL" in statuses


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------


def test_export_jsonl_creates_file(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path)
    out = tmp_path / "runs.jsonl"
    n = export_runs(model_root, out, fmt="jsonl")
    assert out.exists()
    assert n == 3


def test_export_jsonl_each_line_valid_json(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=3)
    out = tmp_path / "runs.jsonl"
    export_runs(model_root, out, fmt="jsonl")
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "run_id" in obj


def test_export_jsonl_excludes_samples_key(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=2)
    out = tmp_path / "runs.jsonl"
    export_runs(model_root, out, fmt="jsonl")
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            assert "samples" not in obj


def test_export_invalid_format_raises(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=1)
    out = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="unsupported export format"):
        export_runs(model_root, out, fmt="xyz")


def test_export_creates_parent_dirs(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path)
    out = tmp_path / "reports" / "subdir" / "metrics.csv"
    export_runs(model_root, out)
    assert out.exists()


def test_export_fallback_to_index_entry_when_run_json_missing(tmp_path: Path) -> None:
    """Runs without run.json (e.g. after aggressive prune) use index entry data."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir()
    ts = datetime.datetime(2026, 2, 1)
    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    # Write ONLY the jsonl entry — no run.json
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "status": "PASS", "sample_count": 0}) + "\n")

    out = tmp_path / "out.csv"
    n = export_runs(model_root, out)
    assert n == 1
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "PASS"


# ---------------------------------------------------------------------------
# Suite-level export (P4)
# ---------------------------------------------------------------------------


def _make_suite_root(tmp_path: Path, model_names: list[str], n_runs: int = 2) -> Path:
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    for model_name in model_names:
        _make_model_root(suite_root / model_name, n_runs=n_runs)
        # But _make_model_root puts its model dir inside suite_root as "model"
        # Let's use a simpler inline approach
    return suite_root


def _populate_suite_root(tmp_path: Path, model_names: list[str], n_runs: int = 2) -> Path:
    """Create suite_root/model_name/... for each model."""
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
                "timestamp_utc": ts.isoformat(),
                "status": "PASS",
                "sample_count": 1,
                "model_name": model_name,
                "gate_failed": False,
                "regression_failed": False,
                "baseline_run_id": None,
                "metrics": {"vbench_temporal": {"score": 0.8 + i * 0.01}},
                "gates": [],
                "regressions": [],
                "samples": [],
            }
            (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
            with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "PASS",
                            "timestamp_utc": ts.isoformat(),
                            "sample_count": 1,
                        }
                    )
                    + "\n"
                )
    return suite_root


def test_export_suite_creates_file(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["model_a", "model_b"], n_runs=2)
    out = tmp_path / "all.csv"
    n = export_suite_runs(suite_root, out)
    assert out.exists()
    assert n == 4  # 2 models × 2 runs


def test_export_suite_csv_has_model_name_column(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["model_a", "model_b"])
    out = tmp_path / "all.csv"
    export_suite_runs(suite_root, out)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    assert "model_name" in fieldnames
    model_names_in_csv = {r["model_name"] for r in rows}
    assert "model_a" in model_names_in_csv
    assert "model_b" in model_names_in_csv


def test_export_suite_csv_row_count(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["m1", "m2", "m3"], n_runs=3)
    out = tmp_path / "all.csv"
    n = export_suite_runs(suite_root, out)
    assert n == 9  # 3 models × 3 runs
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9


def test_export_suite_jsonl(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["model_a"], n_runs=2)
    out = tmp_path / "all.jsonl"
    n = export_suite_runs(suite_root, out, fmt="jsonl")
    assert n == 2
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert "model_name" in obj


def test_export_suite_last_n(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["model_a", "model_b"], n_runs=5)
    out = tmp_path / "all.csv"
    n = export_suite_runs(suite_root, out, last_n=2)
    assert n == 4  # 2 models × last 2 runs each


def test_export_suite_empty_returns_zero(tmp_path: Path) -> None:
    suite_root = tmp_path / "empty_suite"
    suite_root.mkdir()
    out = tmp_path / "all.csv"
    n = export_suite_runs(suite_root, out)
    assert n == 0
    assert not out.exists()


def test_export_suite_metric_columns_present(tmp_path: Path) -> None:
    suite_root = _populate_suite_root(tmp_path, ["model_a"])
    out = tmp_path / "all.csv"
    export_suite_runs(suite_root, out)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
    assert "vbench_temporal.score" in fieldnames
