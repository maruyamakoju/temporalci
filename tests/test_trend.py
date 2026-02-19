from __future__ import annotations

import json
from pathlib import Path

from temporalci.trend import (
    _discover_metric_paths,
    _extract_metric_series,
    load_model_runs,
    write_trend_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    tmp_path: Path,
    run_id: str,
    status: str,
    metrics: dict,
    *,
    write_jsonl: bool = True,
) -> Path:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    run_dir = model_root / run_id
    run_dir.mkdir(exist_ok=True)
    payload = {
        "run_id": run_id,
        "status": status,
        "timestamp_utc": f"2026-02-{run_id[-2:]}T00:00:00+00:00",
        "project": "proj",
        "suite_name": "suite",
        "model_name": "mock",
        "sample_count": 4,
        "metrics": metrics,
        "gates": [],
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    if write_jsonl:
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "status": status, "timestamp_utc": payload["timestamp_utc"]}) + "\n")
    return model_root


# ---------------------------------------------------------------------------
# load_model_runs
# ---------------------------------------------------------------------------


def test_load_model_runs_returns_empty_when_no_jsonl(tmp_path: Path) -> None:
    model_root = tmp_path / "empty_model"
    model_root.mkdir()
    assert load_model_runs(model_root) == []


def test_load_model_runs_loads_runs(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {"vbench_temporal": {"score": 0.8}})
    _make_run(tmp_path, "20260202", "FAIL", {"vbench_temporal": {"score": 0.5}})
    runs = load_model_runs(model_root)
    assert len(runs) == 2
    assert runs[0]["run_id"] == "20260201"
    assert runs[1]["status"] == "FAIL"


def test_load_model_runs_respects_last_n(tmp_path: Path) -> None:
    for i in range(1, 6):
        _make_run(tmp_path, f"2026020{i}", "PASS", {})
    model_root = tmp_path / "model"
    runs = load_model_runs(model_root, last_n=3)
    assert len(runs) == 3
    assert runs[-1]["run_id"] == "20260205"


def test_load_model_runs_stub_when_run_json_missing(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    with (model_root / "runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "ghost_run", "status": "PASS"}) + "\n")
    # No run directory created
    runs = load_model_runs(model_root)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "ghost_run"


# ---------------------------------------------------------------------------
# _extract_metric_series
# ---------------------------------------------------------------------------


def test_extract_metric_series_simple(tmp_path: Path) -> None:
    runs = [
        {"metrics": {"vbench_temporal": {"score": 0.8}}},
        {"metrics": {"vbench_temporal": {"score": 0.6}}},
    ]
    values = _extract_metric_series(runs, "vbench_temporal.score")
    assert values == [0.8, 0.6]


def test_extract_metric_series_missing_becomes_none(tmp_path: Path) -> None:
    runs = [
        {"metrics": {"vbench_temporal": {"score": 0.8}}},
        {"metrics": {}},
    ]
    values = _extract_metric_series(runs, "vbench_temporal.score")
    assert values[0] == 0.8
    assert values[1] is None


def test_extract_metric_series_nested_dims(tmp_path: Path) -> None:
    runs = [
        {"metrics": {"vbench_temporal": {"dims": {"motion_smoothness": 0.9}}}},
    ]
    values = _extract_metric_series(runs, "vbench_temporal.dims.motion_smoothness")
    assert values == [0.9]


# ---------------------------------------------------------------------------
# _discover_metric_paths
# ---------------------------------------------------------------------------


def test_discover_metric_paths(tmp_path: Path) -> None:
    runs = [
        {"metrics": {"vbench_temporal": {"score": 0.8, "dims": {"a": 0.7}}}},
        {"metrics": {"vbench_temporal": {"score": 0.75}}},
    ]
    paths = _discover_metric_paths(runs)
    assert "vbench_temporal.score" in paths
    assert "vbench_temporal.dims.a" in paths


# ---------------------------------------------------------------------------
# write_trend_report
# ---------------------------------------------------------------------------


def test_write_trend_report_creates_file(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {"vbench_temporal": {"score": 0.8}})
    _make_run(tmp_path, "20260202", "FAIL", {"vbench_temporal": {"score": 0.5}})
    runs = load_model_runs(model_root)
    out = tmp_path / "trend.html"
    write_trend_report(out, runs)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


def test_write_trend_report_contains_run_ids(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {"vbench_temporal": {"score": 0.8}})
    _make_run(tmp_path, "20260202", "FAIL", {"vbench_temporal": {"score": 0.5}})
    runs = load_model_runs(model_root)
    out = tmp_path / "trend.html"
    write_trend_report(out, runs)
    content = out.read_text(encoding="utf-8")
    assert "20260201" in content
    assert "20260202" in content


def test_write_trend_report_shows_metric_chart(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {"vbench_temporal": {"score": 0.8}})
    _make_run(tmp_path, "20260202", "PASS", {"vbench_temporal": {"score": 0.85}})
    _make_run(tmp_path, "20260203", "FAIL", {"vbench_temporal": {"score": 0.6}})
    runs = load_model_runs(model_root)
    out = tmp_path / "trend.html"
    write_trend_report(out, runs)
    content = out.read_text(encoding="utf-8")
    assert "vbench_temporal.score" in content
    assert "<svg" in content
    assert "polyline" in content


def test_write_trend_report_empty_runs(tmp_path: Path) -> None:
    out = tmp_path / "trend.html"
    write_trend_report(out, [])
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "No runs" in content


def test_write_trend_report_custom_title(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {})
    runs = load_model_runs(model_root)
    out = tmp_path / "trend.html"
    write_trend_report(out, runs, title="My Custom Title")
    content = out.read_text(encoding="utf-8")
    assert "My Custom Title" in content


def test_write_trend_report_escapes_xss(tmp_path: Path) -> None:
    model_root = _make_run(tmp_path, "20260201", "PASS", {})
    runs = load_model_runs(model_root)
    out = tmp_path / "trend.html"
    write_trend_report(out, runs, title='<script>alert("xss")</script>')
    content = out.read_text(encoding="utf-8")
    assert "<script>" not in content
