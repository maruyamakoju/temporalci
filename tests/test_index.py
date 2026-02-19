from __future__ import annotations

import json
from pathlib import Path

from temporalci.index import discover_models, write_suite_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    suite_root: Path,
    model_name: str,
    *,
    n_runs: int = 2,
    status: str = "PASS",
    score: float = 0.75,
) -> Path:
    """Create a minimal model artifact directory under suite_root."""
    model_root = suite_root / model_name
    model_root.mkdir(parents=True)
    import datetime

    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        run_status = status if i == n_runs - 1 else "PASS"
        payload = {
            "run_id": run_id,
            "status": run_status,
            "timestamp_utc": ts.isoformat(),
            "project": "proj",
            "suite_name": "suite",
            "model_name": model_name,
            "metrics": {"vbench_temporal": {"score": score}},
            "gates": [],
            "samples": [],
            "sample_count": 3,
        }
        (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "run_id": run_id,
                    "status": run_status,
                    "timestamp_utc": ts.isoformat(),
                    "sample_count": 3,
                }) + "\n"
            )
    return model_root


# ---------------------------------------------------------------------------
# discover_models
# ---------------------------------------------------------------------------


def test_discover_models_empty(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    assert discover_models(suite_root) == []


def test_discover_models_missing_dir(tmp_path: Path) -> None:
    assert discover_models(tmp_path / "nonexistent") == []


def test_discover_models_finds_models(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "model-a")
    _make_model(suite_root, "model-b")
    found = discover_models(suite_root)
    names = [name for name, _ in found]
    assert "model-a" in names
    assert "model-b" in names


def test_discover_models_skips_dirs_without_jsonl(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    (suite_root / "no-jsonl-dir").mkdir()
    found = discover_models(suite_root)
    assert found == []


# ---------------------------------------------------------------------------
# write_suite_index
# ---------------------------------------------------------------------------


def test_write_suite_index_creates_file(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "mock-model")
    write_suite_index(suite_root, project="p", suite_name="s")
    index = suite_root / "index.html"
    assert index.exists()
    assert "<!doctype html>" in index.read_text(encoding="utf-8")


def test_write_suite_index_shows_model_name(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "my-special-model")
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "my-special-model" in content


def test_write_suite_index_shows_project_suite(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "m")
    write_suite_index(suite_root, project="my-project", suite_name="core-suite")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "my-project" in content
    assert "core-suite" in content


def test_write_suite_index_shows_pass_badge(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "m", status="PASS")
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "badge-pass" in content


def test_write_suite_index_shows_fail_badge(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "m", status="FAIL")
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "badge-fail" in content


def test_write_suite_index_no_models(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "no-models" in content or "No model" in content


def test_write_suite_index_multiple_models(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "model-a", score=0.8)
    _make_model(suite_root, "model-b", score=0.6)
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "model-a" in content
    assert "model-b" in content


def test_write_suite_index_escapes_xss(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    write_suite_index(suite_root, project='<script>alert("x")</script>', suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_write_suite_index_links_trend_report(tmp_path: Path) -> None:
    """When trend_report.html exists in model_root, a link appears in index."""
    suite_root = tmp_path / "suite"
    model_root = _make_model(suite_root, "m")
    # Create a stub trend report
    (model_root / "trend_report.html").write_text("stub", encoding="utf-8")
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "trend_report.html" in content or "trend report" in content


def test_write_suite_index_shows_metric_values(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    _make_model(suite_root, "m", score=0.8765)
    write_suite_index(suite_root, project="p", suite_name="s")
    content = (suite_root / "index.html").read_text(encoding="utf-8")
    assert "vbench_temporal" in content
    assert "0.8765" in content


def test_write_suite_index_creates_parent_dirs(tmp_path: Path) -> None:
    suite_root = tmp_path / "nested" / "project" / "suite"
    write_suite_index(suite_root, project="p", suite_name="s")
    assert (suite_root / "index.html").exists()
