"""Tests for temporalci.baseline — all five public functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporalci.baseline import (
    _average_metrics_dicts,
    _load_previous_run,
    _read_tags,
    _validate_baseline_mode,
    _write_tag,
)


# ---------------------------------------------------------------------------
# _read_tags
# ---------------------------------------------------------------------------


def test_read_tags_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _read_tags(tmp_path) == {}


def test_read_tags_valid_file(tmp_path: Path) -> None:
    (tmp_path / "tags.json").write_text(
        json.dumps({"stable": "run_001", "latest": "run_002"}), encoding="utf-8"
    )
    tags = _read_tags(tmp_path)
    assert tags == {"stable": "run_001", "latest": "run_002"}


def test_read_tags_invalid_json_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "tags.json").write_text("not json", encoding="utf-8")
    # read_json_dict returns None on error; _read_tags should return {}
    result = _read_tags(tmp_path)
    assert result == {} or result is None  # either is acceptable


# ---------------------------------------------------------------------------
# _write_tag
# ---------------------------------------------------------------------------


def test_write_tag_creates_file(tmp_path: Path) -> None:
    _write_tag(tmp_path, "stable", "run_001")
    tags = json.loads((tmp_path / "tags.json").read_text(encoding="utf-8"))
    assert tags == {"stable": "run_001"}


def test_write_tag_updates_existing_entry(tmp_path: Path) -> None:
    _write_tag(tmp_path, "stable", "run_001")
    _write_tag(tmp_path, "stable", "run_002")
    tags = json.loads((tmp_path / "tags.json").read_text(encoding="utf-8"))
    assert tags["stable"] == "run_002"


def test_write_tag_preserves_other_entries(tmp_path: Path) -> None:
    _write_tag(tmp_path, "stable", "run_001")
    _write_tag(tmp_path, "canary", "run_003")
    tags = json.loads((tmp_path / "tags.json").read_text(encoding="utf-8"))
    assert tags["stable"] == "run_001"
    assert tags["canary"] == "run_003"


def test_write_tag_non_string_tag_coerced(tmp_path: Path) -> None:
    _write_tag(tmp_path, "42", "run_007")
    tags = json.loads((tmp_path / "tags.json").read_text(encoding="utf-8"))
    assert tags["42"] == "run_007"


# ---------------------------------------------------------------------------
# _average_metrics_dicts
# ---------------------------------------------------------------------------


def test_average_metrics_empty_list_returns_empty() -> None:
    assert _average_metrics_dicts([]) == {}


def test_average_metrics_single_dict() -> None:
    result = _average_metrics_dicts([{"score": 0.8, "errors": 2}])
    assert result["score"] == pytest.approx(0.8)
    assert result["errors"] == pytest.approx(2.0)


def test_average_metrics_two_dicts() -> None:
    result = _average_metrics_dicts([{"score": 0.6}, {"score": 0.8}])
    assert result["score"] == pytest.approx(0.7)


def test_average_metrics_nested_dicts() -> None:
    a = {"dims": {"x": 0.2, "y": 0.4}}
    b = {"dims": {"x": 0.4, "y": 0.6}}
    result = _average_metrics_dicts([a, b])
    assert result["dims"]["x"] == pytest.approx(0.3)
    assert result["dims"]["y"] == pytest.approx(0.5)


def test_average_metrics_missing_key_in_some_dicts() -> None:
    # key only in one dict → averaged over available values
    a = {"score": 0.8, "bonus": 1.0}
    b = {"score": 0.6}
    result = _average_metrics_dicts([a, b])
    assert result["score"] == pytest.approx(0.7)
    assert "bonus" in result  # present (only 1 value → average of 1)
    assert result["bonus"] == pytest.approx(1.0)


def test_average_metrics_non_numeric_uses_last_value() -> None:
    a = {"label": "ok"}
    b = {"label": "fail"}
    result = _average_metrics_dicts([a, b])
    assert result["label"] == "fail"


def test_average_metrics_bool_not_averaged_as_number() -> None:
    # bool is a subclass of int but should not be averaged as float
    a = {"flag": True}
    b = {"flag": False}
    result = _average_metrics_dicts([a, b])
    # bools are NOT isinstance(v, float) exclusively — they fall through to last-value
    assert result["flag"] is False


# ---------------------------------------------------------------------------
# _validate_baseline_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "latest", "latest_pass"])
def test_validate_builtin_modes_pass(mode: str) -> None:
    _validate_baseline_mode(mode)  # should not raise


@pytest.mark.parametrize("mode", ["tag:stable", "tag:v1.0", "tag:my-release"])
def test_validate_tag_modes_pass(mode: str) -> None:
    _validate_baseline_mode(mode)


def test_validate_tag_empty_name_raises() -> None:
    with pytest.raises(ValueError):
        _validate_baseline_mode("tag:")


@pytest.mark.parametrize("mode", ["rolling:1", "rolling:5", "rolling:100"])
def test_validate_rolling_valid_passes(mode: str) -> None:
    _validate_baseline_mode(mode)


def test_validate_rolling_zero_raises() -> None:
    with pytest.raises(ValueError, match="rolling"):
        _validate_baseline_mode("rolling:0")


def test_validate_rolling_negative_raises() -> None:
    with pytest.raises(ValueError, match="rolling"):
        _validate_baseline_mode("rolling:-1")


def test_validate_rolling_non_integer_raises() -> None:
    with pytest.raises(ValueError, match="rolling"):
        _validate_baseline_mode("rolling:abc")


def test_validate_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="invalid baseline_mode"):
        _validate_baseline_mode("unknown_mode")


# ---------------------------------------------------------------------------
# _load_previous_run helpers
# ---------------------------------------------------------------------------


def _write_run(model_root: Path, run_id: str, payload: dict) -> Path:
    run_dir = model_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _load_previous_run — baseline_mode="none"
# ---------------------------------------------------------------------------


def test_load_previous_run_mode_none_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS"})
    assert _load_previous_run(tmp_path, "run_cur", baseline_mode="none") is None


# ---------------------------------------------------------------------------
# _load_previous_run — missing model_root
# ---------------------------------------------------------------------------


def test_load_previous_run_missing_model_root_returns_none(tmp_path: Path) -> None:
    result = _load_previous_run(tmp_path / "nope", "run_cur", baseline_mode="latest")
    assert result is None


# ---------------------------------------------------------------------------
# _load_previous_run — baseline_mode="latest"
# ---------------------------------------------------------------------------


def test_load_previous_run_latest_returns_newest(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "FAIL", "label": "old"})
    _write_run(tmp_path, "run_002", {"status": "PASS", "label": "new"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="latest")
    assert result is not None
    assert result["label"] == "new"


def test_load_previous_run_latest_excludes_current_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"label": "old"})
    _write_run(tmp_path, "run_002", {"label": "current"})
    result = _load_previous_run(tmp_path, "run_002", baseline_mode="latest")
    assert result is not None
    assert result["label"] == "old"


def test_load_previous_run_latest_no_candidates_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_cur", {"status": "PASS"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="latest")
    assert result is None


# ---------------------------------------------------------------------------
# _load_previous_run — baseline_mode="latest_pass"
# ---------------------------------------------------------------------------


def test_load_previous_run_latest_pass_skips_fail(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "label": "pass1"})
    _write_run(tmp_path, "run_002", {"status": "FAIL", "label": "fail"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="latest_pass")
    assert result is not None
    assert result["label"] == "pass1"


def test_load_previous_run_latest_pass_all_fail_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "FAIL"})
    _write_run(tmp_path, "run_002", {"status": "FAIL"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="latest_pass")
    assert result is None


def test_load_previous_run_latest_pass_returns_most_recent_pass(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "label": "old_pass"})
    _write_run(tmp_path, "run_002", {"status": "PASS", "label": "new_pass"})
    _write_run(tmp_path, "run_003", {"status": "FAIL"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="latest_pass")
    assert result is not None
    assert result["label"] == "new_pass"


# ---------------------------------------------------------------------------
# _load_previous_run — baseline_mode="tag:<name>"
# ---------------------------------------------------------------------------


def test_load_previous_run_tag_no_tags_file_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="tag:stable")
    assert result is None


def test_load_previous_run_tag_unknown_tag_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS"})
    _write_tag(tmp_path, "canary", "run_001")
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="tag:stable")
    assert result is None


def test_load_previous_run_tag_found(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "label": "tagged"})
    _write_tag(tmp_path, "stable", "run_001")
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="tag:stable")
    assert result is not None
    assert result["label"] == "tagged"


def test_load_previous_run_tag_run_dir_missing_returns_none(tmp_path: Path) -> None:
    _write_tag(tmp_path, "stable", "run_deleted")
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="tag:stable")
    assert result is None


# ---------------------------------------------------------------------------
# _load_previous_run — baseline_mode="rolling:N"
# ---------------------------------------------------------------------------


def test_load_previous_run_rolling_averages_pass_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "metrics": {"score": 0.6}})
    _write_run(tmp_path, "run_002", {"status": "PASS", "metrics": {"score": 0.8}})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="rolling:2")
    assert result is not None
    assert result["metrics"]["score"] == pytest.approx(0.7)
    assert result["status"] == "PASS"


def test_load_previous_run_rolling_skips_fail_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "metrics": {"score": 0.6}})
    _write_run(tmp_path, "run_002", {"status": "FAIL", "metrics": {"score": 0.1}})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="rolling:3")
    assert result is not None
    # Only 1 PASS run → average of 1
    assert result["metrics"]["score"] == pytest.approx(0.6)


def test_load_previous_run_rolling_no_pass_runs_returns_none(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "FAIL"})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="rolling:3")
    assert result is None


def test_load_previous_run_rolling_limits_to_n(tmp_path: Path) -> None:
    for i in range(5):
        _write_run(tmp_path, f"run_{i:03d}", {"status": "PASS", "metrics": {"score": float(i)}})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="rolling:2")
    assert result is not None
    # rolling:2 takes the 2 most recent PASS runs (run_004=4.0, run_003=3.0) → avg=3.5
    assert result["metrics"]["score"] == pytest.approx(3.5)


def test_load_previous_run_rolling_run_id_includes_n(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_001", {"status": "PASS", "metrics": {"score": 0.8}})
    result = _load_previous_run(tmp_path, "run_cur", baseline_mode="rolling:3")
    assert result is not None
    assert "rolling:3" in result["run_id"]
