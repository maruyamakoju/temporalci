from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.engine import (
    _build_sample_rows_with_retention,
    _compare,
    _compute_regressions,
    _create_run_dir,
    _evaluate_gates,
    _resolve_metric_path,
    _safe_unlink,
)
from temporalci.types import GateSpec, GeneratedSample


# ---------------------------------------------------------------------------
# _resolve_metric_path
# ---------------------------------------------------------------------------


def test_resolve_metric_path_nested() -> None:
    payload = {"a": {"b": {"c": 42}}}
    assert _resolve_metric_path(payload, "a.b.c") == 42


def test_resolve_metric_path_top_level() -> None:
    payload = {"score": 0.9}
    assert _resolve_metric_path(payload, "score") == 0.9


def test_resolve_metric_path_missing_key() -> None:
    with pytest.raises(KeyError):
        _resolve_metric_path({"a": 1}, "b")


def test_resolve_metric_path_deep_missing() -> None:
    with pytest.raises(KeyError):
        _resolve_metric_path({"a": {"b": 1}}, "a.c")


# ---------------------------------------------------------------------------
# _compare
# ---------------------------------------------------------------------------


def test_compare_eq() -> None:
    assert _compare(1, "==", 1) is True
    assert _compare(1, "==", 2) is False
    assert _compare("hello", "==", "hello") is True


def test_compare_neq() -> None:
    assert _compare(1, "!=", 2) is True
    assert _compare(1, "!=", 1) is False


def test_compare_gte() -> None:
    assert _compare(5, ">=", 5) is True
    assert _compare(6, ">=", 5) is True
    assert _compare(4, ">=", 5) is False


def test_compare_lte() -> None:
    assert _compare(5, "<=", 5) is True
    assert _compare(4, "<=", 5) is True
    assert _compare(6, "<=", 5) is False


def test_compare_gt() -> None:
    assert _compare(6, ">", 5) is True
    assert _compare(5, ">", 5) is False


def test_compare_lt() -> None:
    assert _compare(4, "<", 5) is True
    assert _compare(5, "<", 5) is False


def test_compare_unsupported_op() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        _compare(1, "~=", 1)


# ---------------------------------------------------------------------------
# _evaluate_gates
# ---------------------------------------------------------------------------


def test_evaluate_gates_pass_and_fail() -> None:
    gates = [
        GateSpec(metric="score", op=">=", value=0.5),
        GateSpec(metric="score", op=">=", value=0.9),
    ]
    metrics = {"score": 0.7}
    results = _evaluate_gates(gates, metrics)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[0]["actual"] == 0.7
    assert results[1]["passed"] is False


def test_evaluate_gates_missing_metric() -> None:
    gates = [GateSpec(metric="missing.path", op=">=", value=0.5)]
    results = _evaluate_gates(gates, {})
    assert results[0]["passed"] is False
    assert "error" in results[0]


def test_evaluate_gates_unsupported_operator() -> None:
    gates = [GateSpec(metric="score", op="~=", value=0.5)]
    results = _evaluate_gates(gates, {"score": 0.5})
    assert results[0]["passed"] is False
    assert "unsupported operator" in results[0]["error"]


# ---------------------------------------------------------------------------
# _compute_regressions
# ---------------------------------------------------------------------------


def test_compute_regressions_detects_regression() -> None:
    gates = [{"metric": "score", "op": ">=", "value": 0.5}]
    current = {"score": 0.4}
    baseline = {"score": 0.6}
    regressions = _compute_regressions(gates, current, baseline)
    assert len(regressions) == 1
    assert regressions[0]["regressed"] is True
    assert regressions[0]["delta"] == pytest.approx(-0.2)


def test_compute_regressions_no_regression() -> None:
    gates = [{"metric": "score", "op": ">=", "value": 0.5}]
    current = {"score": 0.8}
    baseline = {"score": 0.6}
    regressions = _compute_regressions(gates, current, baseline)
    assert regressions[0]["regressed"] is False


def test_compute_regressions_lower_is_better() -> None:
    gates = [{"metric": "errors", "op": "<=", "value": 5}]
    current = {"errors": 10}
    baseline = {"errors": 3}
    regressions = _compute_regressions(gates, current, baseline)
    assert regressions[0]["regressed"] is True
    assert regressions[0]["direction"] == "lower_is_better"


def test_compute_regressions_none_baseline() -> None:
    gates = [{"metric": "score", "op": ">=", "value": 0.5}]
    assert _compute_regressions(gates, {"score": 0.5}, None) == []


def test_compute_regressions_skips_eq_operator() -> None:
    gates = [{"metric": "score", "op": "==", "value": 0.5}]
    regressions = _compute_regressions(gates, {"score": 0.5}, {"score": 0.3})
    assert regressions == []


def test_compute_regressions_skips_non_numeric() -> None:
    gates = [{"metric": "status", "op": ">=", "value": "ok"}]
    regressions = _compute_regressions(gates, {"status": "ok"}, {"status": "ok"})
    assert regressions == []


# ---------------------------------------------------------------------------
# _build_sample_rows_with_retention
# ---------------------------------------------------------------------------


def _make_sample(video_path: str) -> GeneratedSample:
    return GeneratedSample(
        test_id="t1",
        prompt="test",
        seed=0,
        video_path=video_path,
        evaluation_stream=[0.5],
    )


def test_retention_all_policy(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_text("data", encoding="utf-8")
    samples = [_make_sample(str(video))]
    rows = _build_sample_rows_with_retention(
        samples=samples, status="PASS", artifacts_cfg={"video": "all"}
    )
    assert rows[0]["artifact_retained"] is True
    assert video.exists()


def test_retention_none_policy(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_text("data", encoding="utf-8")
    samples = [_make_sample(str(video))]
    rows = _build_sample_rows_with_retention(
        samples=samples, status="PASS", artifacts_cfg={"video": "none"}
    )
    assert rows[0]["artifact_retained"] is False
    assert rows[0]["artifact_deleted"] is True
    assert not video.exists()


def test_retention_failures_only_on_pass(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_text("data", encoding="utf-8")
    samples = [_make_sample(str(video))]
    rows = _build_sample_rows_with_retention(
        samples=samples, status="PASS", artifacts_cfg={"video": "failures_only"}
    )
    assert rows[0]["artifact_retained"] is False


def test_retention_failures_only_on_fail(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_text("data", encoding="utf-8")
    samples = [_make_sample(str(video))]
    rows = _build_sample_rows_with_retention(
        samples=samples, status="FAIL", artifacts_cfg={"video": "failures_only"}
    )
    assert rows[0]["artifact_retained"] is True


def test_retention_max_samples(tmp_path: Path) -> None:
    videos = []
    for i in range(5):
        v = tmp_path / f"v{i}.mp4"
        v.write_text("data", encoding="utf-8")
        videos.append(_make_sample(str(v)))
    rows = _build_sample_rows_with_retention(
        samples=videos, status="PASS", artifacts_cfg={"video": "all", "max_samples": 2}
    )
    retained = [r for r in rows if r["artifact_retained"]]
    deleted = [r for r in rows if r["artifact_deleted"]]
    assert len(retained) == 2
    assert len(deleted) == 3


# ---------------------------------------------------------------------------
# _create_run_dir
# ---------------------------------------------------------------------------


def test_create_run_dir(tmp_path: Path) -> None:
    model_root = tmp_path / "root"
    run_id, run_dir = _create_run_dir(model_root)
    assert run_dir.exists()
    assert run_dir.parent == model_root
    assert run_id == run_dir.name


# ---------------------------------------------------------------------------
# _safe_unlink
# ---------------------------------------------------------------------------


def test_safe_unlink_existing(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("data", encoding="utf-8")
    assert _safe_unlink(f) is True
    assert not f.exists()


def test_safe_unlink_missing(tmp_path: Path) -> None:
    assert _safe_unlink(tmp_path / "nope.txt") is False
