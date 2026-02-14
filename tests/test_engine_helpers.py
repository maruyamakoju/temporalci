from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.engine import (
    _build_sample_rows_with_retention,
    _compare,
    _extract_metric_series,
    _compute_regressions,
    _create_run_dir,
    _evaluate_gates,
    _paired_deltas_for_gate,
    _read_sprt_params,
    _run_sprt,
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
    results = _evaluate_gates(gates, metrics, baseline_metrics=None)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[0]["actual"] == 0.7
    assert results[1]["passed"] is False


def test_evaluate_gates_missing_metric() -> None:
    gates = [GateSpec(metric="missing.path", op=">=", value=0.5)]
    results = _evaluate_gates(gates, {}, baseline_metrics=None)
    assert results[0]["passed"] is False
    assert "error" in results[0]


def test_evaluate_gates_unsupported_operator() -> None:
    gates = [GateSpec(metric="score", op="~=", value=0.5)]
    results = _evaluate_gates(gates, {"score": 0.5}, baseline_metrics=None)
    assert results[0]["passed"] is False
    assert "unsupported operator" in results[0]["error"]


def test_extract_metric_series_supports_temporal_score_from_dims() -> None:
    payload = {
        "vbench_temporal": {
            "per_sample": [
                {
                    "sample_id": "s1",
                    "test_id": "t1",
                    "seed": 0,
                    "prompt": "a",
                    "dims": {"x": 0.2, "y": 0.8},
                },
                {
                    "sample_id": "s2",
                    "test_id": "t1",
                    "seed": 1,
                    "prompt": "a",
                    "dims": {"x": 0.4, "y": 0.6},
                },
            ]
        }
    }
    rows, meta = _extract_metric_series(
        payload,
        "vbench_temporal.score",
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert len(rows) == 2
    assert meta["missing_sample_id_count"] == 0
    assert rows[0] == ("sid:s1", pytest.approx(0.5))
    assert rows[1] == ("sid:s2", pytest.approx(0.5))


def test_extract_metric_series_strict_mode_skips_rows_without_sample_id() -> None:
    payload = {
        "vbench_temporal": {
            "per_sample": [
                {"test_id": "t1", "seed": 0, "prompt": "a", "dims": {"x": 0.2, "y": 0.8}},
                {"test_id": "t1", "seed": 1, "prompt": "a", "dims": {"x": 0.4, "y": 0.6}},
            ]
        }
    }
    rows, meta = _extract_metric_series(
        payload,
        "vbench_temporal.score",
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert rows == []
    assert meta["missing_sample_id_count"] == 2


def test_paired_deltas_for_gate_uses_key_matching() -> None:
    current = {
        "vbench_temporal": {
            "per_sample": [
                {
                    "sample_id": "s1",
                    "test_id": "t1",
                    "seed": 0,
                    "prompt": "p0",
                    "dims": {"motion_smoothness": 0.2},
                },
                {
                    "sample_id": "s2",
                    "test_id": "t1",
                    "seed": 1,
                    "prompt": "p1",
                    "dims": {"motion_smoothness": 0.4},
                },
            ]
        }
    }
    baseline = {
        "vbench_temporal": {
            "per_sample": [
                {
                    "sample_id": "s1",
                    "test_id": "t1",
                    "seed": 0,
                    "prompt": "p0",
                    "dims": {"motion_smoothness": 0.5},
                },
                {
                    "sample_id": "s2",
                    "test_id": "t1",
                    "seed": 1,
                    "prompt": "p1",
                    "dims": {"motion_smoothness": 0.6},
                },
            ]
        }
    }
    deltas, summary = _paired_deltas_for_gate(
        metric_path="vbench_temporal.dims.motion_smoothness",
        op=">=",
        current_metrics=current,
        baseline_metrics=baseline,
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert summary["pairing"] == "key_match"
    assert deltas == pytest.approx([-0.3, -0.2])
    assert summary["paired_ratio"] == 1.0


def test_run_sprt_detects_regression() -> None:
    params = _read_sprt_params(
        {
            "alpha": 0.05,
            "beta": 0.1,
            "effect_size": 0.05,
            "sigma_floor": 0.01,
            "min_pairs": 6,
            "inconclusive": "fail",
        }
    )
    deltas = [-0.2, -0.18, -0.21, -0.17, -0.16, -0.19, -0.2, -0.18]
    payload = _run_sprt(deltas=deltas, params=params)
    assert payload["decision"] == "accept_h0_regression"
    assert payload["decision_passed"] is False


def test_evaluate_gates_sprt_regression_fails_on_degraded_series() -> None:
    current = {
        "vbench_temporal": {
            "score": 0.5,
            "dims": {"motion_smoothness": 0.266667},
            "per_sample": [
                {
                    "sample_id": "s0",
                    "test_id": "t1",
                    "seed": 0,
                    "prompt": "p0",
                    "dims": {"motion_smoothness": 0.2},
                },
                {
                    "sample_id": "s1",
                    "test_id": "t1",
                    "seed": 1,
                    "prompt": "p1",
                    "dims": {"motion_smoothness": 0.25},
                },
                {
                    "sample_id": "s2",
                    "test_id": "t1",
                    "seed": 2,
                    "prompt": "p2",
                    "dims": {"motion_smoothness": 0.3},
                },
                {
                    "sample_id": "s3",
                    "test_id": "t1",
                    "seed": 3,
                    "prompt": "p3",
                    "dims": {"motion_smoothness": 0.35},
                },
                {
                    "sample_id": "s4",
                    "test_id": "t1",
                    "seed": 4,
                    "prompt": "p4",
                    "dims": {"motion_smoothness": 0.28},
                },
                {
                    "sample_id": "s5",
                    "test_id": "t1",
                    "seed": 5,
                    "prompt": "p5",
                    "dims": {"motion_smoothness": 0.22},
                },
            ],
        }
    }
    baseline = {
        "vbench_temporal": {
            "score": 0.8,
            "dims": {"motion_smoothness": 0.655},
            "per_sample": [
                {
                    "sample_id": "s0",
                    "test_id": "t1",
                    "seed": 0,
                    "prompt": "p0",
                    "dims": {"motion_smoothness": 0.65},
                },
                {
                    "sample_id": "s1",
                    "test_id": "t1",
                    "seed": 1,
                    "prompt": "p1",
                    "dims": {"motion_smoothness": 0.66},
                },
                {
                    "sample_id": "s2",
                    "test_id": "t1",
                    "seed": 2,
                    "prompt": "p2",
                    "dims": {"motion_smoothness": 0.67},
                },
                {
                    "sample_id": "s3",
                    "test_id": "t1",
                    "seed": 3,
                    "prompt": "p3",
                    "dims": {"motion_smoothness": 0.64},
                },
                {
                    "sample_id": "s4",
                    "test_id": "t1",
                    "seed": 4,
                    "prompt": "p4",
                    "dims": {"motion_smoothness": 0.63},
                },
                {
                    "sample_id": "s5",
                    "test_id": "t1",
                    "seed": 5,
                    "prompt": "p5",
                    "dims": {"motion_smoothness": 0.68},
                },
            ],
        }
    }
    gates = [
        GateSpec(
            metric="vbench_temporal.dims.motion_smoothness",
            op=">=",
            value=0.2,
            method="sprt_regression",
            params={"effect_size": 0.05, "min_pairs": 6, "inconclusive": "fail"},
        )
    ]
    results = _evaluate_gates(gates, current, baseline_metrics=baseline)
    assert results[0]["threshold_passed"] is True
    assert results[0]["sprt"]["decision"] == "accept_h0_regression"
    assert results[0]["passed"] is False


def test_evaluate_gates_sprt_baseline_missing_defaults_to_fail() -> None:
    gates = [
        GateSpec(
            metric="vbench_temporal.dims.motion_smoothness",
            op=">=",
            value=0.1,
            method="sprt_regression",
            params={},
        )
    ]
    metrics = {
        "vbench_temporal": {
            "dims": {"motion_smoothness": 0.5},
            "per_sample": [{"sample_id": "s1", "dims": {"motion_smoothness": 0.5}}],
        }
    }
    results = _evaluate_gates(gates, metrics, baseline_metrics=None)
    assert results[0]["threshold_passed"] is True
    assert results[0]["sprt"]["baseline_missing_policy"] == "fail"
    assert results[0]["passed"] is False


def test_evaluate_gates_sprt_baseline_missing_skip_allows_threshold_pass() -> None:
    gates = [
        GateSpec(
            metric="vbench_temporal.dims.motion_smoothness",
            op=">=",
            value=0.1,
            method="sprt_regression",
            params={"require_baseline": False, "baseline_missing": "skip"},
        )
    ]
    metrics = {
        "vbench_temporal": {
            "dims": {"motion_smoothness": 0.5},
            "per_sample": [{"sample_id": "s1", "dims": {"motion_smoothness": 0.5}}],
        }
    }
    results = _evaluate_gates(gates, metrics, baseline_metrics=None)
    assert results[0]["threshold_passed"] is True
    assert results[0]["sprt"]["baseline_missing_policy"] == "skip"
    assert results[0]["passed"] is True


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
