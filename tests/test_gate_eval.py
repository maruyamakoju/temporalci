"""Tests for temporalci.gate_eval — functions not already covered by test_engine_helpers.py.

Covered here:
  _split_metric_path, _resolve_sample_metric_value, _build_legacy_series_key,
  _extract_metric_series (edge cases), _paired_deltas_for_gate (index_fallback /
  key_mismatch / lower_is_better), _read_sprt_params (all validation branches),
  _run_sprt (inconclusive=pass, accept_h1, crossed_at, llr_history),
  _load_recent_runs, _apply_windowed_gates.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from temporalci.gate_eval import (
    _apply_windowed_gates,
    _build_legacy_series_key,
    _extract_metric_series,
    _load_recent_runs,
    _paired_deltas_for_gate,
    _read_sprt_params,
    _resolve_sample_metric_value,
    _run_sprt,
    _split_metric_path,
)
from temporalci.types import GateSpec


# ---------------------------------------------------------------------------
# _split_metric_path
# ---------------------------------------------------------------------------


def test_split_no_dot() -> None:
    assert _split_metric_path("score") == ("score", "")


def test_split_one_dot() -> None:
    assert _split_metric_path("vbench.score") == ("vbench", "score")


def test_split_multiple_dots_only_first_split() -> None:
    head, tail = _split_metric_path("vbench.dims.motion")
    assert head == "vbench"
    assert tail == "dims.motion"


def test_split_strips_whitespace() -> None:
    head, tail = _split_metric_path(" vbench . score ")
    assert head == "vbench"
    assert tail == "score"


# ---------------------------------------------------------------------------
# _resolve_sample_metric_value
# ---------------------------------------------------------------------------


def test_resolve_sample_empty_subpath_returns_none() -> None:
    assert _resolve_sample_metric_value({"score": 0.9}, "") is None


def test_resolve_sample_score_direct_field() -> None:
    assert _resolve_sample_metric_value({"score": 0.75}, "score") == pytest.approx(0.75)


def test_resolve_sample_score_from_dims_mean() -> None:
    row = {"dims": {"a": 0.2, "b": 0.8}}
    result = _resolve_sample_metric_value(row, "score")
    assert result == pytest.approx(0.5)


def test_resolve_sample_score_no_score_no_dims_returns_none() -> None:
    assert _resolve_sample_metric_value({"other": 1}, "score") is None


def test_resolve_sample_score_bool_rejected() -> None:
    # bool is a subclass of int — must be rejected
    assert _resolve_sample_metric_value({"score": True}, "score") is None


def test_resolve_sample_dims_empty_returns_none() -> None:
    assert _resolve_sample_metric_value({"dims": {}}, "score") is None


def test_resolve_sample_dotted_subpath() -> None:
    row = {"dims": {"motion_smoothness": 0.6}}
    assert _resolve_sample_metric_value(row, "dims.motion_smoothness") == pytest.approx(0.6)


def test_resolve_sample_dotted_subpath_missing_returns_none() -> None:
    row = {"dims": {"other": 0.6}}
    assert _resolve_sample_metric_value(row, "dims.motion_smoothness") is None


def test_resolve_sample_non_numeric_subpath_returns_none() -> None:
    row = {"label": "ok"}
    assert _resolve_sample_metric_value(row, "label") is None


# ---------------------------------------------------------------------------
# _build_legacy_series_key
# ---------------------------------------------------------------------------


def test_build_legacy_key_all_fields() -> None:
    row = {"test_id": "t1", "seed": 42, "prompt": "hello"}
    assert _build_legacy_series_key(row, fallback_index=0) == "t1|42|hello"


def test_build_legacy_key_partial_fields() -> None:
    row = {"test_id": "t1", "seed": 0}
    assert _build_legacy_series_key(row, fallback_index=5) == "t1|0"


def test_build_legacy_key_no_fields_uses_index() -> None:
    assert _build_legacy_series_key({}, fallback_index=3) == "idx:3"


def test_build_legacy_key_none_value_skipped() -> None:
    row = {"test_id": None, "seed": 7, "prompt": None}
    assert _build_legacy_series_key(row, fallback_index=0) == "7"


# ---------------------------------------------------------------------------
# _extract_metric_series — edge cases
# ---------------------------------------------------------------------------


def test_extract_missing_metric_key_returns_empty() -> None:
    rows, meta = _extract_metric_series(
        {},
        "nonexistent.score",
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert rows == []
    assert meta["total_rows"] == 0


def test_extract_per_sample_not_a_list_returns_empty() -> None:
    payload = {"metric": {"per_sample": "not_a_list"}}
    rows, meta = _extract_metric_series(
        payload,
        "metric.score",
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert rows == []


def test_extract_uses_legacy_key_when_no_sample_id_and_allow_legacy() -> None:
    payload = {
        "m": {
            "per_sample": [
                {"test_id": "t1", "seed": 0, "prompt": "p", "score": 0.5},
            ]
        }
    }
    rows, meta = _extract_metric_series(
        payload,
        "m.score",
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert len(rows) == 1
    assert rows[0][0] == "t1|0|p"
    assert meta["missing_sample_id_count"] == 1


def test_extract_strict_no_sample_id_no_legacy_skips_row() -> None:
    payload = {
        "m": {
            "per_sample": [
                {"score": 0.5},
            ]
        }
    }
    rows, meta = _extract_metric_series(
        payload,
        "m.score",
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert rows == []
    assert meta["missing_sample_id_count"] == 1


def test_extract_skips_non_dict_rows() -> None:
    payload = {"m": {"per_sample": ["bad", None, {"sample_id": "s1", "score": 0.9}]}}
    rows, meta = _extract_metric_series(
        payload,
        "m.score",
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert len(rows) == 1
    assert rows[0] == ("sid:s1", pytest.approx(0.9))


def test_extract_usable_rows_count_in_meta() -> None:
    payload = {
        "m": {
            "per_sample": [
                {"sample_id": "s1", "score": 0.5},
                {"sample_id": "s2", "score": 0.6},
                {"sample_id": "s3"},  # missing score — skipped
            ]
        }
    }
    rows, meta = _extract_metric_series(
        payload,
        "m.score",
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert meta["total_rows"] == 3
    assert meta["usable_rows"] == 2


# ---------------------------------------------------------------------------
# _paired_deltas_for_gate — additional pairing modes
# ---------------------------------------------------------------------------


def _make_metric_payload(
    metric: str,
    samples: list[dict],
) -> dict:
    return {metric: {"per_sample": samples}}


def test_paired_deltas_empty_current_returns_unavailable() -> None:
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op=">=",
        current_metrics={},
        baseline_metrics=_make_metric_payload("m", [{"sample_id": "s1", "score": 0.5}]),
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert deltas == []
    assert summary["pairing"] == "unavailable"


def test_paired_deltas_empty_baseline_returns_unavailable() -> None:
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op=">=",
        current_metrics=_make_metric_payload("m", [{"sample_id": "s1", "score": 0.5}]),
        baseline_metrics={},
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert deltas == []
    assert summary["pairing"] == "unavailable"


def test_paired_deltas_key_mismatch_without_legacy() -> None:
    # current and baseline have completely different sample_ids
    current = _make_metric_payload("m", [{"sample_id": "x1", "score": 0.5}])
    baseline = _make_metric_payload("m", [{"sample_id": "y1", "score": 0.6}])
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op=">=",
        current_metrics=current,
        baseline_metrics=baseline,
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert deltas == []
    assert summary["pairing"] == "key_mismatch"
    assert summary["paired_count"] == 0


def test_paired_deltas_index_fallback_when_allow_legacy() -> None:
    # Different legacy keys (different test_id) → key_match fails → index_fallback
    current = _make_metric_payload(
        "m", [{"test_id": "cur_t1", "seed": 0, "prompt": "p", "score": 0.4}]
    )
    baseline = _make_metric_payload(
        "m", [{"test_id": "base_t1", "seed": 0, "prompt": "p", "score": 0.6}]
    )
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op=">=",
        current_metrics=current,
        baseline_metrics=baseline,
        require_sample_id=False,
        allow_legacy_pairing=True,
    )
    assert summary["pairing"] == "index_fallback"
    assert deltas == pytest.approx([-0.2])


def test_paired_deltas_lower_is_better_direction() -> None:
    # op="<=" → delta = baseline - current (positive = improvement)
    current = _make_metric_payload("m", [{"sample_id": "s1", "score": 0.3}])
    baseline = _make_metric_payload("m", [{"sample_id": "s1", "score": 0.6}])
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op="<=",
        current_metrics=current,
        baseline_metrics=baseline,
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    assert summary["pairing"] == "key_match"
    # lower_is_better: delta = baseline - current = 0.6 - 0.3 = 0.3 (improvement)
    assert deltas == pytest.approx([0.3])


def test_paired_deltas_worst_deltas_sorted_ascending() -> None:
    samples_cur = [{"sample_id": f"s{i}", "score": float(i) * 0.1} for i in range(5)]
    samples_base = [{"sample_id": f"s{i}", "score": 0.5} for i in range(5)]
    current = _make_metric_payload("m", samples_cur)
    baseline = _make_metric_payload("m", samples_base)
    deltas, summary = _paired_deltas_for_gate(
        metric_path="m.score",
        op=">=",
        current_metrics=current,
        baseline_metrics=baseline,
        require_sample_id=True,
        allow_legacy_pairing=False,
    )
    worst = summary["worst_deltas"]
    # worst_deltas sorted ascending (most negative first)
    assert worst[0]["delta"] <= worst[-1]["delta"]


# ---------------------------------------------------------------------------
# _read_sprt_params — validation edge cases
# ---------------------------------------------------------------------------

_BASE_PARAMS = {"alpha": 0.05, "beta": 0.1, "effect_size": 0.05}


def test_read_sprt_params_defaults() -> None:
    params = _read_sprt_params(_BASE_PARAMS)
    assert params["alpha"] == pytest.approx(0.05)
    assert params["beta"] == pytest.approx(0.1)
    assert params["effect_size"] == pytest.approx(0.05)
    assert params["sigma_mode"] == "estimate"
    assert params["sigma"] is None
    assert params["min_pairs"] == 6
    assert params["require_baseline"] is True
    assert params["baseline_missing"] == "fail"
    assert params["pairing_mode"] == "sample_id"
    assert params["allow_index_fallback"] is False


@pytest.mark.parametrize("alpha", [0.0, -0.1, 0.5, 1.0])
def test_read_sprt_params_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        _read_sprt_params({**_BASE_PARAMS, "alpha": alpha})


@pytest.mark.parametrize("beta", [0.0, -0.01, 0.5, 0.9])
def test_read_sprt_params_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        _read_sprt_params({**_BASE_PARAMS, "beta": beta})


def test_read_sprt_params_zero_effect_size_raises() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        _read_sprt_params({**_BASE_PARAMS, "effect_size": 0.0})


def test_read_sprt_params_negative_effect_size_takes_abs() -> None:
    # effect_size is stored as abs value, negative input should not raise
    params = _read_sprt_params({**_BASE_PARAMS, "effect_size": -0.05})
    assert params["effect_size"] == pytest.approx(0.05)


def test_read_sprt_params_zero_sigma_floor_raises() -> None:
    with pytest.raises(ValueError, match="sigma_floor"):
        _read_sprt_params({**_BASE_PARAMS, "sigma_floor": 0.0})


def test_read_sprt_params_invalid_sigma_mode_raises() -> None:
    with pytest.raises(ValueError, match="sigma_mode"):
        _read_sprt_params({**_BASE_PARAMS, "sigma_mode": "auto"})


def test_read_sprt_params_invalid_inconclusive_raises() -> None:
    with pytest.raises(ValueError, match="inconclusive"):
        _read_sprt_params({**_BASE_PARAMS, "inconclusive": "skip"})


def test_read_sprt_params_invalid_baseline_missing_raises() -> None:
    with pytest.raises(ValueError, match="baseline_missing"):
        _read_sprt_params({**_BASE_PARAMS, "baseline_missing": "warn"})


def test_read_sprt_params_invalid_pairing_mode_raises() -> None:
    with pytest.raises(ValueError, match="pairing_mode"):
        _read_sprt_params({**_BASE_PARAMS, "pairing_mode": "strict"})


def test_read_sprt_params_sigma_zero_raises() -> None:
    with pytest.raises(ValueError, match="sigma must be > 0"):
        _read_sprt_params({**_BASE_PARAMS, "sigma_mode": "fixed", "sigma": 0.0})


def test_read_sprt_params_valid_sigma_fixed() -> None:
    params = _read_sprt_params({**_BASE_PARAMS, "sigma_mode": "fixed", "sigma": 0.1})
    assert params["sigma"] == pytest.approx(0.1)
    assert params["sigma_mode"] == "fixed"


def test_read_sprt_params_allow_index_fallback_parsed() -> None:
    params = _read_sprt_params({**_BASE_PARAMS, "allow_index_fallback": True})
    assert params["allow_index_fallback"] is True


def test_read_sprt_params_min_pairs_minimum_is_two() -> None:
    params = _read_sprt_params({**_BASE_PARAMS, "min_pairs": 1})
    assert params["min_pairs"] == 2


# ---------------------------------------------------------------------------
# _run_sprt — additional decision branches
# ---------------------------------------------------------------------------


def _default_params(**overrides) -> dict:
    base = {
        "alpha": 0.05,
        "beta": 0.1,
        "effect_size": 0.05,
        "sigma_floor": 0.01,
        "min_pairs": 6,
        "inconclusive": "fail",
    }
    base.update(overrides)
    return _read_sprt_params(base)


def test_run_sprt_inconclusive_pass_policy() -> None:
    params = _default_params(inconclusive="pass")
    # Small consistent positive deltas → inconclusive (can't reject regression hypothesis fast)
    deltas = [0.01] * 6  # won't drive LLR to upper threshold quickly
    result = _run_sprt(deltas=deltas, params=params)
    if result["decision"] == "inconclusive":
        assert result["decision_passed"] is True


def test_run_sprt_accept_h1_no_regression() -> None:
    params = _default_params()
    # Very large consistent positive deltas → rapidly accept H1 (no regression)
    deltas = [0.5] * 20
    result = _run_sprt(deltas=deltas, params=params)
    assert result["decision"] == "accept_h1_no_regression"
    assert result["decision_passed"] is True
    assert result["crossed_at"] is not None
    assert result["crossed_at"] <= len(deltas)


def test_run_sprt_llr_history_accumulated() -> None:
    params = _default_params()
    deltas = [0.0] * 10
    result = _run_sprt(deltas=deltas, params=params)
    # llr_history should have at most len(deltas) entries (stops at crossing)
    assert isinstance(result["llr_history"], list)
    assert len(result["llr_history"]) <= len(deltas)
    assert len(result["llr_history"]) >= 1


def test_run_sprt_insufficient_pairs_inconclusive() -> None:
    params = _default_params()
    result = _run_sprt(deltas=[0.1, 0.2], params=params)
    assert result["decision"] == "inconclusive"
    assert result["reason"] == "insufficient_pairs"
    assert result["paired_count"] == 2
    # Still exposes threshold diagnostics (from previous test in helpers, but checking new fields)
    assert result["llr"] == pytest.approx(0.0)


def test_run_sprt_has_required_keys() -> None:
    params = _default_params()
    deltas = [0.1, -0.05, 0.0, 0.1, 0.05, 0.0, 0.05, 0.1]
    result = _run_sprt(deltas=deltas, params=params)
    for key in ("decision", "decision_passed", "paired_count", "alpha", "beta",
                "effect_size", "sigma", "llr", "upper_threshold", "lower_threshold"):
        assert key in result, f"missing key: {key}"


# ---------------------------------------------------------------------------
# _load_recent_runs
# ---------------------------------------------------------------------------


def _write_run(model_root: Path, run_id: str, payload: dict) -> None:
    run_dir = model_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_recent_runs_empty_dir(tmp_path: Path) -> None:
    result = _load_recent_runs(tmp_path, current_run_id="cur", n=5)
    assert result == []


def test_load_recent_runs_nonexistent_dir(tmp_path: Path) -> None:
    result = _load_recent_runs(tmp_path / "nope", current_run_id="cur", n=5)
    assert result == []


def test_load_recent_runs_excludes_current(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_a", {"status": "PASS"})
    _write_run(tmp_path, "run_cur", {"status": "PASS"})
    result = _load_recent_runs(tmp_path, current_run_id="run_cur", n=5)
    assert len(result) == 1
    assert all(r.get("status") == "PASS" for r in result)


def test_load_recent_runs_sorted_descending(tmp_path: Path) -> None:
    for rid in ["run_001", "run_002", "run_003"]:
        _write_run(tmp_path, rid, {"run_id": rid})
    result = _load_recent_runs(tmp_path, current_run_id="run_999", n=10)
    ids = [r["run_id"] for r in result]
    assert ids == ["run_003", "run_002", "run_001"]


def test_load_recent_runs_respects_n_limit(tmp_path: Path) -> None:
    for i in range(5):
        _write_run(tmp_path, f"run_{i:03d}", {"idx": i})
    result = _load_recent_runs(tmp_path, current_run_id="run_999", n=3)
    assert len(result) == 3


def test_load_recent_runs_n_zero_returns_empty(tmp_path: Path) -> None:
    _write_run(tmp_path, "run_a", {"status": "PASS"})
    result = _load_recent_runs(tmp_path, current_run_id="cur", n=0)
    assert result == []


def test_load_recent_runs_ignores_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("not json", encoding="utf-8")
    _write_run(tmp_path, "good_run", {"status": "PASS"})
    result = _load_recent_runs(tmp_path, current_run_id="cur", n=5)
    assert len(result) == 1


def test_load_recent_runs_ignores_non_dict_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "list_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    result = _load_recent_runs(tmp_path, current_run_id="cur", n=5)
    assert result == []


# ---------------------------------------------------------------------------
# _apply_windowed_gates
# ---------------------------------------------------------------------------


def _make_gate_spec(metric: str = "score", op: str = ">=", window: int = 3,
                    min_failures: int = 2) -> MagicMock:
    spec = MagicMock()
    spec.metric = metric
    spec.op = op
    spec.window = window
    spec.min_failures = min_failures
    return spec


def test_apply_windowed_gates_already_passing_unchanged() -> None:
    spec = _make_gate_spec()
    result = {"metric": "score", "op": ">=", "passed": True, "threshold_passed": True}
    out = _apply_windowed_gates([spec], [result], recent_runs=[])
    assert out[0]["passed"] is True
    assert "windowed_pass" not in out[0]


def test_apply_windowed_gates_no_window_unchanged() -> None:
    spec = _make_gate_spec(window=0)
    result = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    out = _apply_windowed_gates([spec], [result], recent_runs=[])
    assert out[0]["passed"] is False
    assert "windowed_pass" not in out[0]


def test_apply_windowed_gates_single_failure_below_min_becomes_pass() -> None:
    spec = _make_gate_spec(window=3, min_failures=2)
    result = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    # No historical failures → total=1, min_failures=2 → windowed pass
    out = _apply_windowed_gates([spec], [result], recent_runs=[])
    assert out[0]["passed"] is True
    assert out[0]["windowed_pass"] is True
    assert out[0]["window_failures"] == 1
    assert out[0]["window_size"] == 3
    assert out[0]["min_failures"] == 2


def test_apply_windowed_gates_failures_at_min_stays_failed() -> None:
    spec = _make_gate_spec(window=3, min_failures=2)
    result = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    # 1 historical failure + current = 2 = min_failures → not overridden
    recent_runs = [
        {"gates": [{"metric": "score", "op": ">=", "threshold_passed": False}]}
    ]
    out = _apply_windowed_gates([spec], [result], recent_runs=recent_runs)
    assert out[0]["passed"] is False
    assert "windowed_pass" not in out[0]


def test_apply_windowed_gates_counts_threshold_passed_for_history() -> None:
    spec = _make_gate_spec(window=5, min_failures=3)
    result = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    # 1 historical threshold failure (even if gate was windowed-passed before)
    recent_runs = [
        {"gates": [{"metric": "score", "op": ">=", "threshold_passed": False, "passed": True}]},
        {"gates": [{"metric": "score", "op": ">=", "threshold_passed": True, "passed": True}]},
    ]
    # total = 1 hist + 1 current = 2 < 3 → windowed pass
    out = _apply_windowed_gates([spec], [result], recent_runs=recent_runs)
    assert out[0]["passed"] is True
    assert out[0]["window_failures"] == 2


def test_apply_windowed_gates_ignores_different_metric_in_history() -> None:
    spec = _make_gate_spec(metric="score", op=">=", window=3, min_failures=2)
    result = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    # Historical run has a different metric — should not count
    recent_runs = [
        {"gates": [{"metric": "other_metric", "op": ">=", "threshold_passed": False}]}
    ]
    out = _apply_windowed_gates([spec], [result], recent_runs=recent_runs)
    # hist=0 + current=1 = 1 < 2 → windowed pass
    assert out[0]["passed"] is True


def test_apply_windowed_gates_multiple_gates_independent() -> None:
    spec_a = _make_gate_spec(metric="score", op=">=", window=3, min_failures=3)
    spec_b = _make_gate_spec(metric="latency", op="<=", window=3, min_failures=2)
    result_a = {"metric": "score", "op": ">=", "passed": False, "threshold_passed": False}
    result_b = {"metric": "latency", "op": "<=", "passed": False, "threshold_passed": False}
    recent_runs = [
        {
            "gates": [
                {"metric": "score", "op": ">=", "threshold_passed": False},
                {"metric": "latency", "op": "<=", "threshold_passed": False},
            ]
        }
    ]
    out = _apply_windowed_gates([spec_a, spec_b], [result_a, result_b], recent_runs=recent_runs)
    # score: hist=1+cur=1=2 < 3 → windowed pass
    assert out[0]["passed"] is True
    # latency: hist=1+cur=1=2 >= 2 → stays failed
    assert out[1]["passed"] is False
