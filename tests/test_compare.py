from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.compare import compare_runs, format_compare_text, write_compare_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    *,
    run_id: str = "run1",
    status: str = "PASS",
    score: float = 0.8,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal run payload suitable for compare tests."""
    return {
        "run_id": run_id,
        "status": status,
        "timestamp_utc": "2026-02-12T00:00:00+00:00",
        "project": "test-proj",
        "suite_name": "suite1",
        "model_name": "mock",
        "metrics": {
            "vbench_temporal": {
                "score": score,
                "dims": {
                    "motion_smoothness": score,
                    "temporal_flicker": score,
                },
                "per_sample": [
                    {
                        "sample_id": "abc123",
                        "score": score,
                    }
                ],
            }
        },
        "gates": [
            {
                "metric": "vbench_temporal.score",
                "op": ">=",
                "value": 0.5,
                "actual": score,
                "passed": score >= 0.5,
            }
        ],
        "samples": samples
        if samples is not None
        else [
            {
                "test_id": "core",
                "prompt": "a calm city timelapse",
                "seed": 0,
                "metadata": {"sample_id": "abc123"},
                "artifact_retained": False,
                "artifact_deleted": False,
            }
        ],
    }


# ---------------------------------------------------------------------------
# compare_runs — pure data
# ---------------------------------------------------------------------------


def test_compare_runs_basic_fields() -> None:
    baseline = _make_run(run_id="base", status="PASS", score=0.8)
    candidate = _make_run(run_id="cand", status="PASS", score=0.7)
    cmp = compare_runs(baseline, candidate)

    assert cmp["baseline_run_id"] == "base"
    assert cmp["candidate_run_id"] == "cand"
    assert cmp["baseline_status"] == "PASS"
    assert cmp["candidate_status"] == "PASS"
    assert cmp["project"] == "test-proj"
    assert cmp["suite_name"] == "suite1"
    assert isinstance(cmp["gate_changes"], list)
    assert isinstance(cmp["metric_deltas"], list)
    assert isinstance(cmp["sample_analysis"], dict)


def test_compare_runs_gate_regression() -> None:
    baseline = _make_run(run_id="base", status="PASS", score=0.8)
    candidate = _make_run(run_id="cand", status="FAIL", score=0.3)
    cmp = compare_runs(baseline, candidate)

    regressions = cmp["gate_regressions"]
    assert len(regressions) == 1
    assert regressions[0]["change"] == "regression"
    assert regressions[0]["metric"] == "vbench_temporal.score"
    assert regressions[0]["baseline_passed"] is True
    assert regressions[0]["candidate_passed"] is False


def test_compare_runs_gate_improvement() -> None:
    baseline = _make_run(run_id="base", status="FAIL", score=0.3)
    candidate = _make_run(run_id="cand", status="PASS", score=0.8)
    cmp = compare_runs(baseline, candidate)

    improvements = cmp["gate_improvements"]
    assert len(improvements) == 1
    assert improvements[0]["change"] == "improvement"


def test_compare_runs_gate_unchanged() -> None:
    baseline = _make_run(run_id="base", status="PASS", score=0.8)
    candidate = _make_run(run_id="cand", status="PASS", score=0.9)
    cmp = compare_runs(baseline, candidate)

    changes = cmp["gate_changes"]
    assert len(changes) == 1
    assert changes[0]["change"] == "unchanged"
    assert not cmp["gate_regressions"]
    assert not cmp["gate_improvements"]


def test_compare_runs_new_gate() -> None:
    """A gate only in candidate is classified as 'new'."""
    baseline: dict[str, Any] = _make_run(run_id="base")
    baseline["gates"] = []  # no gates in baseline
    candidate = _make_run(run_id="cand")
    cmp = compare_runs(baseline, candidate)

    changes = cmp["gate_changes"]
    assert len(changes) == 1
    assert changes[0]["change"] == "new"


def test_compare_runs_metric_deltas_sorted_by_abs_delta() -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    cmp = compare_runs(baseline, candidate)

    deltas = cmp["metric_deltas"]
    abs_deltas = [abs(d["delta"]) for d in deltas if d["delta"] is not None]
    assert abs_deltas == sorted(abs_deltas, reverse=True)


def test_compare_runs_metric_delta_sign() -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    cmp = compare_runs(baseline, candidate)

    # Find the aggregate score delta (should be negative: 0.6 - 0.8 = -0.2)
    score_entry = next(
        (d for d in cmp["metric_deltas"] if d["metric"] == "vbench_temporal.score"),
        None,
    )
    assert score_entry is not None
    assert score_entry["delta"] is not None
    assert score_entry["delta"] < 0


def test_compare_runs_sample_analysis_matched() -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    cmp = compare_runs(baseline, candidate)

    analysis = cmp["sample_analysis"]
    # Both payloads share sample_id "abc123"
    assert analysis["total_matched"] >= 1
    assert isinstance(analysis["worst"], list)


def test_compare_runs_no_per_sample_gives_zero_matched() -> None:
    """When metrics have no per_sample rows, sample_analysis reports zero matches."""
    baseline = _make_run(samples=[])
    candidate = _make_run(samples=[])
    # Clear per_sample from metrics so there are no sample-level rows to match
    for run in (baseline, candidate):
        for mdata in run["metrics"].values():
            mdata["per_sample"] = []
    cmp = compare_runs(baseline, candidate)

    assert cmp["sample_analysis"]["total_matched"] == 0


# ---------------------------------------------------------------------------
# write_compare_report — HTML output
# ---------------------------------------------------------------------------


def test_write_compare_report_creates_file(tmp_path: Path) -> None:
    baseline = _make_run(run_id="base", status="PASS", score=0.8)
    candidate = _make_run(run_id="cand", status="FAIL", score=0.3)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)

    assert out.exists()
    assert "<!doctype html>" in out.read_text(encoding="utf-8")


def test_write_compare_report_shows_run_ids(tmp_path: Path) -> None:
    baseline = _make_run(run_id="baseline-001", status="PASS", score=0.8)
    candidate = _make_run(run_id="candidate-002", status="FAIL", score=0.3)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)
    content = out.read_text(encoding="utf-8")

    assert "baseline-001" in content
    assert "candidate-002" in content


def test_write_compare_report_shows_regression_tag(tmp_path: Path) -> None:
    baseline = _make_run(run_id="base", status="PASS", score=0.8)
    candidate = _make_run(run_id="cand", status="FAIL", score=0.3)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)

    assert "REGRESSED" in out.read_text(encoding="utf-8")


def test_write_compare_report_shows_improvement_tag(tmp_path: Path) -> None:
    baseline = _make_run(run_id="base", status="FAIL", score=0.3)
    candidate = _make_run(run_id="cand", status="PASS", score=0.8)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)

    assert "IMPROVED" in out.read_text(encoding="utf-8")


def test_write_compare_report_escapes_xss(tmp_path: Path) -> None:
    baseline = _make_run(run_id='<script>alert("xss")</script>')
    candidate = _make_run(run_id="cand")
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)
    content = out.read_text(encoding="utf-8")

    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_write_compare_report_creates_parent_dirs(tmp_path: Path) -> None:
    baseline = _make_run()
    candidate = _make_run()
    out = tmp_path / "nested" / "deep" / "compare.html"
    write_compare_report(out, baseline, candidate)

    assert out.exists()


def test_write_compare_report_no_samples_no_crash(tmp_path: Path) -> None:
    baseline = _make_run(samples=[])
    candidate = _make_run(samples=[])
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)

    assert "<!doctype html>" in out.read_text(encoding="utf-8")


def test_write_compare_report_metric_section_present(tmp_path: Path) -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)
    content = out.read_text(encoding="utf-8")

    assert "Metric Deltas" in content
    assert "Gate Changes" in content


def test_write_compare_report_sprt_section_no_crash_without_sprt(tmp_path: Path) -> None:
    """SPRT comparison section renders without crashing when no SPRT gates."""
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)
    content = out.read_text(encoding="utf-8")

    assert "SPRT Comparison" in content


def test_write_compare_report_per_sample_prompt_shown(tmp_path: Path) -> None:
    """Prompt text from shared sample_id appears in per-sample analysis."""
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.5)
    out = tmp_path / "compare.html"
    write_compare_report(out, baseline, candidate)
    content = out.read_text(encoding="utf-8")

    assert "a calm city timelapse" in content


def test_write_compare_report_returns_cmp_dict(tmp_path: Path) -> None:
    """write_compare_report returns the comparison data dict."""
    baseline = _make_run(run_id="r1", score=0.8)
    candidate = _make_run(run_id="r2", score=0.6)
    out = tmp_path / "compare.html"
    result = write_compare_report(out, baseline, candidate)
    assert isinstance(result, dict)
    assert result["baseline_run_id"] == "r1"
    assert result["candidate_run_id"] == "r2"
    assert "gate_changes" in result
    assert "metric_deltas" in result


# ---------------------------------------------------------------------------
# format_compare_text
# ---------------------------------------------------------------------------


def test_format_compare_text_shows_run_ids() -> None:
    baseline = _make_run(run_id="baseline_run", score=0.8)
    candidate = _make_run(run_id="candidate_run", score=0.6)
    cmp = compare_runs(baseline, candidate)
    text = format_compare_text(cmp)
    assert "baseline_run" in text
    assert "candidate_run" in text


def test_format_compare_text_shows_gate_change() -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.4, status="FAIL")
    # Manually add gate results
    baseline["gates"] = [
        {"metric": "vbench_temporal.score", "op": ">=", "value": 0.7, "passed": True, "actual": 0.8}
    ]
    candidate["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.7,
            "passed": False,
            "actual": 0.4,
        }
    ]
    cmp = compare_runs(baseline, candidate)
    text = format_compare_text(cmp)
    assert "REGRESSED" in text
    assert "vbench_temporal.score" in text


def test_format_compare_text_shows_metric_deltas() -> None:
    baseline = _make_run(run_id="b", score=0.9)
    candidate = _make_run(run_id="c", score=0.7)
    cmp = compare_runs(baseline, candidate)
    text = format_compare_text(cmp)
    # metric deltas section should appear
    assert "metric deltas" in text
    assert "→" in text


def test_format_compare_text_improvement_tag() -> None:
    baseline = _make_run(score=0.5)
    candidate = _make_run(score=0.8)
    baseline["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.7,
            "passed": False,
            "actual": 0.5,
        }
    ]
    candidate["gates"] = [
        {"metric": "vbench_temporal.score", "op": ">=", "value": 0.7, "passed": True, "actual": 0.8}
    ]
    cmp = compare_runs(baseline, candidate)
    text = format_compare_text(cmp)
    assert "IMPROVED" in text


def test_format_compare_text_no_gates_shows_metric_only() -> None:
    baseline = _make_run(score=0.8)
    candidate = _make_run(score=0.6)
    baseline["gates"] = []
    candidate["gates"] = []
    cmp = compare_runs(baseline, candidate)
    text = format_compare_text(cmp)
    # gates section absent but metric deltas still shown
    assert "REGRESSED" not in text
    assert "metric deltas" in text
