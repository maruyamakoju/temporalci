from __future__ import annotations

from pathlib import Path

from temporalci.report import write_html_report


def _minimal_payload() -> dict:
    return {
        "run_id": "20260212T000000Z",
        "project": "test-project",
        "suite_name": "suite1",
        "model_name": "mock",
        "timestamp_utc": "2026-02-12T00:00:00+00:00",
        "baseline_run_id": None,
        "baseline_mode": "latest_pass",
        "artifacts_policy": {},
        "status": "PASS",
        "sample_count": 0,
        "metrics": {"vbench_temporal": {"score": 0.8}},
        "gates": [
            {
                "metric": "vbench_temporal.score",
                "op": ">=",
                "value": 0.5,
                "actual": 0.8,
                "passed": True,
            }
        ],
        "regressions": [],
        "samples": [],
    }


def test_write_html_report_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html_report(path, _minimal_payload())
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


def test_html_report_contains_status(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html_report(path, _minimal_payload())
    content = path.read_text(encoding="utf-8")
    assert "PASS" in content
    assert "test-project" in content
    assert "20260212T000000Z" in content


def test_html_report_fail_status_color(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["status"] = "FAIL"
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "badge-fail" in content


def test_html_report_escapes_xss(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["project"] = '<script>alert("xss")</script>'
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_html_report_with_regressions(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["regressions"] = [
        {
            "metric": "score",
            "baseline": 0.8,
            "current": 0.6,
            "delta": -0.2,
            "direction": "higher_is_better",
            "regressed": True,
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "REGRESSED" in content


def test_html_report_with_samples(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["samples"] = [
        {
            "test_id": "t1",
            "seed": 0,
            "prompt": "a cat",
            "video_path": "/path/v.mp4",
            "artifact_retained": True,
            "artifact_deleted": False,
            "metadata": {"sample_id": "abc123"},
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "a cat" in content


def test_html_report_empty_metrics(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["metrics"] = {}
    payload["gates"] = []
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


def test_html_report_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "report.html"
    write_html_report(path, _minimal_payload())
    assert path.exists()


def test_html_report_renders_sprt_analysis(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.dims.motion_smoothness",
            "op": ">=",
            "value": 0.45,
            "actual": 0.52,
            "passed": True,
            "method": "sprt_regression",
            "sprt": {
                "decision": "accept_h1_no_regression",
                "decision_passed": True,
                "sigma_mode": "fixed",
                "sigma": 0.04,
                "effect_size": 0.02,
                "llr": 3.2,
                "upper_threshold": 2.89,
                "lower_threshold": -2.25,
                "crossed_at": 11,
                "min_pairs": 6,
                "min_paired_ratio": 1.0,
                "alpha": 0.05,
                "beta": 0.1,
                "llr_history": [0.1, 0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 2.9, 3.2],
                "pairing": {
                    "paired_count": 12,
                    "paired_ratio": 1.0,
                    "expected_pairs": 12,
                    "current_series_count": 12,
                    "baseline_series_count": 12,
                    "worst_deltas": [
                        {
                            "pair_key": "sid:s1",
                            "current": 0.5,
                            "baseline": 0.49,
                            "delta": 0.01,
                        }
                    ],
                },
            },
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "SPRT Analysis" in content
    assert "accept_h1_no_regression" in content
    # stat pills show "ratio" and "paired" (not the full field names)
    assert "ratio" in content
    assert "paired" in content
    # worst deltas table is rendered
    assert "Worst Sample Pairs" in content


def test_html_report_sprt_shows_llr_chart(tmp_path: Path) -> None:
    """LLR chart SVG is rendered when llr_history is present."""
    payload = _minimal_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.dims.motion_smoothness",
            "op": ">=",
            "value": 0.45,
            "actual": 0.52,
            "passed": True,
            "method": "sprt_regression",
            "sprt": {
                "decision": "accept_h1_no_regression",
                "decision_passed": True,
                "sigma": 0.04,
                "llr": 3.2,
                "upper_threshold": 2.89,
                "lower_threshold": -2.25,
                "crossed_at": 9,
                "min_pairs": 6,
                "alpha": 0.05,
                "beta": 0.1,
                "llr_history": [0.1, 0.3, 0.6, 1.0, 1.4, 1.9, 2.4, 2.9, 3.2],
                "pairing": {"paired_count": 9, "paired_ratio": 1.0, "expected_pairs": 9},
            },
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "polyline" in content  # LLR path polyline


def test_html_report_sprt_worst_delta_shows_prompt(tmp_path: Path) -> None:
    """Worst delta table looks up sample prompt via sample_lookup."""
    payload = _minimal_payload()
    payload["samples"] = [
        {
            "test_id": "core",
            "seed": 1,
            "prompt": "a robot walking in a park",
            "video_path": None,
            "artifact_retained": False,
            "artifact_deleted": False,
            "metadata": {"sample_id": "deadbeef1234"},
        }
    ]
    payload["gates"] = [
        {
            "metric": "vbench_temporal.dims.motion_smoothness",
            "op": ">=",
            "value": 0.3,
            "actual": 0.4,
            "passed": True,
            "method": "sprt_regression",
            "sprt": {
                "decision": "accept_h1_no_regression",
                "decision_passed": True,
                "sigma": 0.04,
                "llr": 2.5,
                "upper_threshold": 2.89,
                "lower_threshold": -2.25,
                "min_pairs": 2,
                "pairing": {
                    "paired_count": 5,
                    "paired_ratio": 1.0,
                    "expected_pairs": 5,
                    "worst_deltas": [
                        {
                            "pair_key": "sid:deadbeef1234",
                            "baseline": 0.50,
                            "current": 0.40,
                            "delta": -0.10,
                        }
                    ],
                },
            },
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)
    content = path.read_text(encoding="utf-8")
    # Prompt text resolved from sample_id lookup
    assert "a robot walking in a park" in content
    # Negative delta should be colored
    assert "delta-neg" in content


def test_html_report_sprt_no_llr_history_no_crash(tmp_path: Path) -> None:
    """Report renders fine even when llr_history is absent (older format)."""
    payload = _minimal_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.score",
            "op": ">=",
            "value": 0.5,
            "actual": 0.8,
            "passed": True,
            "method": "sprt_regression",
            "sprt": {
                "decision": "accept_h1_no_regression",
                "decision_passed": True,
                "sigma": 0.04,
                "llr": 2.5,
                "upper_threshold": 2.89,
                "lower_threshold": -2.25,
                "min_pairs": 2,
                "pairing": {"paired_count": 5, "paired_ratio": 1.0, "expected_pairs": 5},
            },
        }
    ]
    path = tmp_path / "report.html"
    write_html_report(path, payload)  # must not raise
    assert path.exists()
