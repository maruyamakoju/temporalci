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
        "metrics": {"vbench_temporal": {"score": 0.8}},
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.5, "actual": 0.8, "passed": True}],
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
    assert "#a52222" in content  # fail color


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
        {"test_id": "t1", "seed": 0, "prompt": "a cat", "video_path": "/path/v.mp4"}
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


def test_html_report_renders_sprt_analysis_details(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["gates"] = [
        {
            "metric": "vbench_temporal.dims.motion_smoothness",
            "op": ">=",
            "value": 0.45,
            "actual": 0.52,
            "passed": True,
            "sprt": {
                "decision": "accept_h1_no_regression",
                "decision_passed": True,
                "pairing_mismatch_policy": "fail",
                "sigma_mode": "fixed",
                "sigma": 0.04,
                "llr": 3.2,
                "upper_threshold": 2.89,
                "lower_threshold": -2.25,
                "crossed_at": 11,
                "min_paired_ratio": 1.0,
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
    assert "pairing_mismatch_policy" in content
    assert "pairing.current_series_count" in content
    assert "worst_deltas" in content
    assert "paired_ratio" in content
