"""Tests for the temporal diff module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.temporal_diff import temporal_diff, write_temporal_diff_report


def _make_result(
    frames: list[tuple[str, float, str]],
) -> dict[str, Any]:
    """Build a mock metric result.

    Each frame is (prompt, risk_score, risk_level).
    """
    per_sample: list[dict[str, Any]] = []
    for prompt, risk, level in frames:
        per_sample.append(
            {
                "prompt": prompt,
                "risk_level": level,
                "dims": {
                    "risk_score": risk,
                    "vegetation_proximity_nn": 1.0 - risk,
                    "vegetation_penetration": max(0.0, 1.0 - risk - 0.1),
                },
            }
        )
    return {
        "score": sum(f[1] for f in frames) / max(len(frames), 1),
        "per_sample": per_sample,
    }


class TestTemporalDiff:
    def test_matching_frames(self) -> None:
        before = _make_result(
            [
                ("frame_001", 0.8, "safe"),
                ("frame_002", 0.7, "caution"),
                ("frame_003", 0.6, "caution"),
            ]
        )
        after = _make_result(
            [
                ("frame_001", 0.6, "caution"),
                ("frame_002", 0.3, "warning"),
                ("frame_003", 0.9, "safe"),
            ]
        )
        diff = temporal_diff(before, after)
        assert diff["matched_count"] == 3
        assert len(diff["frames"]) == 3

    def test_degradation_detected(self) -> None:
        before = _make_result([("f1", 0.9, "safe")])
        after = _make_result([("f1", 0.3, "warning")])
        diff = temporal_diff(before, after)
        assert diff["frames"][0]["risk_delta"] < 0
        assert diff["summary"]["degraded_count"] == 1

    def test_improvement_detected(self) -> None:
        before = _make_result([("f1", 0.3, "warning")])
        after = _make_result([("f1", 0.9, "safe")])
        diff = temporal_diff(before, after)
        assert diff["frames"][0]["risk_delta"] > 0
        assert diff["summary"]["improved_count"] == 1

    def test_hotspots(self) -> None:
        before = _make_result(
            [
                ("f1", 0.9, "safe"),
                ("f2", 0.8, "safe"),
            ]
        )
        after = _make_result(
            [
                ("f1", 0.5, "caution"),  # delta = -0.4, hotspot
                ("f2", 0.75, "caution"),  # delta = -0.05, not hotspot
            ]
        )
        diff = temporal_diff(before, after)
        assert len(diff["hotspots"]) == 1
        assert diff["hotspots"][0]["prompt"] == "f1"

    def test_unmatched_frames_counted(self) -> None:
        before = _make_result([("f1", 0.8, "safe"), ("f_only_before", 0.5, "caution")])
        after = _make_result([("f1", 0.7, "caution"), ("f_only_after", 0.4, "warning")])
        diff = temporal_diff(before, after)
        assert diff["matched_count"] == 1
        assert diff["summary"]["unmatched_before"] == 1
        assert diff["summary"]["unmatched_after"] == 1

    def test_empty_inputs(self) -> None:
        diff = temporal_diff({"per_sample": []}, {"per_sample": []})
        assert diff["matched_count"] == 0
        assert diff["frames"] == []
        assert diff["hotspots"] == []

    def test_dim_deltas_computed(self) -> None:
        before = _make_result([("f1", 0.8, "safe")])
        after = _make_result([("f1", 0.5, "caution")])
        diff = temporal_diff(before, after)
        deltas = diff["frames"][0]["dim_deltas"]
        assert "risk_score" in deltas
        assert "vegetation_proximity_nn" in deltas
        # risk_score delta should be negative (0.5 - 0.8)
        assert deltas["risk_score"] < 0

    def test_summary_statistics(self) -> None:
        before = _make_result(
            [
                ("f1", 0.9, "safe"),
                ("f2", 0.5, "caution"),
                ("f3", 0.3, "warning"),
            ]
        )
        after = _make_result(
            [
                ("f1", 0.4, "warning"),  # degraded
                ("f2", 0.5, "caution"),  # stable
                ("f3", 0.8, "safe"),  # improved
            ]
        )
        diff = temporal_diff(before, after)
        s = diff["summary"]
        assert s["degraded_count"] == 1
        assert s["improved_count"] == 1
        assert s["stable_count"] == 1
        assert "dim_summary" in s

    def test_frames_sorted_by_risk_delta(self) -> None:
        before = _make_result(
            [
                ("f1", 0.9, "safe"),
                ("f2", 0.3, "warning"),
                ("f3", 0.7, "caution"),
            ]
        )
        after = _make_result(
            [
                ("f1", 0.2, "critical"),  # worst: -0.7
                ("f2", 0.8, "safe"),  # best:  +0.5
                ("f3", 0.5, "caution"),  # mid:   -0.2
            ]
        )
        diff = temporal_diff(before, after)
        # Sorted by risk_delta ascending (worst first)
        assert diff["frames"][0]["prompt"] == "f1"
        assert diff["frames"][-1]["prompt"] == "f2"


class TestTemporalDiffReport:
    def test_writes_html(self, tmp_path: Path) -> None:
        before = _make_result(
            [
                ("f1", 0.8, "safe"),
                ("f2", 0.6, "caution"),
            ]
        )
        after = _make_result(
            [
                ("f1", 0.5, "caution"),
                ("f2", 0.7, "caution"),
            ]
        )
        out = tmp_path / "report.html"
        diff = write_temporal_diff_report(out, before, after)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Temporal Vegetation Change Report" in content
        assert "f1" in content
        assert "f2" in content
        assert diff["matched_count"] == 2

    def test_custom_labels(self, tmp_path: Path) -> None:
        before = _make_result([("f1", 0.8, "safe")])
        after = _make_result([("f1", 0.5, "caution")])
        out = tmp_path / "report.html"
        write_temporal_diff_report(
            out,
            before,
            after,
            before_label="2025-06 Summer",
            after_label="2025-12 Winter",
        )
        content = out.read_text(encoding="utf-8")
        assert "2025-06 Summer" in content
        assert "2025-12 Winter" in content

    def test_empty_report(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.html"
        diff = write_temporal_diff_report(out, {"per_sample": []}, {"per_sample": []})
        assert out.exists()
        assert diff["matched_count"] == 0
