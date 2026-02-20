"""Tests for the multi-camera fusion engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.fusion import (
    DEFAULT_CAMERA_WEIGHTS,
    CameraWeight,
    FusionResult,
    aggregate_by_km,
    fuse_cameras,
    generate_km_report,
    prioritize_maintenance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cam_result(
    risk_score: float = 0.5,
    vegetation: float = 0.0,
    equipment: float = 0.0,
    visibility: float = 0.0,
) -> dict[str, Any]:
    """Build a minimal per-camera metric result."""
    return {
        "risk_score": risk_score,
        "dims": {
            "risk_score": risk_score,
            "vegetation_proximity_nn": vegetation,
            "equipment_score": equipment,
            "visibility_score": visibility,
        },
    }


def _sample(
    prompt: str,
    km: float | None = None,
    risk_score: float = 0.5,
    risk_level: str | None = None,
) -> dict[str, Any]:
    """Build a minimal per-sample dict."""
    entry: dict[str, Any] = {
        "prompt": prompt,
        "risk_score": risk_score,
    }
    if km is not None:
        entry["km"] = km
    if risk_level is not None:
        entry["risk_level"] = risk_level
    return entry


# ===========================================================================
# fuse_cameras
# ===========================================================================


class TestFuseCameras:
    def test_empty_input(self) -> None:
        result = fuse_cameras({})
        assert result["camera_count"] == 0
        assert result["risk_score"] == 0.0
        assert result["risk_level"] == "critical"
        assert result["cameras"] == {}

    def test_single_camera(self) -> None:
        result = fuse_cameras({"up": _cam_result(risk_score=0.9)})
        assert result["camera_count"] == 1
        assert result["risk_score"] == 0.9
        assert "up" in result["cameras"]

    def test_multiple_cameras_default_weights(self) -> None:
        results_by_camera = {
            "up": _cam_result(risk_score=0.8, vegetation=0.3),
            "left": _cam_result(risk_score=0.6, vegetation=0.5),
            "right": _cam_result(risk_score=0.7, vegetation=0.4),
            "front": _cam_result(risk_score=0.5, vegetation=0.6),
            "back": _cam_result(risk_score=0.9, vegetation=0.1),
            "down": _cam_result(risk_score=0.4, vegetation=0.2),
        }
        result = fuse_cameras(results_by_camera)

        assert result["camera_count"] == 6
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_level"] in ("safe", "caution", "warning", "critical")
        assert len(result["cameras"]) == 6
        # Each camera should have a weight breakdown
        for cam_name, cam_data in result["cameras"].items():
            assert "risk_score" in cam_data
            assert "weight" in cam_data

    def test_custom_weights(self) -> None:
        custom_weights = {
            "up": CameraWeight(
                "up", vegetation_weight=1.0, equipment_weight=0.0, visibility_weight=0.0
            ),
            "front": CameraWeight(
                "front", vegetation_weight=0.0, equipment_weight=1.0, visibility_weight=0.0
            ),
        }
        results_by_camera = {
            "up": _cam_result(risk_score=0.3, vegetation=0.8),
            "front": _cam_result(risk_score=0.9, equipment=0.7),
        }
        result = fuse_cameras(results_by_camera, weights=custom_weights)
        assert result["camera_count"] == 2
        # With these extreme weights the vegetation_score should be dominated by "up"
        assert result["vegetation_score"] == 0.8

    def test_unknown_camera_uses_zero_weights(self) -> None:
        result = fuse_cameras({"unknown_cam": _cam_result(risk_score=0.6)})
        assert result["camera_count"] == 1
        # Should still produce a valid result even without default weight entry
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_level_safe(self) -> None:
        result = fuse_cameras({"up": _cam_result(risk_score=0.95)})
        assert result["risk_level"] == "safe"

    def test_risk_level_caution(self) -> None:
        result = fuse_cameras({"up": _cam_result(risk_score=0.7)})
        assert result["risk_level"] == "caution"

    def test_risk_level_warning(self) -> None:
        result = fuse_cameras({"up": _cam_result(risk_score=0.5)})
        assert result["risk_level"] == "warning"

    def test_risk_level_critical(self) -> None:
        result = fuse_cameras({"up": _cam_result(risk_score=0.2)})
        assert result["risk_level"] == "critical"

    def test_result_includes_per_camera_breakdown(self) -> None:
        result = fuse_cameras(
            {
                "left": _cam_result(risk_score=0.6, vegetation=0.4),
                "right": _cam_result(risk_score=0.7, vegetation=0.3),
            }
        )
        assert "left" in result["cameras"]
        assert "right" in result["cameras"]
        assert result["cameras"]["left"]["risk_score"] == 0.6
        assert result["cameras"]["right"]["vegetation_score"] == 0.3

    def test_dims_fallback(self) -> None:
        """When top-level keys are missing, values are read from dims."""
        result = fuse_cameras(
            {
                "up": {
                    "dims": {
                        "risk_score": 0.75,
                        "vegetation_proximity_nn": 0.3,
                        "equipment_score": 0.5,
                        "visibility_score": 0.6,
                    },
                },
            }
        )
        assert result["cameras"]["up"]["risk_score"] == 0.75
        assert result["cameras"]["up"]["vegetation_score"] == 0.3

    def test_non_finite_values_use_default(self) -> None:
        result = fuse_cameras(
            {
                "up": {"risk_score": float("nan")},
            }
        )
        # NaN should fall back to default (0.5)
        assert result["cameras"]["up"]["risk_score"] == 0.5

    def test_bool_values_use_default(self) -> None:
        """Booleans should not be treated as numeric."""
        result = fuse_cameras(
            {
                "up": {"risk_score": True},
            }
        )
        # True should not be treated as 1.0
        assert result["cameras"]["up"]["risk_score"] == 0.5


# ===========================================================================
# aggregate_by_km
# ===========================================================================


class TestAggregateByKm:
    def test_empty_input(self) -> None:
        assert aggregate_by_km([]) == []

    def test_no_km_data(self) -> None:
        samples = [_sample("f1"), _sample("f2")]
        assert aggregate_by_km(samples) == []

    def test_single_bin(self) -> None:
        samples = [
            _sample("f1", km=0.1, risk_score=0.8),
            _sample("f2", km=0.3, risk_score=0.6),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        assert len(bins) == 1
        assert bins[0]["km_start"] == 0.0
        assert bins[0]["km_end"] == 0.5
        assert bins[0]["frame_count"] == 2
        assert bins[0]["worst_frame"] == "f2"  # lower risk = worse

    def test_multiple_bins(self) -> None:
        samples = [
            _sample("f1", km=0.1, risk_score=0.8),
            _sample("f2", km=0.7, risk_score=0.3),
            _sample("f3", km=1.2, risk_score=0.9),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        assert len(bins) == 3
        # Should be sorted by km_start
        assert bins[0]["km_start"] < bins[1]["km_start"] < bins[2]["km_start"]

    def test_bin_risk_aggregation(self) -> None:
        samples = [
            _sample("f1", km=0.1, risk_score=0.8),
            _sample("f2", km=0.2, risk_score=0.4),
            _sample("f3", km=0.3, risk_score=0.6),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        assert len(bins) == 1
        km_bin = bins[0]
        assert km_bin["min_risk"] == 0.4
        assert km_bin["max_risk"] == 0.8
        assert abs(km_bin["avg_risk"] - 0.6) < 0.001

    def test_custom_bin_size(self) -> None:
        samples = [
            _sample("f1", km=0.1, risk_score=0.5),
            _sample("f2", km=0.15, risk_score=0.5),
            _sample("f3", km=0.25, risk_score=0.5),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.1)
        # 0.1 -> bin 1 (0.1-0.2), 0.15 -> bin 1 (0.1-0.2), 0.25 -> bin 2 (0.2-0.3)
        assert len(bins) == 2

    def test_mixed_km_and_no_km(self) -> None:
        samples = [
            _sample("f1", km=1.0, risk_score=0.5),
            _sample("f2", risk_score=0.8),  # no km
            _sample("f3", km=1.3, risk_score=0.7),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        total_frames = sum(b["frame_count"] for b in bins)
        assert total_frames == 2  # f2 skipped

    def test_frames_include_risk_level(self) -> None:
        samples = [
            _sample("f1", km=0.1, risk_score=0.2, risk_level="critical"),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        frame = bins[0]["frames"][0]
        assert frame["risk_level"] == "critical"

    def test_frames_auto_classify_risk_level(self) -> None:
        """When risk_level is absent it should be auto-classified from score."""
        samples = [
            _sample("f1", km=0.1, risk_score=0.9),
        ]
        bins = aggregate_by_km(samples, bin_size_km=0.5)
        frame = bins[0]["frames"][0]
        assert frame["risk_level"] == "safe"

    def test_negative_bin_size_uses_default(self) -> None:
        samples = [_sample("f1", km=0.1, risk_score=0.5)]
        bins = aggregate_by_km(samples, bin_size_km=-1.0)
        assert len(bins) == 1

    def test_nan_km_skipped(self) -> None:
        samples = [{"prompt": "f1", "km": float("nan"), "risk_score": 0.5}]
        assert aggregate_by_km(samples) == []


# ===========================================================================
# prioritize_maintenance
# ===========================================================================


class TestPrioritizeMaintenance:
    def test_empty_bins(self) -> None:
        assert prioritize_maintenance([]) == []

    def test_budget_selects_worst_first(self) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.3,
                "worst_frame": "f1",
                "frame_count": 2,
                "min_risk": 0.2,
                "max_risk": 0.4,
                "frames": [],
            },
            {
                "km_start": 0.5,
                "km_end": 1.0,
                "avg_risk": 0.9,
                "worst_frame": "f2",
                "frame_count": 1,
                "min_risk": 0.9,
                "max_risk": 0.9,
                "frames": [],
            },
            {
                "km_start": 1.0,
                "km_end": 1.5,
                "avg_risk": 0.5,
                "worst_frame": "f3",
                "frame_count": 3,
                "min_risk": 0.4,
                "max_risk": 0.6,
                "frames": [],
            },
        ]
        selected = prioritize_maintenance(km_bins, budget_km=1.0)
        # Budget = 1.0 km, each bin is 0.5 km -> selects 2 worst bins
        assert len(selected) == 2
        # First selected should be the worst (lowest risk)
        assert selected[0]["avg_risk"] == 0.3
        assert selected[1]["avg_risk"] == 0.5

    def test_budget_not_exceeded(self) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 1.0,
                "avg_risk": 0.2,
                "worst_frame": "f1",
                "frame_count": 5,
                "min_risk": 0.1,
                "max_risk": 0.3,
                "frames": [],
            },
            {
                "km_start": 1.0,
                "km_end": 2.0,
                "avg_risk": 0.3,
                "worst_frame": "f2",
                "frame_count": 3,
                "min_risk": 0.2,
                "max_risk": 0.4,
                "frames": [],
            },
        ]
        selected = prioritize_maintenance(km_bins, budget_km=1.5)
        # Only first bin fits within budget
        assert len(selected) == 1

    def test_urgency_labels(self) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.2,
                "worst_frame": "f1",
                "frame_count": 1,
                "min_risk": 0.2,
                "max_risk": 0.2,
                "frames": [],
            },
            {
                "km_start": 0.5,
                "km_end": 1.0,
                "avg_risk": 0.5,
                "worst_frame": "f2",
                "frame_count": 1,
                "min_risk": 0.5,
                "max_risk": 0.5,
                "frames": [],
            },
            {
                "km_start": 1.0,
                "km_end": 1.5,
                "avg_risk": 0.7,
                "worst_frame": "f3",
                "frame_count": 1,
                "min_risk": 0.7,
                "max_risk": 0.7,
                "frames": [],
            },
            {
                "km_start": 1.5,
                "km_end": 2.0,
                "avg_risk": 0.9,
                "worst_frame": "f4",
                "frame_count": 1,
                "min_risk": 0.9,
                "max_risk": 0.9,
                "frames": [],
            },
        ]
        selected = prioritize_maintenance(km_bins, budget_km=10.0)
        urgencies = [s["urgency"] for s in selected]
        assert urgencies == ["critical", "high", "medium", "low"]

    def test_cumulative_km(self) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.2,
                "worst_frame": "f1",
                "frame_count": 1,
                "min_risk": 0.2,
                "max_risk": 0.2,
                "frames": [],
            },
            {
                "km_start": 0.5,
                "km_end": 1.0,
                "avg_risk": 0.4,
                "worst_frame": "f2",
                "frame_count": 1,
                "min_risk": 0.4,
                "max_risk": 0.4,
                "frames": [],
            },
        ]
        selected = prioritize_maintenance(km_bins, budget_km=5.0)
        assert selected[0]["cumulative_km"] == 0.5
        assert selected[1]["cumulative_km"] == 1.0

    def test_large_budget_selects_all(self) -> None:
        km_bins = [
            {
                "km_start": float(i),
                "km_end": float(i + 1),
                "avg_risk": 0.5,
                "worst_frame": f"f{i}",
                "frame_count": 1,
                "min_risk": 0.5,
                "max_risk": 0.5,
                "frames": [],
            }
            for i in range(5)
        ]
        selected = prioritize_maintenance(km_bins, budget_km=100.0)
        assert len(selected) == 5

    def test_zero_length_bins_skipped(self) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.0,
                "avg_risk": 0.1,
                "worst_frame": "f1",
                "frame_count": 1,
                "min_risk": 0.1,
                "max_risk": 0.1,
                "frames": [],
            },
        ]
        selected = prioritize_maintenance(km_bins, budget_km=5.0)
        assert len(selected) == 0


# ===========================================================================
# generate_km_report
# ===========================================================================


class TestGenerateKmReport:
    def test_empty_bins_generates_placeholder(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        result_path = generate_km_report([], out)
        assert result_path == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "No km data" in content

    def test_report_contains_heatmap(self, tmp_path: Path) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.3,
                "min_risk": 0.2,
                "max_risk": 0.4,
                "worst_frame": "f1",
                "frame_count": 2,
                "frames": [
                    {"prompt": "f1", "km": 0.1, "risk_score": 0.2, "risk_level": "critical"},
                    {"prompt": "f2", "km": 0.3, "risk_score": 0.4, "risk_level": "warning"},
                ],
            },
            {
                "km_start": 0.5,
                "km_end": 1.0,
                "avg_risk": 0.8,
                "min_risk": 0.7,
                "max_risk": 0.9,
                "worst_frame": "f3",
                "frame_count": 1,
                "frames": [
                    {"prompt": "f3", "km": 0.7, "risk_score": 0.8, "risk_level": "safe"},
                ],
            },
        ]
        out = tmp_path / "report.html"
        generate_km_report(km_bins, out)
        content = out.read_text(encoding="utf-8")
        assert "heatmap" in content.lower()
        assert "Priority Maintenance" in content
        assert "f1" in content
        assert "f3" in content

    def test_report_contains_summary_stats(self, tmp_path: Path) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 1.0,
                "avg_risk": 0.5,
                "min_risk": 0.3,
                "max_risk": 0.7,
                "worst_frame": "f1",
                "frame_count": 5,
                "frames": [],
            },
        ]
        out = tmp_path / "report.html"
        generate_km_report(km_bins, out)
        content = out.read_text(encoding="utf-8")
        assert "segments" in content
        assert "frames" in content
        assert "avg risk" in content

    def test_custom_title(self, tmp_path: Path) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.5,
                "min_risk": 0.5,
                "max_risk": 0.5,
                "worst_frame": "f1",
                "frame_count": 1,
                "frames": [],
            },
        ]
        out = tmp_path / "report.html"
        generate_km_report(km_bins, out, title="JR East Inspection Report")
        content = out.read_text(encoding="utf-8")
        assert "JR East Inspection Report" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "report.html"
        generate_km_report([], out)
        assert out.exists()

    def test_report_valid_html(self, tmp_path: Path) -> None:
        km_bins = [
            {
                "km_start": 0.0,
                "km_end": 0.5,
                "avg_risk": 0.5,
                "min_risk": 0.5,
                "max_risk": 0.5,
                "worst_frame": "f1",
                "frame_count": 1,
                "frames": [],
            },
        ]
        out = tmp_path / "report.html"
        generate_km_report(km_bins, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!doctype html>")
        assert "</html>" in content


# ===========================================================================
# Data structures
# ===========================================================================


class TestDataStructures:
    def test_camera_weight_defaults(self) -> None:
        cw = CameraWeight("test")
        assert cw.position == "test"
        assert cw.vegetation_weight == 0.0
        assert cw.equipment_weight == 0.0
        assert cw.visibility_weight == 0.0

    def test_camera_weight_frozen(self) -> None:
        cw = CameraWeight("up", vegetation_weight=0.3)
        try:
            cw.vegetation_weight = 0.5  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_default_weights_structure(self) -> None:
        assert set(DEFAULT_CAMERA_WEIGHTS.keys()) == {
            "up",
            "left",
            "right",
            "front",
            "back",
            "down",
        }
        for name, cw in DEFAULT_CAMERA_WEIGHTS.items():
            assert cw.position == name
            assert 0.0 <= cw.vegetation_weight <= 1.0
            assert 0.0 <= cw.equipment_weight <= 1.0
            assert 0.0 <= cw.visibility_weight <= 1.0

    def test_fusion_result_creation(self) -> None:
        fr = FusionResult(
            location_id="loc_001",
            fused_risk_score=0.65,
            camera_count=3,
            risk_level="caution",
        )
        assert fr.location_id == "loc_001"
        assert fr.fused_risk_score == 0.65
        assert fr.camera_count == 3
        assert fr.camera_breakdown == {}

    def test_fusion_result_defaults(self) -> None:
        fr = FusionResult(location_id="x", fused_risk_score=0.5)
        assert fr.fused_vegetation_score == 0.0
        assert fr.fused_equipment_score == 0.0
        assert fr.fused_visibility_score == 0.0
        assert fr.camera_count == 0
        assert fr.risk_level == "unknown"
        assert fr.metadata == {}


# ===========================================================================
# Integration: aggregate + report pipeline
# ===========================================================================


class TestIntegration:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Run the full pipeline: samples -> aggregate -> prioritize -> report."""
        samples = [
            _sample("f1", km=0.1, risk_score=0.2, risk_level="critical"),
            _sample("f2", km=0.3, risk_score=0.4, risk_level="warning"),
            _sample("f3", km=0.7, risk_score=0.9, risk_level="safe"),
            _sample("f4", km=1.1, risk_score=0.5, risk_level="caution"),
            _sample("f5", km=1.4, risk_score=0.3, risk_level="warning"),
            _sample("f6", km=2.0, risk_score=0.85, risk_level="safe"),
        ]

        km_bins = aggregate_by_km(samples, bin_size_km=0.5)
        assert len(km_bins) >= 3

        priority = prioritize_maintenance(km_bins, budget_km=1.5)
        assert len(priority) >= 1
        # Worst bin should be selected first
        assert priority[0]["avg_risk"] <= priority[-1]["avg_risk"]

        out = tmp_path / "full_report.html"
        generate_km_report(km_bins, out, title="Full Pipeline Test")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "f1" in content
        assert "Full Pipeline Test" in content

    def test_fusion_then_aggregate(self, tmp_path: Path) -> None:
        """Fuse cameras then aggregate fused results by km."""
        # Simulate fused results for multiple locations
        fused_samples = []
        for i in range(10):
            fused = fuse_cameras(
                {
                    "up": _cam_result(risk_score=0.3 + i * 0.05),
                    "front": _cam_result(risk_score=0.5 + i * 0.03),
                }
            )
            fused_samples.append(
                {
                    "prompt": f"loc_{i:03d}",
                    "km": i * 0.2,
                    "risk_score": fused["risk_score"],
                    "risk_level": fused["risk_level"],
                }
            )

        km_bins = aggregate_by_km(fused_samples, bin_size_km=0.5)
        assert len(km_bins) > 0

        out = tmp_path / "fused_report.html"
        generate_km_report(km_bins, out)
        assert out.exists()
