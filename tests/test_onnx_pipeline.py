"""Tests for the ONNX pipeline and dashboard modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.dashboard import generate_dashboard


def _make_inspection_result(n_frames: int = 5) -> dict[str, Any]:
    """Build a mock inspection result for testing."""
    per_sample: list[dict[str, Any]] = []
    alert_frames: list[dict[str, Any]] = []

    levels = ["critical", "warning", "caution", "safe", "safe"]
    scores = [0.15, 0.35, 0.55, 0.75, 0.90]

    for i in range(n_frames):
        idx = i % len(levels)
        dims = {
            "risk_score": scores[idx],
            "vegetation_proximity_nn": 1.0 - scores[idx],
            "vegetation_penetration": max(0.0, 0.8 - scores[idx]),
            "clearance_relative": scores[idx] * 0.5,
            "depth_clearance": scores[idx] * 0.8,
        }
        entry = {
            "prompt": f"frame_{i:05d}",
            "test_id": "inspect",
            "risk_level": levels[idx],
            "dims": dims,
            "wire_count": i % 3,
            "clearance_px": scores[idx] * 100,
        }
        per_sample.append(entry)
        if scores[idx] < 0.5:
            alert_frames.append(
                {
                    "prompt": f"frame_{i:05d}",
                    "risk_level": levels[idx],
                    "risk_score": scores[idx],
                    "clearance_px": scores[idx] * 100,
                    "vegetation_zone": 1.0 - scores[idx],
                }
            )

    dim_avgs = {
        "risk_score": sum(scores[:n_frames]) / n_frames,
        "vegetation_proximity_nn": 1.0 - sum(scores[:n_frames]) / n_frames,
        "vegetation_penetration": 0.3,
        "clearance_relative": 0.25,
        "depth_clearance": 0.4,
    }

    return {
        "score": sum(scores[:n_frames]) / n_frames,
        "dims": dim_avgs,
        "sample_count": n_frames,
        "per_sample": per_sample,
        "alert_frames": alert_frames,
    }


class TestDashboard:
    def test_generates_html(self, tmp_path: Path) -> None:
        result = _make_inspection_result()
        out = tmp_path / "dashboard.html"
        path = generate_dashboard(result, out)
        assert path == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Catenary Inspection Dashboard" in content
        assert "frame_00000" in content

    def test_custom_title(self, tmp_path: Path) -> None:
        result = _make_inspection_result(3)
        out = tmp_path / "dash.html"
        generate_dashboard(result, out, title="JR East Line 23")
        content = out.read_text(encoding="utf-8")
        assert "JR East Line 23" in content

    def test_with_performance_stats(self, tmp_path: Path) -> None:
        result = _make_inspection_result()
        stats = {
            "total_frames": 5,
            "total_elapsed_ms": 5000.0,
            "avg_ms_per_frame": 1000.0,
            "fps": 1.0,
            "seg_elapsed_ms": 2000.0,
            "depth_elapsed_ms": 2500.0,
            "clearance_elapsed_ms": 500.0,
            "risk_distribution": {"critical": 1, "safe": 4},
        }
        out = tmp_path / "dash.html"
        generate_dashboard(result, out, stats=stats)
        content = out.read_text(encoding="utf-8")
        assert "Performance" in content
        assert "1000ms" in content

    def test_empty_results(self, tmp_path: Path) -> None:
        result = {
            "score": 0.0,
            "dims": {},
            "sample_count": 0,
            "per_sample": [],
            "alert_frames": [],
        }
        out = tmp_path / "empty.html"
        generate_dashboard(result, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "0 frames analyzed" in content

    def test_dark_theme_styling(self, tmp_path: Path) -> None:
        result = _make_inspection_result(2)
        out = tmp_path / "dark.html"
        generate_dashboard(result, out)
        content = out.read_text(encoding="utf-8")
        # Dark background color
        assert "#0f172a" in content

    def test_risk_distribution_bar(self, tmp_path: Path) -> None:
        result = _make_inspection_result(5)
        out = tmp_path / "dist.html"
        generate_dashboard(result, out)
        content = out.read_text(encoding="utf-8")
        # Should contain risk colors
        assert "#ef4444" in content  # critical red
        assert "#22c55e" in content  # safe green

    def test_all_critical(self, tmp_path: Path) -> None:
        per_sample = [
            {
                "prompt": f"f{i}",
                "risk_level": "critical",
                "dims": {"risk_score": 0.1},
                "wire_count": 0,
                "clearance_px": 5.0,
            }
            for i in range(3)
        ]
        result = {
            "score": 0.1,
            "dims": {"risk_score": 0.1},
            "per_sample": per_sample,
            "alert_frames": [
                {
                    "prompt": "f0",
                    "risk_level": "critical",
                    "risk_score": 0.1,
                    "clearance_px": 5.0,
                    "vegetation_zone": 0.9,
                }
            ],
        }
        out = tmp_path / "critical.html"
        generate_dashboard(result, out)
        content = out.read_text(encoding="utf-8")
        assert "POOR" in content  # score label for < 0.4


class TestOnnxPipelineDataclasses:
    def test_frame_analysis_to_dict(self) -> None:
        from temporalci.vision.onnx_pipeline import FrameAnalysis

        import numpy as np

        from temporalci.vision.clearance import ClearanceResult, WireDetection
        from temporalci.vision.segmentation import SegmentationResult

        seg = SegmentationResult(
            seg_map=np.zeros((10, 10), dtype=np.int32),
            vegetation_mask=np.zeros((10, 10), dtype=bool),
            sky_mask=np.zeros((10, 10), dtype=bool),
            infrastructure_mask=np.zeros((10, 10), dtype=bool),
            ground_mask=np.zeros((10, 10), dtype=bool),
            vegetation_ratio=0.3,
            vegetation_upper_ratio=0.2,
            classes_found={0, 4},
        )
        wires = WireDetection(
            wire_mask=np.zeros((10, 10), dtype=bool),
            wire_lines=[],
            wire_count=0,
            search_zone_h=5,
        )
        clearance = ClearanceResult(
            min_clearance_px=50.0,
            min_clearance_relative=0.5,
            depth_adjusted_clearance=60.0,
            risk_level="caution",
            risk_score=0.55,
            closest_vegetation=[5, 5],
            closest_wire=[2, 5],
            vegetation_in_wire_zone=0.2,
            vegetation_penetration=0.3,
        )
        analysis = FrameAnalysis(
            frame_id="test_001",
            frame_path="/tmp/test.jpg",
            segmentation=seg,
            depth=None,
            wires=wires,
            clearance=clearance,
            risk_level="caution",
            risk_score=0.55,
            elapsed_ms=150.0,
        )
        d = analysis.to_dict()
        assert d["frame_id"] == "test_001"
        assert d["risk_score"] == 0.55
        assert d["vegetation_ratio"] == 0.3
        assert d["elapsed_ms"] == 150.0

    def test_pipeline_stats(self) -> None:
        from temporalci.vision.onnx_pipeline import PipelineStats

        stats = PipelineStats(
            total_frames=10,
            total_elapsed_ms=5000.0,
            seg_elapsed_ms=2000.0,
            depth_elapsed_ms=2500.0,
            clearance_elapsed_ms=500.0,
            risk_distribution={"critical": 3, "safe": 7},
        )
        assert stats.avg_ms_per_frame == 500.0
        assert stats.fps == 2.0
        d = stats.to_dict()
        assert d["total_frames"] == 10
        assert d["fps"] == 2.0
