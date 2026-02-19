from __future__ import annotations

from pathlib import Path

import pytest

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

pytestmark = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def _make_run_data(status: str = "PASS", alert_frames: list | None = None) -> dict:
    return {
        "run_id": "test-run-001",
        "project": "test-project",
        "suite_name": "test-suite",
        "model_name": "cam-left",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "status": status,
        "sample_count": 3,
        "metrics": {
            "catenary_vegetation": {
                "score": 0.85,
                "dims": {
                    "vegetation_proximity": 0.02,
                    "green_coverage": 0.08,
                    "catenary_visibility": 0.45,
                },
                "per_sample": [
                    {
                        "prompt": "f_00",
                        "dims": {
                            "vegetation_proximity": 0.01,
                            "green_coverage": 0.05,
                            "catenary_visibility": 0.5,
                        },
                    },
                    {
                        "prompt": "f_01",
                        "dims": {
                            "vegetation_proximity": 0.02,
                            "green_coverage": 0.1,
                            "catenary_visibility": 0.4,
                        },
                    },
                    {
                        "prompt": "f_02",
                        "dims": {
                            "vegetation_proximity": 0.03,
                            "green_coverage": 0.09,
                            "catenary_visibility": 0.45,
                        },
                    },
                ],
                "alert_frames": alert_frames or [],
            },
        },
        "gates": [
            {
                "metric": "catenary_vegetation.score",
                "op": ">=",
                "value": 0.7,
                "actual": 0.85,
                "passed": True,
            },
        ],
    }


class TestInspectionReport:
    def test_generates_html(self, tmp_path: Path) -> None:
        from temporalci.inspection_report import write_inspection_report

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(3):
            Image.new("RGB", (100, 80), (100, 130, 230)).save(str(frames / f"f_{i:02d}.jpg"))

        out = tmp_path / "report.html"
        result = write_inspection_report(out, run_data=_make_run_data(), frame_dir=frames)

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Catenary Vegetation Inspection Report" in content
        assert "PASS" in content
        assert "data:image/png;base64," in content

    def test_alert_frames_highlighted(self, tmp_path: Path) -> None:
        from temporalci.inspection_report import write_inspection_report

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(3):
            Image.new("RGB", (100, 80), (20, 150, 20)).save(str(frames / f"f_{i:02d}.jpg"))

        alerts = [{"prompt": "f_01", "frame": "f_01.jpg", "vegetation_proximity": 0.8}]
        out = tmp_path / "report.html"
        result = write_inspection_report(
            out, run_data=_make_run_data("FAIL", alerts), frame_dir=frames
        )

        content = result.read_text(encoding="utf-8")
        assert "ALERT" in content
        assert "FAIL" in content

    def test_self_contained(self, tmp_path: Path) -> None:
        """Report should have no external dependencies (images are base64)."""
        from temporalci.inspection_report import write_inspection_report

        frames = tmp_path / "frames"
        frames.mkdir()
        Image.new("RGB", (100, 80), (100, 130, 230)).save(str(frames / "f_00.jpg"))

        out = tmp_path / "report.html"
        write_inspection_report(out, run_data=_make_run_data(), frame_dir=frames)

        content = out.read_text(encoding="utf-8")
        # No external image references
        assert 'src="http' not in content
        assert 'src="/' not in content
        # Has base64 images
        assert "base64," in content

    def test_proximity_bars_present(self, tmp_path: Path) -> None:
        from temporalci.inspection_report import write_inspection_report

        frames = tmp_path / "frames"
        frames.mkdir()
        Image.new("RGB", (100, 80), (100, 130, 230)).save(str(frames / "f_00.jpg"))

        out = tmp_path / "report.html"
        write_inspection_report(out, run_data=_make_run_data(), frame_dir=frames)

        content = out.read_text(encoding="utf-8")
        assert "bar" in content
