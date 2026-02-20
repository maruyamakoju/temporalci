"""Tests for the E2E demo pipeline script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSimulateGps:
    def test_returns_correct_count(self) -> None:
        from scripts.demo_full_pipeline import _simulate_gps

        data = _simulate_gps(10)
        assert len(data) == 10

    def test_has_required_keys(self) -> None:
        from scripts.demo_full_pipeline import _simulate_gps

        data = _simulate_gps(5)
        for d in data:
            assert "lat" in d
            assert "lon" in d
            assert "km" in d

    def test_km_increases(self) -> None:
        from scripts.demo_full_pipeline import _simulate_gps

        data = _simulate_gps(10, start_km=5.0)
        kms = [d["km"] for d in data]
        assert kms == sorted(kms)
        assert kms[0] == 5.0

    def test_single_frame(self) -> None:
        from scripts.demo_full_pipeline import _simulate_gps

        data = _simulate_gps(1)
        assert len(data) == 1
        assert "lat" in data[0]


class TestParseArgs:
    def test_basic(self) -> None:
        from scripts.demo_full_pipeline import _parse_args

        args = _parse_args(["--input", "video.mp4"])
        assert args.input == "video.mp4"
        assert args.output_dir == "demo_output"
        assert args.fps == 1.0

    def test_all_flags(self) -> None:
        from scripts.demo_full_pipeline import _parse_args

        args = _parse_args(
            [
                "--input",
                "v.mp4",
                "--output-dir",
                "/tmp/out",
                "--fps",
                "2",
                "--max-frames",
                "10",
                "--skip-depth",
                "--skip-anomaly",
                "--device",
                "cpu",
                "--title",
                "Test",
            ]
        )
        assert args.fps == 2.0
        assert args.max_frames == 10
        assert args.skip_depth is True
        assert args.skip_anomaly is True
        assert args.device == "cpu"
        assert args.title == "Test"


class TestRunDemo:
    def test_missing_video(self, tmp_path: Path) -> None:
        from scripts.demo_full_pipeline import run_demo

        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.mp4"),
            output_dir=str(tmp_path / "out"),
            fps=1.0,
            max_frames=0,
            skip_depth=False,
            skip_anomaly=True,
            device="cpu",
            title="Test",
        )
        result = run_demo(args)
        assert result == 1

    @patch("temporalci.vision.video.extract_frames")
    def test_no_frames_extracted(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        from scripts.demo_full_pipeline import run_demo

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        mock_extract.return_value = []

        args = argparse.Namespace(
            input=str(video),
            output_dir=str(tmp_path / "out"),
            fps=1.0,
            max_frames=0,
            skip_depth=True,
            skip_anomaly=True,
            device="cpu",
            title="Test",
        )
        result = run_demo(args)
        assert result == 1

    @patch("temporalci.metrics.catenary_clearance.evaluate")
    @patch("temporalci.vision.video.extract_frames")
    def test_full_pipeline_mock(
        self,
        mock_extract: MagicMock,
        mock_eval: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.demo_full_pipeline import run_demo

        # Setup
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        frames_dir = tmp_path / "out" / "frames"
        frames_dir.mkdir(parents=True)
        frame_files = []
        for i in range(3):
            f = frames_dir / f"frame_{i:05d}.jpg"
            f.write_bytes(b"fake")
            frame_files.append(f)
        mock_extract.return_value = frame_files

        mock_eval.return_value = {
            "score": 0.75,
            "dims": {"risk_score": 0.75, "vegetation_proximity_nn": 0.2},
            "sample_count": 3,
            "per_sample": [
                {
                    "prompt": f"frame_{i:05d}",
                    "risk_level": "safe",
                    "dims": {"risk_score": 0.75},
                    "clearance_px": 50.0,
                    "wire_count": 1,
                }
                for i in range(3)
            ],
            "alert_frames": [],
        }

        args = argparse.Namespace(
            input=str(video),
            output_dir=str(tmp_path / "out"),
            fps=1.0,
            max_frames=0,
            skip_depth=True,
            skip_anomaly=True,
            device="cpu",
            title="Test Demo",
        )
        result = run_demo(args)
        assert result == 0

        # Check outputs
        out_dir = tmp_path / "out"
        assert (out_dir / "dashboard.html").exists()
        assert (out_dir / "route_map.html").exists()
        assert (out_dir / "run.json").exists()
        assert (out_dir / "summary.txt").exists()

        # Check run.json content
        run_data = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        assert run_data["n_frames"] == 3
        assert run_data["clearance"]["score"] == 0.75

    @patch("temporalci.metrics.catenary_anomaly.evaluate")
    @patch("temporalci.metrics.catenary_clearance.evaluate")
    @patch("temporalci.vision.video.extract_frames")
    def test_with_anomaly(
        self,
        mock_extract: MagicMock,
        mock_clearance: MagicMock,
        mock_anomaly: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.demo_full_pipeline import run_demo

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        frames_dir = tmp_path / "out" / "frames"
        frames_dir.mkdir(parents=True)
        f = frames_dir / "frame_00000.jpg"
        f.write_bytes(b"fake")
        mock_extract.return_value = [f]

        mock_clearance.return_value = {
            "score": 0.5,
            "dims": {"risk_score": 0.5},
            "sample_count": 1,
            "per_sample": [
                {
                    "prompt": "frame_00000",
                    "risk_level": "warning",
                    "dims": {"risk_score": 0.5},
                    "clearance_px": 20.0,
                }
            ],
            "alert_frames": [],
        }
        mock_anomaly.return_value = {
            "score": 0.6,
            "dims": {"anomaly_score": 0.4},
            "sample_count": 1,
            "per_sample": [],
            "alert_frames": [{"prompt": "frame_00000", "anomaly_score": 0.6}],
        }

        args = argparse.Namespace(
            input=str(video),
            output_dir=str(tmp_path / "out"),
            fps=1.0,
            max_frames=0,
            skip_depth=True,
            skip_anomaly=False,
            device="cpu",
            title="Test",
        )
        result = run_demo(args)
        assert result == 0

        out_dir = tmp_path / "out"
        assert (out_dir / "anomaly_report.json").exists()
