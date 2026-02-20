"""Tests for the video processing pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_HAS_CV2 = bool(importlib.util.find_spec("cv2"))
_HAS_PIL = bool(importlib.util.find_spec("PIL"))

pytestmark = pytest.mark.skipif(
    not _HAS_CV2 or not _HAS_PIL,
    reason="OpenCV and Pillow required",
)


class TestExtractFrames:
    def test_extracts_frames(self, tmp_path: Path) -> None:
        from temporalci.vision.video import extract_frames

        # Create a test video
        import cv2

        video_path = tmp_path / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30.0,
            (320, 240),
        )
        for i in range(90):  # 3 seconds at 30fps
            frame = np.full((240, 320, 3), i * 2 % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        out_dir = tmp_path / "frames"
        frames = extract_frames(video_path, out_dir, fps=1.0)
        assert len(frames) >= 2
        assert all(f.exists() for f in frames)
        assert all(f.suffix == ".jpg" for f in frames)

    def test_max_frames_limit(self, tmp_path: Path) -> None:
        from temporalci.vision.video import extract_frames

        import cv2

        video_path = tmp_path / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30.0,
            (320, 240),
        )
        for _ in range(90):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        out_dir = tmp_path / "frames"
        frames = extract_frames(video_path, out_dir, fps=10.0, max_frames=5)
        assert len(frames) == 5

    def test_invalid_video_raises(self, tmp_path: Path) -> None:
        from temporalci.vision.video import extract_frames

        bad_path = tmp_path / "nonexistent.mp4"
        with pytest.raises(RuntimeError, match="Cannot open video"):
            extract_frames(bad_path, tmp_path / "out")


class TestProcessVideo:
    def test_process_with_mock(self, tmp_path: Path) -> None:
        from temporalci.vision.video import process_video

        # Create a minimal test video
        import cv2

        video_path = tmp_path / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (100, 100),
        )
        for _ in range(10):
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.release()

        mock_result = {
            "score": 0.75,
            "dims": {"risk_score": 0.75},
            "sample_count": 1,
            "per_sample": [],
            "alert_frames": [],
        }

        with patch("temporalci.metrics.catenary_clearance.evaluate", return_value=mock_result):
            result = process_video(
                video_path,
                output_dir=str(tmp_path / "output"),
                fps=1.0,
                max_frames=2,
            )
            assert result["score"] == 0.75
            assert "_meta" in result
            assert result["_meta"]["frames_extracted"] >= 1
