from __future__ import annotations

from pathlib import Path

import pytest

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

pytestmark = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def _make_frame(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (120, 80)) -> Path:
    Image.new("RGB", size, color).save(str(path))
    return path


class TestGenerateHeatmap:
    def test_basic_output(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmap

        frame = _make_frame(tmp_path / "frame.jpg", (100, 130, 230))
        out = tmp_path / "out" / "heatmap.png"
        result = generate_heatmap(frame, out)

        assert Path(result["output_path"]).exists()
        assert result["green_ratio_quarter"] < 0.01
        assert result["green_ratio_half"] < 0.01

    def test_green_frame_detected(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmap

        frame = _make_frame(tmp_path / "green.jpg", (20, 150, 20))
        out = tmp_path / "heatmap.png"
        result = generate_heatmap(frame, out)

        assert result["green_ratio_quarter"] > 0.9
        assert result["green_ratio_half"] > 0.9
        # Output should be a valid image
        img = Image.open(str(out))
        assert img.size == (120, 80)

    def test_overlay_is_visible(self, tmp_path: Path) -> None:
        """Green frame heatmap should have red-tinted overlay pixels."""
        from temporalci.heatmap import generate_heatmap

        frame = _make_frame(tmp_path / "green.jpg", (20, 150, 20))
        out = tmp_path / "heatmap.png"
        generate_heatmap(frame, out, overlay_alpha=0.5)

        import numpy as np

        img = np.asarray(Image.open(str(out)).convert("RGB"))
        # Center pixel should have significant red component from overlay
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        assert img[cy, cx, 0] > 80  # Red channel boosted by overlay

    def test_zone_lines_drawn(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmap

        frame = _make_frame(tmp_path / "frame.jpg", (100, 130, 230), size=(200, 100))
        out = tmp_path / "heatmap.png"
        generate_heatmap(frame, out, zone_line=True)

        import numpy as np

        img = np.asarray(Image.open(str(out)).convert("RGB"))
        # At y=25 (quarter line), there should be cyan pixels
        row = img[25, :, :]
        # At least some cyan pixels (0, 220, 255)
        has_cyan = (row[:, 2] > 200).any()
        assert has_cyan

    def test_no_zone_lines(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmap

        frame = _make_frame(tmp_path / "frame.jpg", (128, 128, 128), size=(200, 100))
        out_with = tmp_path / "with_lines.png"
        out_without = tmp_path / "without_lines.png"
        generate_heatmap(frame, out_with, zone_line=True)
        generate_heatmap(frame, out_without, zone_line=False)

        import numpy as np

        with_lines = np.asarray(Image.open(str(out_with)))
        without_lines = np.asarray(Image.open(str(out_without)))
        # Images should differ (zone lines present vs absent)
        assert not np.array_equal(with_lines, without_lines)


class TestGenerateHeatmaps:
    def test_batch_processing(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmaps

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(5):
            _make_frame(frames / f"f_{i:02d}.jpg", (100, 130, 230))

        out = tmp_path / "heatmaps"
        results = generate_heatmaps(frames, out, pattern="*.jpg")

        assert len(results) == 5
        assert all(Path(r["output_path"]).exists() for r in results)
        assert all("source_frame" in r for r in results)

    def test_pattern_filter(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmaps

        frames = tmp_path / "frames"
        frames.mkdir()
        _make_frame(frames / "a.jpg", (100, 130, 230))
        _make_frame(frames / "b.png", (100, 130, 230))

        results = generate_heatmaps(frames, tmp_path / "out", pattern="*.jpg")
        assert len(results) == 1
        assert results[0]["source_frame"] == "a.jpg"

    def test_sorted_output(self, tmp_path: Path) -> None:
        from temporalci.heatmap import generate_heatmaps

        frames = tmp_path / "frames"
        frames.mkdir()
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            _make_frame(frames / name, (100, 130, 230))

        results = generate_heatmaps(frames, tmp_path / "out")
        names = [r["source_frame"] for r in results]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]
