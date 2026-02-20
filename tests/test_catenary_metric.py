from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.types import GeneratedSample

try:
    from PIL import Image

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="Pillow/numpy not installed")


def _make_sample(path: str, prompt: str = "f") -> GeneratedSample:
    return GeneratedSample(
        test_id="t1",
        prompt=prompt,
        seed=0,
        video_path=path,
        evaluation_stream=[],
    )


def _save_solid_image(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (100, 100)
) -> str:
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return str(path)


def _save_split_image(
    path: Path,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
    size: tuple[int, int] = (100, 100),
) -> str:
    """Create an image with top half one color and bottom half another."""
    img = Image.new("RGB", size, bottom_color)
    pixels = img.load()
    assert pixels is not None
    w, h = size
    for y in range(h // 2):
        for x in range(w):
            pixels[x, y] = top_color
    img.save(str(path))
    return str(path)


class TestCatenaryVegetationMetric:
    def test_blue_sky_low_proximity(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        # Blue sky = no green → low vegetation_proximity
        p = _save_solid_image(tmp_path / "blue.jpg", (100, 130, 230))
        result = evaluate([_make_sample(p)])
        assert result["dims"]["vegetation_proximity"] < 0.1
        assert result["score"] > 0.7

    def test_green_high_proximity(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        # All green → high vegetation_proximity
        p = _save_solid_image(tmp_path / "green.jpg", (20, 150, 20))
        result = evaluate([_make_sample(p)])
        assert result["dims"]["vegetation_proximity"] > 0.8
        assert result["score"] < 0.4

    def test_empty_samples(self) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        result = evaluate([])
        assert result["score"] == 0.0
        assert result["sample_count"] == 0
        assert result["alert_frames"] == []

    def test_missing_frame_file(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        result = evaluate([_make_sample(str(tmp_path / "missing.jpg"))])
        assert result["sample_count"] == 1
        assert result["per_sample"][0]["error"]

    def test_alert_frames_triggered(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        p = _save_solid_image(tmp_path / "green.jpg", (20, 150, 20))
        result = evaluate([_make_sample(p)], params={"proximity_threshold": 0.1})
        assert len(result["alert_frames"]) == 1
        assert result["alert_frames"][0]["vegetation_proximity"] > 0.1

    def test_alert_frames_not_triggered(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        p = _save_solid_image(tmp_path / "blue.jpg", (100, 130, 230))
        result = evaluate([_make_sample(p)], params={"proximity_threshold": 0.3})
        assert result["alert_frames"] == []

    def test_score_direction(self, tmp_path: Path) -> None:
        """Higher score should mean safer (less vegetation)."""
        from temporalci.metrics.catenary_vegetation import evaluate

        blue = _save_solid_image(tmp_path / "blue.jpg", (100, 130, 230))
        green = _save_solid_image(tmp_path / "green.jpg", (20, 150, 20))

        safe = evaluate([_make_sample(blue)])
        danger = evaluate([_make_sample(green)])
        assert safe["score"] > danger["score"]

    def test_multiple_samples(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        p1 = _save_solid_image(tmp_path / "a.jpg", (100, 130, 230))
        p2 = _save_solid_image(tmp_path / "b.jpg", (20, 150, 20))
        result = evaluate([_make_sample(p1, "a"), _make_sample(p2, "b")])
        assert result["sample_count"] == 2
        assert len(result["per_sample"]) == 2

    def test_green_top_only(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        # Green top (catenary zone), blue bottom
        p = _save_split_image(
            tmp_path / "split.jpg",
            top_color=(20, 150, 20),
            bottom_color=(100, 130, 230),
        )
        result = evaluate([_make_sample(p)])
        # Upper half is all green → high proximity
        assert result["dims"]["vegetation_proximity"] > 0.5
        assert result["dims"]["green_coverage"] > 0.5

    def test_dims_keys(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation import evaluate

        p = _save_solid_image(tmp_path / "x.jpg", (128, 128, 128))
        result = evaluate([_make_sample(p)])
        assert set(result["dims"].keys()) == {
            "vegetation_proximity",
            "green_coverage",
            "catenary_visibility",
        }
