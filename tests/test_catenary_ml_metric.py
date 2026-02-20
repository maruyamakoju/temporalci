"""Tests for the ML metric stub — verifies HSV fallback works correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.types import GeneratedSample

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

pytestmark = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def _make_sample(path: str, prompt: str = "f") -> GeneratedSample:
    return GeneratedSample(
        test_id="t1", prompt=prompt, seed=0, video_path=path, evaluation_stream=[]
    )


class TestCatenaryVegetationML:
    def test_fallback_to_hsv(self, tmp_path: Path) -> None:
        """Without model_path, ML metric should fall back to HSV and produce same results."""
        from temporalci.metrics.catenary_vegetation import evaluate as hsv_eval
        from temporalci.metrics.catenary_vegetation_ml import evaluate as ml_eval

        img_path = tmp_path / "blue.jpg"
        Image.new("RGB", (100, 100), (100, 130, 230)).save(str(img_path))

        sample = _make_sample(str(img_path))
        hsv_result = hsv_eval([sample])
        ml_result = ml_eval([sample])

        assert ml_result["score"] == hsv_result["score"]
        assert ml_result["dims"] == hsv_result["dims"]

    def test_nonexistent_model_path_falls_back(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_vegetation_ml import evaluate as ml_eval

        img_path = tmp_path / "green.jpg"
        Image.new("RGB", (100, 100), (20, 150, 20)).save(str(img_path))

        result = ml_eval(
            [_make_sample(str(img_path))],
            params={"model_path": "/nonexistent/model.pt"},
        )
        # Should still work via HSV fallback — a fully green image
        # gives proximity ≈ 1.0, so composite score ≈ 0.0 (high danger).
        assert result["score"] >= 0
        assert result["dims"]["vegetation_proximity"] > 0.5

    def test_registered_in_metrics(self) -> None:
        from temporalci.metrics import available_metrics

        assert "catenary_vegetation_ml" in available_metrics()

    def test_empty_samples(self) -> None:
        from temporalci.metrics.catenary_vegetation_ml import evaluate

        result = evaluate([])
        assert result["score"] == 0.0
        assert result["sample_count"] == 0
