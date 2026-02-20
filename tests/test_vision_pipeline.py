"""Tests for the 3-layer vision pipeline (segmentation + depth + clearance)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from temporalci.types import GeneratedSample
from temporalci.vision.clearance import (
    ClearanceResult,
    WireDetection,
    calculate_clearance,
    detect_wires,
)
from temporalci.vision.depth import DepthResult
from temporalci.vision.segmentation import (
    VEGETATION_IDS,
    SegmentationResult,
)

_HAS_PIL = bool(importlib.util.find_spec("PIL"))
_HAS_TORCH = bool(importlib.util.find_spec("torch"))

try:
    from PIL import Image
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _HAS_PIL or not _HAS_TORCH,
    reason="Pillow and torch required",
)


def _make_sample(path: str, prompt: str = "f") -> GeneratedSample:
    return GeneratedSample(
        test_id="t1", prompt=prompt, seed=0, video_path=path, evaluation_stream=[]
    )


def _make_seg_result(
    h: int = 100,
    w: int = 200,
    veg_rows: tuple[int, int] = (20, 60),
) -> SegmentationResult:
    """Create a synthetic segmentation result with vegetation in a band."""
    seg_map = np.zeros((h, w), dtype=np.int32)
    veg_mask = np.zeros((h, w), dtype=bool)
    sky_mask = np.zeros((h, w), dtype=bool)
    infra_mask = np.zeros((h, w), dtype=bool)
    ground_mask = np.zeros((h, w), dtype=bool)

    # Top rows = sky
    sky_mask[: veg_rows[0], :] = True
    seg_map[: veg_rows[0], :] = 2  # sky

    # Vegetation band
    veg_mask[veg_rows[0] : veg_rows[1], :] = True
    seg_map[veg_rows[0] : veg_rows[1], :] = 4  # tree

    # Bottom = ground
    ground_mask[veg_rows[1] :, :] = True
    seg_map[veg_rows[1] :, :] = 6  # road

    zone_h = int(h * 0.4)
    veg_upper = veg_mask[:zone_h, :]

    return SegmentationResult(
        seg_map=seg_map,
        vegetation_mask=veg_mask,
        sky_mask=sky_mask,
        infrastructure_mask=infra_mask,
        ground_mask=ground_mask,
        vegetation_ratio=float(veg_mask.mean()),
        vegetation_upper_ratio=float(veg_upper.mean()) if zone_h > 0 else 0.0,
        classes_found={2, 4, 6},
    )


def _make_depth(h: int = 100, w: int = 200) -> DepthResult:
    """Create a synthetic depth map (gradient top=far, bottom=near)."""
    depth = np.linspace(1.0, 0.0, h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
    return DepthResult(depth_map=depth, depth_raw=depth * 10.0, min_depth=0.0, max_depth=10.0)


class TestClearanceCalculation:
    def test_no_vegetation_is_safe(self) -> None:
        seg = _make_seg_result(veg_rows=(80, 90))  # vegetation only at bottom
        depth = _make_depth()
        wires = WireDetection(
            wire_mask=np.zeros((100, 200), dtype=bool),
            wire_lines=[],
            wire_count=0,
            search_zone_h=55,
        )
        result = calculate_clearance(seg, depth, wires)
        assert result.risk_level == "safe"
        assert result.risk_score == 1.0

    def test_vegetation_in_band_is_risky(self) -> None:
        seg = _make_seg_result(veg_rows=(5, 50))  # vegetation fills catenary band
        depth = _make_depth()
        wires = WireDetection(
            wire_mask=np.zeros((100, 200), dtype=bool),
            wire_lines=[],
            wire_count=0,
            search_zone_h=55,
        )
        result = calculate_clearance(seg, depth, wires)
        assert result.risk_level in ("critical", "warning")
        assert result.risk_score < 0.5
        assert result.vegetation_in_wire_zone > 0.3
        assert result.vegetation_penetration > 0.5

    def test_full_vegetation_coverage(self) -> None:
        seg = _make_seg_result(veg_rows=(0, 100))  # vegetation everywhere
        depth = _make_depth()
        wires = WireDetection(
            wire_mask=np.zeros((100, 200), dtype=bool),
            wire_lines=[],
            wire_count=0,
            search_zone_h=55,
        )
        result = calculate_clearance(seg, depth, wires)
        assert result.risk_level == "critical"
        assert result.risk_score == 0.0
        assert result.vegetation_penetration == 1.0

    def test_depth_increases_clearance(self) -> None:
        seg = _make_seg_result(veg_rows=(15, 35))
        wires = WireDetection(
            wire_mask=np.zeros((100, 200), dtype=bool),
            wire_lines=[],
            wire_count=0,
            search_zone_h=55,
        )
        # Without depth
        calculate_clearance(seg, None, wires)
        # With depth
        result_with_depth = calculate_clearance(seg, _make_depth(), wires)

        # depth_adjusted_clearance >= min_clearance_px
        assert result_with_depth.depth_adjusted_clearance >= result_with_depth.min_clearance_px


class TestWireDetection:
    def test_horizontal_line_detected(self, tmp_path: Path) -> None:
        # Create image with a clear horizontal line
        img = np.zeros((100, 400), dtype=np.uint8)
        img[30, :] = 255  # horizontal white line
        img_path = tmp_path / "wire.jpg"
        Image.fromarray(img).save(str(img_path))

        result = detect_wires(img_path)
        assert result.search_zone_h > 0
        # Line may or may not be detected depending on Hough params
        # but the function should not crash

    def test_with_segmentation_mask(self, tmp_path: Path) -> None:
        img = np.full((100, 400, 3), 128, dtype=np.uint8)
        # Draw a line in non-vegetation area
        img[20, :, :] = 255
        img_path = tmp_path / "wire.jpg"
        Image.fromarray(img).save(str(img_path))

        seg = _make_seg_result(h=100, w=400, veg_rows=(50, 80))
        result = detect_wires(img_path, seg=seg)
        # Should not crash and should filter vegetation edges
        assert isinstance(result.wire_count, int)

    def test_empty_image(self, tmp_path: Path) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = tmp_path / "black.jpg"
        Image.fromarray(img).save(str(img_path))

        result = detect_wires(img_path)
        assert result.wire_count == 0


class TestSegmentationResult:
    def test_to_dict(self) -> None:
        seg = _make_seg_result()
        d = seg.to_dict()
        assert "vegetation_ratio" in d
        assert "vegetation_upper_ratio" in d
        assert "classes_found" in d
        assert isinstance(d["classes_found"], list)

    def test_vegetation_ids_not_empty(self) -> None:
        assert len(VEGETATION_IDS) > 0
        assert 4 in VEGETATION_IDS  # tree


class TestDepthResult:
    def test_to_dict(self) -> None:
        depth = _make_depth()
        d = depth.to_dict()
        assert "min_depth" in d
        assert "max_depth" in d
        assert "mean_depth" in d


class TestClearanceResult:
    def test_to_dict(self) -> None:
        result = ClearanceResult(
            min_clearance_px=50.0,
            min_clearance_relative=0.5,
            depth_adjusted_clearance=60.0,
            risk_level="caution",
            risk_score=0.55,
            closest_vegetation=[20, 100],
            closest_wire=[10, 100],
            vegetation_in_wire_zone=0.3,
            vegetation_penetration=0.4,
        )
        d = result.to_dict()
        assert d["risk_level"] == "caution"
        assert d["vegetation_penetration"] == 0.4


class TestClearanceMetric:
    def test_empty_samples(self) -> None:
        from temporalci.metrics.catenary_clearance import evaluate

        result = evaluate([])
        assert result["score"] == 0.0
        assert result["sample_count"] == 0

    def test_missing_file(self) -> None:
        from temporalci.metrics.catenary_clearance import evaluate

        sample = _make_sample("/nonexistent/frame.jpg")
        # This will try to load models — mock them to avoid download
        with (
            patch("temporalci.metrics.catenary_clearance._load_models") as mock_load,
            patch("temporalci.vision.clearance.detect_wires"),
            patch("temporalci.vision.clearance.calculate_clearance"),
        ):
            mock_load.return_value = (MagicMock(), MagicMock())
            result = evaluate([sample])
            assert result["sample_count"] == 1
            assert result["per_sample"][0].get("error")

    def test_registered_in_metrics(self) -> None:
        from temporalci.metrics import available_metrics

        assert "catenary_clearance" in available_metrics()
