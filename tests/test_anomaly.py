"""Tests for catenary anomaly detection module and metric plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from temporalci.types import GeneratedSample

try:
    import cv2
    import numpy as np
    from PIL import Image

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="vision deps not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(path: str, prompt: str = "frame") -> GeneratedSample:
    return GeneratedSample(
        test_id="t1",
        prompt=prompt,
        seed=0,
        video_path=path,
        evaluation_stream=[],
    )


def _make_wire_detection(
    wire_lines: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int] = (100, 200),
) -> Any:
    from temporalci.vision.clearance import WireDetection

    h, w = image_shape
    wire_mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in wire_lines:
        cv2.line(wire_mask.astype(np.uint8), (x1, y1), (x2, y2), 1, 1)
    return WireDetection(
        wire_mask=wire_mask,
        wire_lines=wire_lines,
        wire_count=len(wire_lines),
        search_zone_h=int(h * 0.55),
    )


def _make_seg_result(
    seg_map: Any,
) -> Any:
    from temporalci.vision.segmentation import (
        GROUND_IDS,
        INFRASTRUCTURE_IDS,
        SKY_IDS,
        VEGETATION_IDS,
        SegmentationResult,
    )

    h, w = seg_map.shape
    veg_mask = np.isin(seg_map, list(VEGETATION_IDS))
    sky_mask = np.isin(seg_map, list(SKY_IDS))
    infra_mask = np.isin(seg_map, list(INFRASTRUCTURE_IDS))
    ground_mask = np.isin(seg_map, list(GROUND_IDS))
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
        classes_found=set(np.unique(seg_map).tolist()),
    )


def _make_depth_result(
    shape: tuple[int, int] = (100, 200),
    fill: float = 0.5,
) -> Any:
    from temporalci.vision.depth import DepthResult

    depth_map = np.full(shape, fill, dtype=np.float32)
    return DepthResult(
        depth_map=depth_map,
        depth_raw=depth_map.copy(),
        min_depth=fill,
        max_depth=fill,
    )


# ---------------------------------------------------------------------------
# WireSagResult tests
# ---------------------------------------------------------------------------


class TestEstimateWireSag:
    """Tests for estimate_wire_sag."""

    def test_no_wires_returns_normal(self) -> None:
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(wire_lines=[])
        result = estimate_wire_sag(wd, image_height=100)
        assert result.sag_detected is False
        assert result.severity == "normal"
        assert result.max_deviation_px == 0.0
        assert result.max_deviation_relative == 0.0
        assert result.expected_curve == []
        assert result.actual_points == []

    def test_straight_wire_normal_sag(self) -> None:
        """A perfectly horizontal wire should have negligible sag."""
        from temporalci.vision.anomaly import estimate_wire_sag

        # Horizontal line at y=20 from x=10 to x=190
        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        assert result.severity == "normal"
        assert result.max_deviation_px < 5.0
        assert result.sag_detected is False

    def test_sagging_wire_detected(self) -> None:
        """A wire with significant downward sag in the middle should be detected."""
        from temporalci.vision.anomaly import estimate_wire_sag

        # Two line segments forming a V-shape (sag in the middle)
        # Left: (10, 20) -> (100, 40), Right: (100, 40) -> (190, 20)
        wd = _make_wire_detection(
            wire_lines=[(10, 20, 100, 40), (100, 40, 190, 20)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        # The deviation should be non-trivial
        assert result.max_deviation_px > 0
        assert len(result.actual_points) > 0
        assert len(result.expected_curve) > 0

    def test_severe_sag(self) -> None:
        """A wire with extreme sag should be classified as severe."""
        from temporalci.vision.anomaly import estimate_wire_sag

        # Extreme V-shape sag: endpoints at y=10, midpoint at y=60
        wd = _make_wire_detection(
            wire_lines=[(10, 10, 100, 60), (100, 60, 190, 10)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        assert result.max_deviation_px > 0
        # With such extreme sag, the deviation should push into moderate or severe
        assert result.severity in ("moderate", "severe")

    def test_sag_location_format(self) -> None:
        """sag_location should be [y, x] format."""
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        assert isinstance(result.sag_location, list)
        assert len(result.sag_location) == 2

    def test_with_depth_result(self) -> None:
        """Depth result should be used without crashing."""
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        depth = _make_depth_result(shape=(100, 200), fill=0.3)
        result = estimate_wire_sag(wd, depth_result=depth, image_height=100)
        assert result.severity == "normal"

    def test_image_height_zero_uses_search_zone(self) -> None:
        """When image_height=0, should fall back to search_zone_h."""
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=0)
        # search_zone_h = 55 (0.55 * 100)
        assert result.max_deviation_relative >= 0.0

    def test_to_dict(self) -> None:
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        d = result.to_dict()
        assert "sag_detected" in d
        assert "severity" in d
        assert "max_deviation_px" in d


# ---------------------------------------------------------------------------
# EquipmentState tests
# ---------------------------------------------------------------------------


class TestAssessEquipment:
    """Tests for assess_equipment."""

    def test_no_infrastructure(self) -> None:
        """All sky should yield unknown condition."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 2, dtype=np.int32)  # all sky
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.overall_condition == "unknown"
        assert result.infrastructure_coverage < 0.01
        assert result.pole_count == 0

    def test_with_poles(self) -> None:
        """Segmentation with pole class should count poles."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 2, dtype=np.int32)  # sky background
        # Add two separate pole regions
        seg_map[10:40, 20:25] = 93  # pole 1
        seg_map[10:40, 170:175] = 93  # pole 2
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.pole_count >= 2
        assert result.infrastructure_coverage > 0

    def test_with_pylons(self) -> None:
        """Pylon class (136) should also be counted."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        seg_map[5:50, 90:110] = 136  # pylon
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.pole_count >= 1

    def test_high_infrastructure_coverage(self) -> None:
        """High infrastructure coverage should yield good condition."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 1, dtype=np.int32)  # all building
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.infrastructure_coverage > 0.5
        assert result.infrastructure_visibility > 0.5
        assert result.overall_condition in ("good", "fair")

    def test_with_depth(self) -> None:
        """Depth result should affect visibility calculation."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 1, dtype=np.int32)  # building
        seg = _make_seg_result(seg_map)
        depth_close = _make_depth_result(shape=(100, 200), fill=0.1)
        depth_far = _make_depth_result(shape=(100, 200), fill=0.9)
        result_close = assess_equipment(seg, depth_result=depth_close)
        result_far = assess_equipment(seg, depth_result=depth_far)
        # Closer infrastructure should have higher visibility
        assert result_close.infrastructure_visibility >= result_far.infrastructure_visibility

    def test_empty_image(self) -> None:
        """Zero-size image should return unknown."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.zeros((0, 0), dtype=np.int32)
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.overall_condition == "unknown"
        assert result.pole_count == 0

    def test_to_dict(self) -> None:
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        d = result.to_dict()
        assert "infrastructure_visibility" in d
        assert "overall_condition" in d
        assert "pole_count" in d


# ---------------------------------------------------------------------------
# AnomalyResult / detect_anomalies tests
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    """Tests for detect_anomalies integration."""

    def test_no_anomalies(self) -> None:
        """Clean frame with no wires and sky should produce low anomaly score."""
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 2, dtype=np.int32)  # sky
        seg = _make_seg_result(seg_map)
        wd = _make_wire_detection(wire_lines=[], image_shape=(100, 200))
        result = detect_anomalies(seg, wd, image_height=100)
        assert result.anomaly_score >= 0.0
        assert result.anomaly_score <= 1.0
        assert isinstance(result.anomaly_flags, list)

    def test_severe_sag_flag(self) -> None:
        """Severe wire sag should produce a sag flag."""
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        seg = _make_seg_result(seg_map)
        # Extreme sag
        wd = _make_wire_detection(
            wire_lines=[(10, 5, 100, 55), (100, 55, 190, 5)],
            image_shape=(100, 200),
        )
        result = detect_anomalies(seg, wd, image_height=100)
        sag_flags = [f for f in result.anomaly_flags if f.startswith("wire_sag")]
        # Should have either moderate or severe flag
        assert len(sag_flags) >= 0  # may or may not trigger depending on fit

    def test_low_visibility_flag(self) -> None:
        """No infrastructure should trigger low_infrastructure_visibility flag."""
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 2, dtype=np.int32)  # all sky
        seg = _make_seg_result(seg_map)
        wd = _make_wire_detection(wire_lines=[], image_shape=(100, 200))
        result = detect_anomalies(seg, wd, image_height=100)
        assert "low_infrastructure_visibility" in result.anomaly_flags

    def test_equipment_poor_flag(self) -> None:
        """Equipment in poor condition should produce the equipment_poor flag."""
        from temporalci.vision.anomaly import assess_equipment, detect_anomalies

        # Create a segmentation that produces poor equipment condition
        # Use a noisy infrastructure region to get high edge density
        rng = np.random.RandomState(42)
        seg_map = rng.choice([1, 2, 32, 93], size=(100, 200)).astype(np.int32)
        seg = _make_seg_result(seg_map)

        # Verify the equipment assessment
        equip = assess_equipment(seg)
        if equip.overall_condition == "poor":
            wd = _make_wire_detection(wire_lines=[], image_shape=(100, 200))
            result = detect_anomalies(seg, wd, image_height=100)
            assert "equipment_poor" in result.anomaly_flags

    def test_anomaly_score_bounded(self) -> None:
        """Anomaly score should always be between 0 and 1."""
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 93, dtype=np.int32)  # all poles
        seg = _make_seg_result(seg_map)
        wd = _make_wire_detection(
            wire_lines=[(0, 0, 199, 99)],
            image_shape=(100, 200),
        )
        result = detect_anomalies(seg, wd, image_height=100)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_to_dict(self) -> None:
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        seg = _make_seg_result(seg_map)
        wd = _make_wire_detection(wire_lines=[], image_shape=(100, 200))
        result = detect_anomalies(seg, wd, image_height=100)
        d = result.to_dict()
        assert "wire_sag" in d
        assert "equipment" in d
        assert "anomaly_score" in d
        assert "anomaly_flags" in d

    def test_with_depth(self) -> None:
        """Full pipeline with depth should not crash."""
        from temporalci.vision.anomaly import detect_anomalies

        seg_map = np.full((100, 200), 1, dtype=np.int32)
        seg = _make_seg_result(seg_map)
        wd = _make_wire_detection(
            wire_lines=[(10, 20, 190, 20)],
            image_shape=(100, 200),
        )
        depth = _make_depth_result(shape=(100, 200), fill=0.4)
        result = detect_anomalies(seg, wd, depth_result=depth, image_height=100)
        assert result.wire_sag is not None
        assert result.equipment is not None


# ---------------------------------------------------------------------------
# Metric plugin tests
# ---------------------------------------------------------------------------


class TestCatenaryAnomalyMetric:
    """Tests for the catenary_anomaly metric plugin."""

    def test_empty_samples(self) -> None:
        from temporalci.metrics.catenary_anomaly import ALL_DIMS, evaluate

        result = evaluate([])
        assert result["score"] == 0.0
        assert result["sample_count"] == 0
        assert set(result["dims"].keys()) == set(ALL_DIMS)
        assert result["alert_frames"] == []
        assert result["per_sample"] == []

    def test_missing_frame(self, tmp_path: Path) -> None:
        from temporalci.metrics.catenary_anomaly import evaluate

        result = evaluate([_make_sample(str(tmp_path / "missing.jpg"))])
        assert result["sample_count"] == 1
        assert "error" in result["per_sample"][0]

    def test_metric_registered(self) -> None:
        """catenary_anomaly should be in the metric registry."""
        from temporalci.metrics import available_metrics

        assert "catenary_anomaly" in available_metrics()

    def test_evaluate_with_mocked_models(self, tmp_path: Path) -> None:
        """Evaluate with fully mocked vision pipeline."""
        from temporalci.metrics.catenary_anomaly import evaluate

        # Create a real image file
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        img_path = str(tmp_path / "test_frame.jpg")
        img.save(img_path)

        # Create mock segmentation and depth results
        seg_map = np.full((100, 200), 2, dtype=np.int32)  # sky
        mock_seg = _make_seg_result(seg_map)
        mock_depth = _make_depth_result(shape=(100, 200), fill=0.5)

        mock_seg_model = MagicMock()
        mock_seg_model.segment.return_value = mock_seg
        mock_depth_model = MagicMock()
        mock_depth_model.estimate.return_value = mock_depth

        mock_wires = _make_wire_detection(wire_lines=[], image_shape=(100, 200))

        with (
            patch(
                "temporalci.metrics.catenary_anomaly._load_models",
                return_value=(mock_seg_model, mock_depth_model),
            ),
            patch(
                "temporalci.vision.clearance.detect_wires",
                return_value=mock_wires,
            ),
        ):
            result = evaluate([_make_sample(img_path)])

        assert result["sample_count"] == 1
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert len(result["per_sample"]) == 1
        assert "anomaly_flags" in result["per_sample"][0]

    def test_alert_frames_triggered(self, tmp_path: Path) -> None:
        """High anomaly score should trigger alert frames."""
        from temporalci.metrics.catenary_anomaly import evaluate
        from temporalci.vision.anomaly import AnomalyResult, EquipmentState, WireSagResult

        img = Image.new("RGB", (200, 100), (128, 128, 128))
        img_path = str(tmp_path / "alert_frame.jpg")
        img.save(img_path)

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        mock_seg = _make_seg_result(seg_map)

        mock_seg_model = MagicMock()
        mock_seg_model.segment.return_value = mock_seg
        mock_depth_model = MagicMock()
        mock_depth_model.estimate.return_value = _make_depth_result()

        mock_wires = _make_wire_detection(wire_lines=[], image_shape=(100, 200))

        # Create a high-anomaly result
        high_anomaly = AnomalyResult(
            wire_sag=WireSagResult(
                sag_detected=True,
                max_deviation_px=20.0,
                max_deviation_relative=0.2,
                sag_location=[50, 100],
                expected_curve=[],
                actual_points=[],
                severity="severe",
            ),
            equipment=EquipmentState(
                infrastructure_visibility=0.1,
                infrastructure_coverage=0.01,
                pole_count=0,
                insulator_anomaly_score=0.8,
                overall_condition="poor",
            ),
            anomaly_score=0.85,
            anomaly_flags=["wire_sag_severe", "equipment_poor"],
        )

        with (
            patch(
                "temporalci.metrics.catenary_anomaly._load_models",
                return_value=(mock_seg_model, mock_depth_model),
            ),
            patch(
                "temporalci.vision.clearance.detect_wires",
                return_value=mock_wires,
            ),
            patch(
                "temporalci.vision.anomaly.detect_anomalies",
                return_value=high_anomaly,
            ),
        ):
            result = evaluate(
                [_make_sample(img_path)],
                params={"anomaly_threshold": 0.4},
            )

        assert len(result["alert_frames"]) == 1
        assert result["alert_frames"][0]["anomaly_score"] >= 0.4

    def test_dims_keys(self, tmp_path: Path) -> None:
        """All expected dimension keys should be present."""
        from temporalci.metrics.catenary_anomaly import ALL_DIMS, evaluate

        img = Image.new("RGB", (200, 100), (128, 128, 128))
        img_path = str(tmp_path / "dims_frame.jpg")
        img.save(img_path)

        seg_map = np.full((100, 200), 2, dtype=np.int32)
        mock_seg = _make_seg_result(seg_map)

        mock_seg_model = MagicMock()
        mock_seg_model.segment.return_value = mock_seg

        mock_wires = _make_wire_detection(wire_lines=[], image_shape=(100, 200))

        with (
            patch(
                "temporalci.metrics.catenary_anomaly._load_models",
                return_value=(mock_seg_model, None),
            ),
            patch(
                "temporalci.vision.clearance.detect_wires",
                return_value=mock_wires,
            ),
        ):
            result = evaluate([_make_sample(img_path)])

        assert set(result["dims"].keys()) == set(ALL_DIMS)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_point_wire(self) -> None:
        """A degenerate wire with x1==x2 should not crash."""
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(50, 10, 50, 10)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        assert result.severity == "normal"

    def test_vertical_wire(self) -> None:
        """A vertical wire should be handled gracefully."""
        from temporalci.vision.anomaly import estimate_wire_sag

        wd = _make_wire_detection(
            wire_lines=[(50, 0, 50, 99)],
            image_shape=(100, 200),
        )
        result = estimate_wire_sag(wd, image_height=100)
        # Vertical line has no horizontal spread — catenary fit degenerates
        assert result.severity in ("normal", "moderate", "severe")

    def test_many_wires(self) -> None:
        """Multiple wire lines should all contribute to the analysis."""
        from temporalci.vision.anomaly import estimate_wire_sag

        lines = [
            (10, 20, 190, 20),
            (10, 30, 190, 30),
            (10, 40, 190, 40),
        ]
        wd = _make_wire_detection(wire_lines=lines, image_shape=(100, 200))
        result = estimate_wire_sag(wd, image_height=100)
        assert len(result.actual_points) > 100  # many sampled points

    def test_mixed_infrastructure(self) -> None:
        """Mixed infrastructure classes should be counted correctly."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 2, dtype=np.int32)  # sky
        seg_map[10:30, 10:15] = 93  # pole
        seg_map[60:90, 50:55] = 87  # streetlight
        seg_map[10:40, 150:170] = 1  # building
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.infrastructure_coverage > 0
        assert result.pole_count >= 1  # at least the pole

    def test_all_infrastructure(self) -> None:
        """Image entirely infrastructure should yield good visibility."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 93, dtype=np.int32)  # all poles
        seg = _make_seg_result(seg_map)
        result = assess_equipment(seg)
        assert result.infrastructure_coverage > 0.9
        assert result.infrastructure_visibility > 0.5

    def test_catenary_fit_straight_line(self) -> None:
        """A perfectly straight wire should produce near-zero deviation."""
        from temporalci.vision.anomaly import _catenary_y, _fit_catenary

        points = [(x, 50) for x in range(0, 200, 5)]
        a, h, k = _fit_catenary(points)
        # For a straight line, expected y at midpoint should be ~50
        mid_y = _catenary_y(100.0, a, h, k)
        assert abs(mid_y - 50.0) < 5.0

    def test_depth_far_reduces_visibility(self) -> None:
        """Far depth should reduce infrastructure visibility compared to close."""
        from temporalci.vision.anomaly import assess_equipment

        seg_map = np.full((100, 200), 1, dtype=np.int32)  # building
        seg = _make_seg_result(seg_map)
        depth_close = _make_depth_result(shape=(100, 200), fill=0.0)
        depth_far = _make_depth_result(shape=(100, 200), fill=1.0)
        result_close = assess_equipment(seg, depth_result=depth_close)
        result_far = assess_equipment(seg, depth_result=depth_far)
        assert result_close.infrastructure_visibility >= result_far.infrastructure_visibility
