"""Catenary equipment anomaly detection.

Detects wire sag anomalies by fitting ideal catenary curves to detected
wire lines and measuring deviation.  Assesses equipment condition from
segmentation infrastructure classes using connected-component analysis.

Combines both signals into a composite anomaly score with descriptive
flags for downstream alerting.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import cv2
import numpy as np

from temporalci.vision.clearance import WireDetection
from temporalci.vision.depth import DepthResult
from temporalci.vision.segmentation import SegmentationResult


# -- Severity thresholds (relative deviation normalised by image height) -----
_SAG_MODERATE_THRESHOLD = 0.02
_SAG_SEVERE_THRESHOLD = 0.05


@dataclasses.dataclass
class WireSagResult:
    """Wire sag analysis output."""

    sag_detected: bool
    max_deviation_px: float  # max deviation from expected catenary curve
    max_deviation_relative: float  # normalised by image height
    sag_location: list[int]  # [y, x] of maximum sag point
    expected_curve: list[tuple[int, int]]  # fitted catenary points
    actual_points: list[tuple[int, int]]  # detected wire points
    severity: str  # "normal", "moderate", "severe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sag_detected": self.sag_detected,
            "max_deviation_px": round(self.max_deviation_px, 2),
            "max_deviation_relative": round(self.max_deviation_relative, 6),
            "sag_location": self.sag_location,
            "severity": self.severity,
            "expected_curve_len": len(self.expected_curve),
            "actual_points_len": len(self.actual_points),
        }


@dataclasses.dataclass
class EquipmentState:
    """Infrastructure / equipment condition assessment."""

    infrastructure_visibility: float  # 0-1, how much infrastructure is visible
    infrastructure_coverage: float  # ratio of image with infrastructure classes
    pole_count: int  # detected poles/supports
    insulator_anomaly_score: float  # 0-1, higher = more anomalous
    overall_condition: str  # "good", "fair", "poor", "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "infrastructure_visibility": round(self.infrastructure_visibility, 6),
            "infrastructure_coverage": round(self.infrastructure_coverage, 6),
            "pole_count": self.pole_count,
            "insulator_anomaly_score": round(self.insulator_anomaly_score, 6),
            "overall_condition": self.overall_condition,
        }


@dataclasses.dataclass
class AnomalyResult:
    """Composite anomaly detection output."""

    wire_sag: WireSagResult
    equipment: EquipmentState
    anomaly_score: float  # composite 0-1 (0=normal, 1=critical anomaly)
    anomaly_flags: list[str]  # list of detected anomaly types

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_sag": self.wire_sag.to_dict(),
            "equipment": self.equipment.to_dict(),
            "anomaly_score": round(self.anomaly_score, 6),
            "anomaly_flags": self.anomaly_flags,
        }


# ---------------------------------------------------------------------------
# Wire sag estimation
# ---------------------------------------------------------------------------


def _collect_wire_points(
    wire_detection: WireDetection,
) -> list[tuple[int, int]]:
    """Collect sampled (x, y) points along all detected wire lines."""
    points: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in wire_detection.wire_lines:
        n = max(abs(x2 - x1), abs(y2 - y1), 10)
        for i in range(n + 1):
            t = i / max(n, 1)
            px = int(x1 + (x2 - x1) * t)
            py = int(y1 + (y2 - y1) * t)
            points.append((px, py))
    return points


def _fit_catenary(
    points: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Fit a catenary curve y = a * cosh((x - h) / a) + k to the given points.

    Returns (a, h, k).  Uses least-squares fitting with a simplified
    approach: estimate *h* from midpoint, *k* from the minimum y among
    endpoints, and *a* from the sag magnitude.

    Falls back to a straight-line fit (a -> inf) when points do not exhibit
    meaningful curvature.
    """
    if len(points) < 2:
        return (1e6, 0.0, 0.0)

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)

    h = float((xs.min() + xs.max()) / 2.0)
    x_range = float(xs.max() - xs.min())
    if x_range < 1.0:
        # Vertical or near-vertical — no meaningful catenary
        k = float(ys.min())
        return (1e6, h, k)

    # Endpoints average vs midpoint sag
    left_mask = xs < (xs.min() + x_range * 0.25)
    right_mask = xs > (xs.max() - x_range * 0.25)
    mid_mask = (xs > (h - x_range * 0.15)) & (xs < (h + x_range * 0.15))

    if left_mask.any() and right_mask.any():
        endpoint_y = (float(ys[left_mask].mean()) + float(ys[right_mask].mean())) / 2.0
    else:
        endpoint_y = float((ys[0] + ys[-1]) / 2.0)

    if mid_mask.any():
        mid_y = float(ys[mid_mask].mean())
    else:
        mid_y = float(ys.mean())

    sag = mid_y - endpoint_y  # positive = wire sags downward (y increases)
    if abs(sag) < 1.0:
        # Essentially straight
        k = float(ys.mean())
        return (1e6, h, k)

    # From catenary formula: sag ≈ a * (cosh(L/(2a)) - 1) where L = x_range
    # For small sag/L ratio: sag ≈ L^2 / (8a)  →  a ≈ L^2 / (8 * sag)
    a = max(x_range**2 / (8.0 * abs(sag)), 1.0)
    if sag < 0:
        # Wire bows upward (unusual), flip sign
        a = -a

    k = endpoint_y - a * (math.cosh(0.0) - 1.0)  # at x=h, cosh(0)=1 → y=k

    return (a, h, k)


def _catenary_y(x: float, a: float, h: float, k: float) -> float:
    """Evaluate catenary at a given x."""
    arg = (x - h) / a if abs(a) > 1e-9 else 0.0
    # Clamp to avoid overflow
    arg = max(min(arg, 20.0), -20.0)
    return a * (math.cosh(arg) - 1.0) + k


def estimate_wire_sag(
    wire_detection: WireDetection,
    depth_result: DepthResult | None = None,
    image_height: int = 0,
) -> WireSagResult:
    """Estimate wire sag by comparing detected wire points to an ideal catenary.

    Parameters
    ----------
    wire_detection:
        Wire detection output from :func:`temporalci.vision.clearance.detect_wires`.
    depth_result:
        Optional depth map for 3D sag estimation.
    image_height:
        Image height in pixels (used for relative deviation).  If 0, uses
        the wire detection search zone height as a fallback.
    """
    if not wire_detection.wire_lines:
        return WireSagResult(
            sag_detected=False,
            max_deviation_px=0.0,
            max_deviation_relative=0.0,
            sag_location=[0, 0],
            expected_curve=[],
            actual_points=[],
            severity="normal",
        )

    actual_points = _collect_wire_points(wire_detection)
    if not actual_points:
        return WireSagResult(
            sag_detected=False,
            max_deviation_px=0.0,
            max_deviation_relative=0.0,
            sag_location=[0, 0],
            expected_curve=[],
            actual_points=[],
            severity="normal",
        )

    a, h, k = _fit_catenary(actual_points)

    # Generate expected curve at same x positions
    xs_unique = sorted({p[0] for p in actual_points})
    expected_curve: list[tuple[int, int]] = []
    for x in xs_unique:
        ey = int(round(_catenary_y(float(x), a, h, k)))
        expected_curve.append((x, ey))

    # Measure deviation of actual points from expected curve
    max_dev = 0.0
    max_dev_point = actual_points[0] if actual_points else (0, 0)
    for px, py in actual_points:
        expected_y = _catenary_y(float(px), a, h, k)
        dev = abs(py - expected_y)
        if dev > max_dev:
            max_dev = dev
            max_dev_point = (px, py)

    # Depth-based adjustment: scale deviation by relative depth at sag point
    if depth_result is not None and max_dev > 0:
        dh, dw = depth_result.depth_map.shape[:2]
        sy, sx = max_dev_point[1], max_dev_point[0]
        if 0 <= sy < dh and 0 <= sx < dw:
            depth_val = float(depth_result.depth_map[sy, sx])
            # Objects farther away (depth closer to 1) have less real sag
            # Scale deviation: multiply by inverse depth factor
            depth_factor = max(1.0 - depth_val * 0.5, 0.5)
            max_dev *= depth_factor

    effective_height = image_height if image_height > 0 else wire_detection.search_zone_h
    if effective_height <= 0:
        effective_height = 1  # avoid division by zero

    max_dev_relative = max_dev / effective_height

    # Classify severity
    if max_dev_relative >= _SAG_SEVERE_THRESHOLD:
        severity = "severe"
    elif max_dev_relative >= _SAG_MODERATE_THRESHOLD:
        severity = "moderate"
    else:
        severity = "normal"

    sag_detected = severity != "normal"

    return WireSagResult(
        sag_detected=sag_detected,
        max_deviation_px=max_dev,
        max_deviation_relative=max_dev_relative,
        sag_location=[max_dev_point[1], max_dev_point[0]],  # [y, x]
        expected_curve=expected_curve,
        actual_points=actual_points,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Equipment assessment
# ---------------------------------------------------------------------------


def assess_equipment(
    seg_result: SegmentationResult,
    depth_result: DepthResult | None = None,
) -> EquipmentState:
    """Assess infrastructure equipment condition from segmentation results.

    Parameters
    ----------
    seg_result:
        Segmentation result containing ``infrastructure_mask`` and ``seg_map``.
    depth_result:
        Optional depth map (currently used for visibility weighting).
    """
    infra_mask = seg_result.infrastructure_mask
    h, w = infra_mask.shape[:2]
    total_pixels = h * w

    if total_pixels == 0:
        return EquipmentState(
            infrastructure_visibility=0.0,
            infrastructure_coverage=0.0,
            pole_count=0,
            insulator_anomaly_score=0.0,
            overall_condition="unknown",
        )

    infrastructure_coverage = float(infra_mask.sum()) / total_pixels

    # Infrastructure visibility: weighted by how well-lit / close it is
    if depth_result is not None and infra_mask.any():
        # Closer infrastructure (lower depth) is more visible
        infra_depths = depth_result.depth_map[infra_mask]
        mean_depth = float(infra_depths.mean())
        # Visibility higher when infrastructure is close (depth near 0)
        depth_visibility = max(1.0 - mean_depth, 0.0)
        infrastructure_visibility = min(infrastructure_coverage * 5.0 + depth_visibility * 0.3, 1.0)
    else:
        # Without depth, use coverage as a proxy (scaled up)
        infrastructure_visibility = min(infrastructure_coverage * 5.0, 1.0)

    # Count poles/supports using connected components
    pole_count = _count_poles(seg_result)

    # Insulator anomaly score from edge density around infrastructure
    insulator_anomaly_score = _compute_insulator_anomaly(seg_result)

    # Overall condition assessment
    if infrastructure_coverage < 0.001:
        overall_condition = "unknown"
    elif insulator_anomaly_score > 0.7:
        overall_condition = "poor"
    elif insulator_anomaly_score > 0.4 or infrastructure_visibility < 0.3:
        overall_condition = "fair"
    else:
        overall_condition = "good"

    return EquipmentState(
        infrastructure_visibility=infrastructure_visibility,
        infrastructure_coverage=infrastructure_coverage,
        pole_count=pole_count,
        insulator_anomaly_score=insulator_anomaly_score,
        overall_condition=overall_condition,
    )


def _count_poles(seg_result: SegmentationResult) -> int:
    """Count distinct pole/support structures via connected components.

    Focuses on pole (93) and pylon (136) classes from ADE20K.
    """
    pole_ids = {93, 136}
    seg_map = seg_result.seg_map
    pole_mask = np.isin(seg_map, list(pole_ids)).astype(np.uint8)

    if pole_mask.sum() == 0:
        return 0

    num_labels, _ = cv2.connectedComponents(pole_mask, connectivity=8)
    # num_labels includes background (label 0)
    return max(int(num_labels) - 1, 0)


def _compute_insulator_anomaly(seg_result: SegmentationResult) -> float:
    """Compute insulator anomaly score from edge density around infrastructure.

    High edge density in infrastructure regions may indicate damage,
    corrosion, or structural irregularities.  Low infrastructure coverage
    yields 0.0 (no data to assess).
    """
    infra_mask = seg_result.infrastructure_mask
    if not infra_mask.any():
        return 0.0

    seg_map = seg_result.seg_map
    h, w = seg_map.shape[:2]

    # Create a grayscale-like image from class IDs for edge detection
    # Normalise seg_map to 0-255 for Sobel
    seg_norm = ((seg_map.astype(np.float32) / 150.0) * 255.0).astype(np.uint8)

    # Dilate infrastructure mask to include boundary regions
    kernel = np.ones((5, 5), dtype=np.uint8)
    infra_dilated = cv2.dilate(infra_mask.astype(np.uint8), kernel, iterations=2)

    # Compute edges in the infrastructure neighbourhood
    edges = cv2.Canny(seg_norm, 30, 100)
    infra_edges = edges & (infra_dilated > 0)

    infra_area = float(infra_dilated.sum())
    if infra_area < 1:
        return 0.0

    edge_density = float(infra_edges.sum()) / infra_area
    # Normalise: typical edge density around 0.05-0.15 is normal
    # Above ~0.3 indicates anomalous texture
    anomaly_score = min(edge_density / 0.3, 1.0)
    return float(anomaly_score)


# ---------------------------------------------------------------------------
# Composite anomaly detection
# ---------------------------------------------------------------------------


def detect_anomalies(
    seg_result: SegmentationResult,
    wire_detection: WireDetection,
    depth_result: DepthResult | None = None,
    image_height: int = 0,
) -> AnomalyResult:
    """Run full anomaly detection combining wire sag and equipment assessment.

    Parameters
    ----------
    seg_result:
        Segmentation result from :class:`~temporalci.vision.segmentation.SegmentationModel`.
    wire_detection:
        Wire detection from :func:`~temporalci.vision.clearance.detect_wires`.
    depth_result:
        Optional depth result from :class:`~temporalci.vision.depth.DepthModel`.
    image_height:
        Image height in pixels for relative calculations.
    """
    wire_sag = estimate_wire_sag(wire_detection, depth_result, image_height)
    equipment = assess_equipment(seg_result, depth_result)

    anomaly_flags: list[str] = []

    # Wire sag flags
    if wire_sag.severity == "moderate":
        anomaly_flags.append("wire_sag_moderate")
    elif wire_sag.severity == "severe":
        anomaly_flags.append("wire_sag_severe")

    # Infrastructure visibility flags
    if equipment.infrastructure_visibility < 0.2:
        anomaly_flags.append("low_infrastructure_visibility")

    # Equipment condition flags
    if equipment.overall_condition == "poor":
        anomaly_flags.append("equipment_poor")
    elif equipment.overall_condition == "fair":
        anomaly_flags.append("equipment_fair")

    # High insulator anomaly
    if equipment.insulator_anomaly_score > 0.6:
        anomaly_flags.append("insulator_anomaly_high")

    # Composite anomaly score (0 = normal, 1 = critical)
    sag_score = 0.0
    if wire_sag.severity == "moderate":
        sag_score = 0.4
    elif wire_sag.severity == "severe":
        sag_score = 0.8

    equip_score = 0.0
    if equipment.overall_condition == "poor":
        equip_score = 0.7
    elif equipment.overall_condition == "fair":
        equip_score = 0.3
    elif equipment.overall_condition == "unknown":
        equip_score = 0.1

    insulator_contrib = equipment.insulator_anomaly_score * 0.3
    visibility_penalty = max(0.3 - equipment.infrastructure_visibility, 0.0)

    anomaly_score = min(
        sag_score * 0.4 + equip_score * 0.3 + insulator_contrib + visibility_penalty * 0.5,
        1.0,
    )

    return AnomalyResult(
        wire_sag=wire_sag,
        equipment=equipment,
        anomaly_score=anomaly_score,
        anomaly_flags=anomaly_flags,
    )
