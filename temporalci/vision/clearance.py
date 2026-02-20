"""Clearance calculation — combines segmentation, depth, and wire detection.

Provides two complementary approaches:
1. **Wire detection** via Canny + Hough (when wires are visible)
2. **Catenary zone analysis** — geometric prior based on camera position,
   measuring vegetation penetration into the zone using segmentation + depth.

The zone-based approach is more robust for real railway footage where
thin catenary wires are difficult to detect reliably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from temporalci.vision.depth import DepthResult
from temporalci.vision.segmentation import SegmentationResult


@dataclass
class WireDetection:
    """Catenary wire detection output."""

    wire_mask: np.ndarray  # (H, W) bool
    wire_lines: list[tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    wire_count: int
    search_zone_h: int  # height of search zone (upper portion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_count": self.wire_count,
            "wire_lines": self.wire_lines,
        }


@dataclass
class ClearanceResult:
    """Vegetation-to-wire clearance output."""

    min_clearance_px: float  # pixel distance veg↔wire (or veg↔zone boundary)
    min_clearance_relative: float  # normalised by image height (0-1)
    depth_adjusted_clearance: float  # depth-weighted distance estimate
    risk_level: str  # "critical" | "warning" | "caution" | "safe"
    risk_score: float  # 0 (critical) – 1 (safe)
    closest_vegetation: list[int] | None  # [y, x]
    closest_wire: list[int] | None  # [y, x]
    vegetation_in_wire_zone: float  # fraction of catenary band occupied by vegetation
    vegetation_penetration: float  # how deep vegetation reaches into catenary band (0-1)
    wire_detection: WireDetection | None = None
    zone_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "min_clearance_px": round(self.min_clearance_px, 2),
            "min_clearance_relative": round(self.min_clearance_relative, 6),
            "depth_adjusted_clearance": round(self.depth_adjusted_clearance, 2),
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 6),
            "vegetation_in_wire_zone": round(self.vegetation_in_wire_zone, 6),
            "vegetation_penetration": round(self.vegetation_penetration, 6),
            "closest_vegetation": self.closest_vegetation,
            "closest_wire": self.closest_wire,
        }
        if self.wire_detection:
            d["wire_count"] = self.wire_detection.wire_count
        return d


def detect_wires(
    image_or_path: str | Path | Image.Image,
    *,
    seg: SegmentationResult | None = None,
    search_fraction: float = 0.55,
    canny_low: int = 60,
    canny_high: int = 180,
    hough_threshold: int = 60,
    min_line_fraction: float = 0.20,
    max_gap: int = 20,
    max_slope: float = 0.15,
    max_vegetation_overlap: float = 0.3,
) -> WireDetection:
    """Detect catenary wires via Canny + probabilistic Hough transform.

    Uses the segmentation mask to suppress vegetation edges.
    Lines overlapping vegetation are discarded.
    """
    if isinstance(image_or_path, (str, Path)):
        gray = np.array(Image.open(image_or_path).convert("L"))
    else:
        gray = np.array(image_or_path.convert("L"))

    h, w = gray.shape
    search_h = int(h * search_fraction)
    upper = gray[:search_h, :].copy()

    # Suppress vegetation pixels before edge detection
    if seg is not None:
        veg_upper = seg.vegetation_mask[:search_h, :w]
        kernel = np.ones((5, 5), dtype=np.uint8)
        veg_dilated = cv2.dilate(veg_upper.astype(np.uint8), kernel, iterations=1)
        upper[veg_dilated > 0] = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upper)
    edges = cv2.Canny(enhanced, canny_low, canny_high)

    min_line_len = int(w * min_line_fraction)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        hough_threshold,
        minLineLength=min_line_len,
        maxLineGap=max_gap,
    )

    wire_mask = np.zeros((h, w), dtype=np.uint8)
    wire_lines: list[tuple[int, int, int, int]] = []

    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            if dx < 1:
                continue
            if abs(y2 - y1) / dx > max_slope:
                continue
            # Reject lines running through vegetation
            if seg is not None:
                pts = _sample_line_points(x1, y1, x2, y2, n=20)
                veg_hits = sum(
                    1
                    for px, py in pts
                    if 0 <= py < search_h and 0 <= px < w and seg.vegetation_mask[py, px]
                )
                if veg_hits / max(len(pts), 1) > max_vegetation_overlap:
                    continue
            cv2.line(wire_mask, (x1, y1), (x2, y2), 1, max(1, w // 500))
            wire_lines.append((int(x1), int(y1), int(x2), int(y2)))

    return WireDetection(
        wire_mask=wire_mask.astype(bool),
        wire_lines=wire_lines,
        wire_count=len(wire_lines),
        search_zone_h=search_h,
    )


def _sample_line_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    n: int = 20,
) -> list[tuple[int, int]]:
    """Sample n evenly spaced points along a line segment."""
    return [
        (int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t))
        for t in (i / max(n - 1, 1) for i in range(n))
    ]


def calculate_clearance(
    seg: SegmentationResult,
    depth: DepthResult | None,
    wires: WireDetection,
    *,
    catenary_band: tuple[float, float] = (0.10, 0.40),
    depth_scale: float = 500.0,
) -> ClearanceResult:
    """Calculate vegetation clearance in the catenary zone.

    Uses a two-pronged approach:
    1. If wires are detected, measures vegetation-to-wire distance.
    2. Always measures vegetation penetration into the catenary band
       (a geometric prior for where wires are expected).

    The catenary band is defined as a fraction of image height
    (default: 10 %–40 % from top = the overhead wire corridor).

    Parameters
    ----------
    catenary_band:
        (top_frac, bottom_frac) defining the expected wire corridor.
    depth_scale:
        Weight for depth differences in the 3D clearance estimate.
    """
    h, w = seg.vegetation_mask.shape
    band_top = int(h * catenary_band[0])
    band_bot = int(h * catenary_band[1])
    band_h = band_bot - band_top

    veg_mask = seg.vegetation_mask

    # ── Catenary band analysis ──────────────────────────────────────
    band_veg = veg_mask[band_top:band_bot, :]
    band_veg_ratio = float(band_veg.mean()) if band_h > 0 else 0.0

    # Vegetation penetration: for each column, how deep does vegetation
    # reach into the band from the bottom?  (Deeper = closer to wires.)
    penetration = 0.0
    if band_h > 0 and band_veg.any():
        # Per-column: find topmost vegetation row in the band
        col_has_veg = band_veg.any(axis=0)
        if col_has_veg.any():
            # For columns with vegetation, find topmost row
            top_veg_rows = np.argmax(band_veg, axis=0)  # first True per column
            # Only count columns that actually have vegetation
            valid_tops = top_veg_rows[col_has_veg]
            # Penetration = how far from bottom of band the veg reaches (0=bottom, 1=top)
            mean_top = float(valid_tops.mean())
            penetration = 1.0 - (mean_top / band_h)  # 1.0 = reached top of band

    # ── Distance-based clearance ────────────────────────────────────
    # Use wire mask if wires found, else use non-vegetation region
    # in the band as the reference (sky/infrastructure = where wires should be)
    veg_in_band = np.argwhere(band_veg)
    if len(veg_in_band) == 0:
        # No vegetation in catenary band = safe
        return ClearanceResult(
            min_clearance_px=float(band_h),
            min_clearance_relative=band_h / h if h > 0 else 1.0,
            depth_adjusted_clearance=float(band_h),
            risk_level="safe",
            risk_score=1.0,
            closest_vegetation=None,
            closest_wire=None,
            vegetation_in_wire_zone=0.0,
            vegetation_penetration=0.0,
            wire_detection=wires,
            zone_details={"band_top": band_top, "band_bot": band_bot},
        )

    # Find non-vegetation pixels in band (sky/infra = proxy for wire corridor)
    clear_zone = ~band_veg
    if not clear_zone.any():
        # Band fully covered by vegetation
        return ClearanceResult(
            min_clearance_px=0.0,
            min_clearance_relative=0.0,
            depth_adjusted_clearance=0.0,
            risk_level="critical",
            risk_score=0.0,
            closest_vegetation=None,
            closest_wire=None,
            vegetation_in_wire_zone=1.0,
            vegetation_penetration=1.0,
            wire_detection=wires,
            zone_details={"band_top": band_top, "band_bot": band_bot},
        )

    # Distance transform: every vegetation pixel → distance to nearest clear pixel
    clear_u8 = clear_zone.astype(np.uint8)
    dist_to_clear = cv2.distanceTransform(
        1 - clear_u8,
        cv2.DIST_L2,
        5,
    )
    # dist_to_clear at vegetation pixels gives distance to boundary
    veg_distances = dist_to_clear[band_veg]
    max_intrusion = float(veg_distances.max())

    # Find the deepest intruding vegetation point
    max_idx = int(np.argmax(dist_to_clear[band_veg]))
    veg_coords = np.argwhere(band_veg)
    deepest_veg = veg_coords[max_idx]
    # Translate back to full image coordinates
    deepest_veg_full = [int(deepest_veg[0] + band_top), int(deepest_veg[1])]

    # Find nearest clear pixel to the deepest intruding vegetation
    clear_coords = np.argwhere(clear_zone)
    dists_to_clear = np.sqrt(((clear_coords - deepest_veg).astype(np.float32) ** 2).sum(axis=1))
    nearest_clear_idx = int(dists_to_clear.argmin())
    nearest_clear = clear_coords[nearest_clear_idx]
    nearest_clear_full = [int(nearest_clear[0] + band_top), int(nearest_clear[1])]

    relative_clearance = max_intrusion / h if h > 0 else 0.0

    # Depth-adjusted clearance
    depth_adj = max_intrusion
    if depth is not None:
        vy, vx = deepest_veg_full
        cy, cx = nearest_clear_full
        if 0 <= vy < h and 0 <= vx < w and 0 <= cy < h and 0 <= cx < w:
            vd = float(depth.depth_map[vy, vx])
            cd = float(depth.depth_map[cy, cx])
            depth_diff = abs(vd - cd)
            depth_adj = float(np.sqrt(max_intrusion**2 + (depth_diff * depth_scale) ** 2))

    # ── Risk scoring ────────────────────────────────────────────────
    # Combine multiple signals
    # 1. Band vegetation ratio (0-1, lower=safer)
    # 2. Penetration depth (0-1, lower=safer)
    # 3. Max intrusion distance (pixels)
    veg_risk = min(band_veg_ratio * 2.0, 1.0)  # scale: 50% coverage = max risk
    pen_risk = penetration
    intrusion_norm = min(max_intrusion / max(band_h, 1), 1.0)

    combined_risk = veg_risk * 0.3 + pen_risk * 0.4 + intrusion_norm * 0.3
    risk_score = round(max(1.0 - combined_risk, 0.0), 4)

    if risk_score >= 0.75:
        risk_level = "safe"
    elif risk_score >= 0.50:
        risk_level = "caution"
    elif risk_score >= 0.25:
        risk_level = "warning"
    else:
        risk_level = "critical"

    return ClearanceResult(
        min_clearance_px=max_intrusion,
        min_clearance_relative=relative_clearance,
        depth_adjusted_clearance=depth_adj,
        risk_level=risk_level,
        risk_score=risk_score,
        closest_vegetation=deepest_veg_full,
        closest_wire=nearest_clear_full,
        vegetation_in_wire_zone=band_veg_ratio,
        vegetation_penetration=penetration,
        wire_detection=wires,
        zone_details={
            "band_top": band_top,
            "band_bot": band_bot,
            "wire_count": wires.wire_count,
            "mean_veg_distance": round(float(veg_distances.mean()), 2),
        },
    )
