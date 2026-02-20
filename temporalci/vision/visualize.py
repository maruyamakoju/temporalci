"""Multi-panel visualization for the 3-layer vision pipeline.

Generates a composite image with four panels:
1. Original frame
2. Semantic segmentation overlay
3. Depth map (coloured)
4. Clearance heatmap (risk zones + wire + vegetation overlay)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from temporalci.vision.clearance import ClearanceResult, WireDetection
from temporalci.vision.depth import DepthResult
from temporalci.vision.segmentation import SegmentationResult

# Colour palette for segmentation overlay (RGBA)
_SEG_COLORS: dict[str, tuple[int, int, int, int]] = {
    "vegetation": (34, 197, 94, 140),  # green
    "sky": (96, 165, 250, 100),  # blue
    "infrastructure": (251, 146, 60, 120),  # orange
    "ground": (168, 162, 158, 80),  # gray
}

# Risk level colours for clearance heatmap
_RISK_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (239, 68, 68),  # red
    "warning": (251, 146, 60),  # orange
    "caution": (250, 204, 21),  # yellow
    "safe": (34, 197, 94),  # green
}


def _segmentation_overlay(
    original: np.ndarray,
    seg: SegmentationResult,
) -> np.ndarray:
    """Create segmentation overlay on original image."""
    h, w = original.shape[:2]
    overlay = original.copy()

    masks_and_colors = [
        (seg.vegetation_mask, _SEG_COLORS["vegetation"]),
        (seg.sky_mask, _SEG_COLORS["sky"]),
        (seg.infrastructure_mask, _SEG_COLORS["infrastructure"]),
        (seg.ground_mask, _SEG_COLORS["ground"]),
    ]

    for mask, (r, g, b, a) in masks_and_colors:
        alpha = a / 255.0
        colour = np.array([b, g, r], dtype=np.float32)  # BGR for OpenCV
        region = mask[:h, :w]
        overlay[region] = (
            overlay[region].astype(np.float32) * (1 - alpha) + colour * alpha
        ).astype(np.uint8)

    return overlay


def _depth_colormap(depth: DepthResult) -> np.ndarray:
    """Create a coloured depth map visualisation."""
    d = (depth.depth_map * 255).astype(np.uint8)
    # INFERNO: near (dark) → far (bright yellow)
    coloured = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    return coloured


def _clearance_heatmap(
    original: np.ndarray,
    seg: SegmentationResult,
    depth: DepthResult | None,
    wires: WireDetection,
    clearance: ClearanceResult,
) -> np.ndarray:
    """Create clearance risk heatmap with catenary band and vegetation overlay."""
    h, w = original.shape[:2]
    canvas = original.copy()

    # Extract catenary band boundaries from zone_details
    band_top = clearance.zone_details.get("band_top", int(h * 0.10))
    band_bot = clearance.zone_details.get("band_bot", int(h * 0.40))

    # 1. Tint vegetation in the catenary band by risk level
    veg_mask = seg.vegetation_mask[:h, :w]
    band_veg = veg_mask.copy()
    band_veg[:band_top, :] = False
    band_veg[band_bot:, :] = False

    risk_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    risk_alpha = np.zeros((h, w), dtype=np.float32)

    if band_veg.any():
        # Vegetation inside catenary band: colour by vertical position
        # (higher = closer to wires = more dangerous)
        for row in range(band_top, band_bot):
            if not veg_mask[row, :].any():
                continue
            # normalise: 0 at band_bot (far from wires), 1 at band_top (near wires)
            pos = 1.0 - (row - band_top) / max(band_bot - band_top, 1)
            col_mask = veg_mask[row, :]
            if pos > 0.6:
                risk_overlay[row, col_mask] = [68, 68, 239]  # red
                risk_alpha[row, col_mask] = 0.65
            elif pos > 0.3:
                risk_overlay[row, col_mask] = [60, 146, 251]  # orange
                risk_alpha[row, col_mask] = 0.5
            else:
                risk_overlay[row, col_mask] = [21, 204, 250]  # yellow
                risk_alpha[row, col_mask] = 0.4

    # Vegetation outside band: subtle green tint
    outside_veg = veg_mask & ~band_veg
    risk_overlay[outside_veg] = [94, 197, 34]
    risk_alpha[outside_veg] = 0.2

    alpha_3d = risk_alpha[:, :, None]
    canvas = (
        canvas.astype(np.float32) * (1 - alpha_3d) + risk_overlay.astype(np.float32) * alpha_3d
    ).astype(np.uint8)

    # 2. Draw catenary band boundaries (dashed cyan lines)
    for band_y in [band_top, band_bot]:
        for x_start in range(0, w, 20):
            x_end = min(x_start + 12, w)
            cv2.line(canvas, (x_start, band_y), (x_end, band_y), (255, 255, 0), 2)

    # 3. Draw detected wires in bright cyan
    for x1, y1, x2, y2 in wires.wire_lines:
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)

    # 4. Draw deepest intrusion line
    if clearance.closest_vegetation and clearance.closest_wire:
        vy, vx = clearance.closest_vegetation
        wy, wx = clearance.closest_wire
        cv2.line(canvas, (vx, vy), (wx, wy), (0, 0, 255), 2)
        cv2.circle(canvas, (vx, vy), 6, (0, 0, 255), -1)  # red dot on vegetation
        cv2.circle(canvas, (wx, wy), 6, (255, 255, 0), -1)  # cyan dot on clear zone

    # 5. Risk label on image
    risk_color = _RISK_COLORS.get(clearance.risk_level, (200, 200, 200))
    label = f"{clearance.risk_level.upper()} ({clearance.risk_score:.2f})"
    cv2.putText(
        canvas,
        label,
        (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (risk_color[2], risk_color[1], risk_color[0]),
        2,
        cv2.LINE_AA,
    )

    return canvas


def _add_label(
    panel: np.ndarray,
    label: str,
    *,
    font_scale: float = 0.6,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a text label to the top-left of a panel."""
    out = panel.copy()
    thickness = 1
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tw, th = size[0]
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + pad * 2, th + pad * 2), bg_color, -1)
    cv2.putText(
        out,
        label,
        (pad, th + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        fg_color,
        thickness,
        cv2.LINE_AA,
    )
    return out


def _add_stats_bar(
    composite: np.ndarray,
    clearance: ClearanceResult,
    seg: SegmentationResult,
    depth: DepthResult | None,
) -> np.ndarray:
    """Add a stats bar at the bottom of the composite image."""
    h, w = composite.shape[:2]
    bar_h = 36
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    risk_color = _RISK_COLORS.get(clearance.risk_level, (200, 200, 200))
    # Risk level badge
    cv2.rectangle(bar, (0, 0), (w, bar_h), (30, 30, 30), -1)

    parts = [
        f"RISK: {clearance.risk_level.upper()}",
        f"Score: {clearance.risk_score:.2f}",
        f"Clearance: {clearance.min_clearance_px:.0f}px ({clearance.min_clearance_relative:.3f})",
        f"Veg zone: {clearance.vegetation_in_wire_zone:.1%}",
        f"Veg total: {seg.vegetation_ratio:.1%}",
        f"Wires: {clearance.wire_detection.wire_count if clearance.wire_detection else 0}",
    ]
    text = "  |  ".join(parts)

    cv2.putText(
        bar,
        text,
        (10, bar_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (risk_color[2], risk_color[1], risk_color[0]),  # BGR
        1,
        cv2.LINE_AA,
    )

    return np.vstack([composite, bar])


def generate_panel(
    image_path: str | Path,
    seg: SegmentationResult,
    depth: DepthResult | None,
    wires: WireDetection,
    clearance: ClearanceResult,
    *,
    output_path: str | Path | None = None,
    panel_width: int = 640,
) -> np.ndarray:
    """Generate a 2x2 multi-panel visualization.

    Returns the composite image as a numpy array (BGR).
    Optionally saves to ``output_path``.
    """
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    oh, ow = original.shape[:2]
    aspect = oh / ow
    panel_h = int(panel_width * aspect)

    def _resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (panel_width, panel_h), interpolation=cv2.INTER_AREA)

    # Four panels
    p1 = _add_label(_resize(original), "Original")
    p2 = _add_label(_resize(_segmentation_overlay(original, seg)), "Segmentation")

    if depth is not None:
        depth_vis = _depth_colormap(depth)
        # Resize depth vis to match original dimensions if needed
        if depth_vis.shape[:2] != (oh, ow):
            depth_vis = cv2.resize(depth_vis, (ow, oh))
        p3 = _add_label(_resize(depth_vis), "Depth (near=dark, far=bright)")
    else:
        p3 = _add_label(
            _resize(np.full_like(original, 40)),
            "Depth: N/A",
        )

    p4 = _add_label(
        _resize(_clearance_heatmap(original, seg, depth, wires, clearance)),
        f"Clearance [{clearance.risk_level.upper()}]",
    )

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    composite = np.vstack([top_row, bottom_row])

    composite = _add_stats_bar(composite, clearance, seg, depth)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), composite)

    return composite
