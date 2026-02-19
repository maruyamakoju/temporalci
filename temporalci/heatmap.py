"""Heatmap overlay generator for catenary vegetation detection.

Produces annotated images that show exactly which pixels the HSV green
detector classified as vegetation, overlaid on the original frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
    from PIL import Image, ImageDraw

    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from temporalci.metrics.catenary_vegetation import (
    _GREEN_H_HI,
    _GREEN_H_LO,
    _GREEN_S_MIN,
    _GREEN_V_MIN,
)


def _green_mask(pixels: "np.ndarray") -> "np.ndarray":
    """Reproduce the same green detection used by the metric."""
    img = Image.fromarray(pixels, "RGB")
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h >= _GREEN_H_LO) & (h <= _GREEN_H_HI) & (s > _GREEN_S_MIN) & (v > _GREEN_V_MIN)


def generate_heatmap(
    frame_path: str | Path,
    output_path: str | Path,
    *,
    overlay_alpha: float = 0.45,
    zone_line: bool = True,
) -> dict[str, Any]:
    """Generate a heatmap overlay image for a single frame.

    Parameters
    ----------
    frame_path:
        Path to the source image.
    output_path:
        Where to write the annotated PNG.
    overlay_alpha:
        Opacity of the red detection overlay (0-1).
    zone_line:
        If *True*, draw a cyan dashed line at the 1/4 height boundary
        marking the catenary zone.

    Returns
    -------
    dict with ``green_ratio_quarter``, ``green_ratio_half``, and
    ``output_path``.
    """
    if not _HAS_DEPS:
        raise RuntimeError("heatmap requires Pillow and numpy")

    src = Image.open(str(frame_path)).convert("RGB")
    pixels = np.asarray(src, dtype=np.uint8)
    h, w = pixels.shape[:2]

    mask = _green_mask(pixels)

    # Build overlay: red channel where green detected.
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    alpha_val = int(overlay_alpha * 255)
    overlay[mask, 0] = 220  # R
    overlay[mask, 1] = 40  # G
    overlay[mask, 2] = 40  # B
    overlay[mask, 3] = alpha_val

    overlay_img = Image.fromarray(overlay, "RGBA")
    base = src.convert("RGBA")
    composite = Image.alpha_composite(base, overlay_img).convert("RGB")

    # Draw zone boundary lines.
    draw = ImageDraw.Draw(composite)
    if zone_line:
        quarter_y = h // 4
        half_y = h // 2
        # Catenary zone upper boundary (1/4 height) — cyan dashed.
        _draw_dashed_line(draw, 0, quarter_y, w, quarter_y, fill=(0, 220, 255), width=2, dash=12)
        # Upper half boundary — lighter.
        _draw_dashed_line(draw, 0, half_y, w, half_y, fill=(100, 180, 220), width=1, dash=8)

    # Stats text in top-left corner.
    quarter_mask = mask[: h // 4]
    half_mask = mask[: h // 2]
    q_total = quarter_mask.size if quarter_mask.size else 1
    h_total = half_mask.size if half_mask.size else 1
    green_ratio_quarter = float(quarter_mask.sum() / q_total)
    green_ratio_half = float(half_mask.sum() / h_total)

    label = f"prox={green_ratio_quarter:.3f}  cov={green_ratio_half:.3f}"
    # Draw text with background.
    bbox = draw.textbbox((0, 0), label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([4, 4, text_w + 12, text_h + 12], fill=(0, 0, 0, 180))
    draw.text((8, 6), label, fill=(255, 255, 255))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(str(out), "PNG")

    return {
        "green_ratio_quarter": round(green_ratio_quarter, 6),
        "green_ratio_half": round(green_ratio_half, 6),
        "output_path": str(out),
    }


def generate_heatmaps(
    frame_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.jpg",
    overlay_alpha: float = 0.45,
) -> list[dict[str, Any]]:
    """Generate heatmaps for all frames in a directory.

    Returns a list of per-frame result dicts, sorted by filename.
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(p for p in frame_dir.glob(pattern) if p.is_file())
    results: list[dict[str, Any]] = []
    for frame in frames:
        out_name = f"{frame.stem}_heatmap.png"
        info = generate_heatmap(
            frame,
            output_dir / out_name,
            overlay_alpha=overlay_alpha,
        )
        info["source_frame"] = frame.name
        results.append(info)
    return results


def _draw_dashed_line(
    draw: "ImageDraw.ImageDraw",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    fill: tuple[int, int, int],
    width: int = 1,
    dash: int = 10,
) -> None:
    """Draw a horizontal dashed line."""
    x = x0
    drawing = True
    while x < x1:
        end = min(x + dash, x1)
        if drawing:
            draw.line([(x, y0), (end, y1)], fill=fill, width=width)
        x = end
        drawing = not drawing
