"""Catenary vegetation proximity metric.

Analyses image frames to detect vegetation encroachment near overhead
catenary wires.  Three dimensions are computed per frame:

- **vegetation_proximity** — green pixel density in the catenary zone
  (upper quarter).  *Higher = more dangerous.*
- **green_coverage** — green pixel ratio in the upper half.
- **catenary_visibility** — edge/contrast density in the upper half
  (image-quality indicator).

The composite ``score`` is oriented so that **higher = safer**, consistent
with the TemporalCI gate convention for ``>=`` checks.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.types import GeneratedSample
from temporalci.utils import clamp

try:
    import numpy as np
    from PIL import Image

    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

ALL_DIMS = ["vegetation_proximity", "green_coverage", "catenary_visibility"]

# HSV-space green detection thresholds (PIL scale: H 0-255, S 0-255, V 0-255).
# H=35-120 covers ~50°-170° (yellow-green through green).
# S>15 excludes near-gray pixels; V>25 excludes very dark pixels.
_GREEN_H_LO = 35
_GREEN_H_HI = 120
_GREEN_S_MIN = 15
_GREEN_V_MIN = 25


def _load_image(path: str) -> "np.ndarray":
    """Load an image file and return an RGB uint8 numpy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _green_mask(pixels: "np.ndarray") -> "np.ndarray":
    """Return a boolean mask where pixels are classified as green.

    Uses HSV colour space for robust detection under varying lighting
    conditions.  This avoids the false positives that a simple RGB
    heuristic (``G > R+margin``) produces on desaturated bright pixels.
    """
    img = Image.fromarray(pixels, "RGB")
    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h >= _GREEN_H_LO) & (h <= _GREEN_H_HI) & (s > _GREEN_S_MIN) & (v > _GREEN_V_MIN)


def _edge_density(gray: "np.ndarray") -> float:
    """Compute a simple edge density using horizontal+vertical Sobel-like diffs."""
    dx = np.abs(np.diff(gray.astype(np.int16), axis=1))
    dy = np.abs(np.diff(gray.astype(np.int16), axis=0))
    edge_pixels = (dx > 20).sum() + (dy > 20).sum()
    total = gray.shape[0] * gray.shape[1]
    if total == 0:
        return 0.0
    return float(edge_pixels / total)


def _analyse_frame(pixels: "np.ndarray") -> dict[str, float]:
    """Analyse a single RGB frame and return the three dimension scores."""
    h = pixels.shape[0]
    upper_half = pixels[: h // 2]
    upper_quarter = pixels[: h // 4] if h >= 4 else upper_half

    # Vegetation proximity: green density in the catenary zone (upper quarter).
    green_quarter = _green_mask(upper_quarter)
    total_quarter = green_quarter.size
    vegetation_proximity = float(green_quarter.sum() / total_quarter) if total_quarter else 0.0

    # Green coverage: green ratio in upper half.
    green_half = _green_mask(upper_half)
    total_half = green_half.size
    green_coverage = float(green_half.sum() / total_half) if total_half else 0.0

    # Catenary visibility: edge density in upper half (grayscale).
    gray_half = np.mean(upper_half, axis=2).astype(np.uint8)
    catenary_visibility = clamp(_edge_density(gray_half) * 5.0)

    return {
        "vegetation_proximity": round(vegetation_proximity, 6),
        "green_coverage": round(green_coverage, 6),
        "catenary_visibility": round(catenary_visibility, 6),
    }


def evaluate(
    samples: list[GeneratedSample], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate catenary vegetation proximity across all samples."""
    if not _HAS_DEPS:
        raise RuntimeError(
            "catenary_vegetation metric requires Pillow and numpy. "
            "Install with: pip install temporalci[catenary]"
        )

    params = params or {}
    proximity_threshold = float(params.get("proximity_threshold", 0.3))

    if not samples:
        dim_scores = {dim: 0.0 for dim in ALL_DIMS}
        return {"score": 0.0, "dims": dim_scores, "sample_count": 0, "alert_frames": []}

    per_dim_values: dict[str, list[float]] = {dim: [] for dim in ALL_DIMS}
    per_sample: list[dict[str, Any]] = []
    alert_frames: list[dict[str, Any]] = []

    for sample in samples:
        frame_path = sample.video_path
        if not Path(frame_path).is_file():
            per_sample.append(
                {
                    "test_id": sample.test_id,
                    "prompt": sample.prompt,
                    "seed": sample.seed,
                    "dims": {dim: 0.0 for dim in ALL_DIMS},
                    "error": f"file not found: {frame_path}",
                }
            )
            for dim in ALL_DIMS:
                per_dim_values[dim].append(0.0)
            continue

        pixels = _load_image(frame_path)
        dims = _analyse_frame(pixels)

        for dim in ALL_DIMS:
            per_dim_values[dim].append(dims[dim])

        per_sample.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "sample_id": sample.metadata.get("sample_id"),
                "dims": dims,
            }
        )

        if dims["vegetation_proximity"] > proximity_threshold:
            alert_frames.append(
                {
                    "prompt": sample.prompt,
                    "frame": frame_path,
                    "vegetation_proximity": dims["vegetation_proximity"],
                }
            )

    dim_scores = {dim: round(mean(values), 6) for dim, values in per_dim_values.items()}

    # Composite: higher = safer.
    prox = dim_scores.get("vegetation_proximity", 0.0)
    cov = dim_scores.get("green_coverage", 0.0)
    vis = dim_scores.get("catenary_visibility", 0.0)
    score = clamp((1.0 - prox) * 0.6 + (1.0 - cov) * 0.2 + vis * 0.2)

    return {
        "score": round(score, 6),
        "dims": dim_scores,
        "sample_count": len(samples),
        "per_sample": per_sample,
        "alert_frames": alert_frames,
    }
