"""3-layer catenary clearance metric.

Combines semantic segmentation (SegFormer), monocular depth estimation
(Depth Anything V2), and wire detection (Canny + Hough) to produce
physics-informed clearance measurements between vegetation and catenary
wires.

Output schema is compatible with the existing TemporalCI pipeline
(score, dims, per_sample, alert_frames).

Dimensions
----------
vegetation_proximity_nn
    Neural-network segmentation-based vegetation ratio in the catenary
    zone (upper 40 %).  Higher = more vegetation = more danger.
clearance_relative
    Minimum pixel distance between vegetation and detected wires,
    normalised by image height.  Lower = closer = more danger.
depth_clearance
    Depth-adjusted clearance estimate.  Incorporates monocular depth
    differences between closest vegetation and wire points.
risk_score
    Composite risk score (0 = critical, 1 = safe).
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.types import GeneratedSample
from temporalci.utils import clamp


ALL_DIMS = (
    "vegetation_proximity_nn",
    "vegetation_penetration",
    "clearance_relative",
    "depth_clearance",
    "risk_score",
)


_seg_model: Any = None
_depth_model: Any = None


def _load_models(params: dict[str, Any]) -> tuple[Any, Any]:
    """Lazy-load segmentation and depth models.

    Returns (seg_model, depth_model).  Models are cached on the module
    to avoid reloading across calls within the same process.
    """
    global _seg_model, _depth_model  # noqa: PLW0603

    device = str(params.get("device", "auto"))

    if _seg_model is None:
        from temporalci.vision.segmentation import SegmentationModel

        _seg_model = SegmentationModel(device=device)

    if _depth_model is None:
        skip_depth = str(params.get("skip_depth", "false")).lower() in (
            "true",
            "1",
            "yes",
        )
        if skip_depth:
            _depth_model = "skip"
        else:
            from temporalci.vision.depth import DepthModel

            _depth_model = DepthModel(device=device)

    dep = None if _depth_model == "skip" else _depth_model
    return _seg_model, dep


def evaluate(
    samples: list[GeneratedSample],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate catenary clearance using the 3-layer vision pipeline."""
    params = params or {}

    if not samples:
        return {
            "score": 0.0,
            "dims": {dim: 0.0 for dim in ALL_DIMS},
            "sample_count": 0,
            "per_sample": [],
            "alert_frames": [],
        }

    from temporalci.vision.clearance import (
        calculate_clearance,
        detect_wires,
    )
    from temporalci.vision.visualize import generate_panel

    seg_model, depth_model = _load_models(params)

    output_dir = params.get("output_dir")
    risk_threshold = float(params.get("risk_threshold", 0.5))

    per_dim_values: dict[str, list[float]] = {dim: [] for dim in ALL_DIMS}
    per_sample: list[dict[str, Any]] = []
    alert_frames: list[dict[str, Any]] = []

    for sample in samples:
        frame_path = sample.video_path
        if not Path(frame_path).is_file():
            entry = {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "dims": {dim: 0.0 for dim in ALL_DIMS},
                "error": f"file not found: {frame_path}",
            }
            per_sample.append(entry)
            for dim in ALL_DIMS:
                per_dim_values[dim].append(0.0)
            continue

        # Layer 1: Semantic segmentation
        seg_result = seg_model.segment(frame_path)

        # Layer 2: Depth estimation
        depth_result = None
        if depth_model is not None:
            depth_result = depth_model.estimate(frame_path)

        # Wire detection (segmentation-aware) + Layer 3: Clearance calculation
        wires = detect_wires(frame_path, seg=seg_result)
        clearance = calculate_clearance(seg_result, depth_result, wires)

        # Map to metric dimensions
        dims = {
            "vegetation_proximity_nn": seg_result.vegetation_upper_ratio,
            "vegetation_penetration": clearance.vegetation_penetration,
            "clearance_relative": clearance.min_clearance_relative,
            "depth_clearance": min(clearance.depth_adjusted_clearance / 100.0, 1.0),
            "risk_score": clearance.risk_score,
        }

        for dim in ALL_DIMS:
            per_dim_values[dim].append(dims[dim])

        entry = {
            "test_id": sample.test_id,
            "prompt": sample.prompt,
            "seed": sample.seed,
            "sample_id": sample.metadata.get("sample_id"),
            "dims": {k: round(v, 6) for k, v in dims.items()},
            "risk_level": clearance.risk_level,
            "wire_count": wires.wire_count,
            "clearance_px": round(clearance.min_clearance_px, 2),
            "seg_classes": sorted(seg_result.classes_found),
        }
        per_sample.append(entry)

        if clearance.risk_score < risk_threshold:
            alert_frames.append(
                {
                    "prompt": sample.prompt,
                    "frame": frame_path,
                    "risk_level": clearance.risk_level,
                    "risk_score": clearance.risk_score,
                    "clearance_px": round(clearance.min_clearance_px, 2),
                    "vegetation_zone": round(clearance.vegetation_in_wire_zone, 4),
                }
            )

        # Generate multi-panel visualisation if output_dir specified
        if output_dir:
            out_path = Path(output_dir) / f"{sample.prompt}_clearance.jpg"
            generate_panel(
                frame_path,
                seg_result,
                depth_result,
                wires,
                clearance,
                output_path=out_path,
            )

    # Aggregate
    dim_scores = {dim: round(mean(values), 6) for dim, values in per_dim_values.items()}

    # Composite score: use risk_score directly (it already combines all signals)
    risk = dim_scores.get("risk_score", 0.5)
    veg = dim_scores.get("vegetation_proximity_nn", 0.0)
    pen = dim_scores.get("vegetation_penetration", 0.0)

    score = clamp(risk * 0.6 + (1.0 - veg) * 0.2 + (1.0 - pen) * 0.2)

    return {
        "score": round(score, 6),
        "dims": dim_scores,
        "sample_count": len(samples),
        "per_sample": per_sample,
        "alert_frames": alert_frames,
    }
