"""Catenary anomaly detection metric.

Evaluates wire sag and equipment condition anomalies for catenary
infrastructure frames.  Wraps the vision anomaly detection pipeline
into the standard TemporalCI metric interface.

Dimensions
----------
wire_sag_severity
    Numeric severity of wire sag (0 = normal, 0.5 = moderate, 1 = severe).
infrastructure_visibility
    How visible infrastructure is in the frame (0-1, higher = better).
equipment_condition
    Equipment condition score (0 = poor/unknown, 0.5 = fair, 1 = good).
anomaly_score
    Composite anomaly score (0 = normal, 1 = critical anomaly).
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.types import GeneratedSample
from temporalci.utils import clamp

ALL_DIMS = (
    "wire_sag_severity",
    "infrastructure_visibility",
    "equipment_condition",
    "anomaly_score",
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


_SEVERITY_MAP = {"normal": 0.0, "moderate": 0.5, "severe": 1.0}
_CONDITION_MAP = {"good": 1.0, "fair": 0.5, "poor": 0.0, "unknown": 0.0}


def evaluate(
    samples: list[GeneratedSample],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate catenary anomaly detection across all samples."""
    params = params or {}

    if not samples:
        return {
            "score": 0.0,
            "dims": {dim: 0.0 for dim in ALL_DIMS},
            "sample_count": 0,
            "per_sample": [],
            "alert_frames": [],
        }

    from temporalci.vision.anomaly import detect_anomalies
    from temporalci.vision.clearance import detect_wires

    seg_model, depth_model = _load_models(params)

    anomaly_threshold = float(params.get("anomaly_threshold", 0.4))

    per_dim_values: dict[str, list[float]] = {dim: [] for dim in ALL_DIMS}
    per_sample: list[dict[str, Any]] = []
    alert_frames: list[dict[str, Any]] = []

    for sample in samples:
        frame_path = sample.video_path
        if not Path(frame_path).is_file():
            entry: dict[str, Any] = {
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

        # Run vision pipeline
        seg_result = seg_model.segment(frame_path)

        depth_result = None
        if depth_model is not None:
            depth_result = depth_model.estimate(frame_path)

        wires = detect_wires(frame_path, seg=seg_result)

        from PIL import Image as _PILImage

        img = _PILImage.open(frame_path)
        img_h = img.height

        anomaly = detect_anomalies(seg_result, wires, depth_result, image_height=img_h)

        # Map to metric dimensions
        dims = {
            "wire_sag_severity": _SEVERITY_MAP.get(anomaly.wire_sag.severity, 0.0),
            "infrastructure_visibility": anomaly.equipment.infrastructure_visibility,
            "equipment_condition": _CONDITION_MAP.get(anomaly.equipment.overall_condition, 0.0),
            "anomaly_score": anomaly.anomaly_score,
        }

        for dim in ALL_DIMS:
            per_dim_values[dim].append(dims[dim])

        entry = {
            "test_id": sample.test_id,
            "prompt": sample.prompt,
            "seed": sample.seed,
            "sample_id": sample.metadata.get("sample_id"),
            "dims": {k: round(v, 6) for k, v in dims.items()},
            "anomaly_flags": anomaly.anomaly_flags,
            "wire_sag_severity": anomaly.wire_sag.severity,
            "equipment_condition": anomaly.equipment.overall_condition,
        }
        per_sample.append(entry)

        if anomaly.anomaly_score >= anomaly_threshold:
            alert_frames.append(
                {
                    "prompt": sample.prompt,
                    "frame": frame_path,
                    "anomaly_score": round(anomaly.anomaly_score, 4),
                    "anomaly_flags": anomaly.anomaly_flags,
                }
            )

    # Aggregate
    dim_scores = {dim: round(mean(values), 6) for dim, values in per_dim_values.items()}

    # Composite score: higher = safer (invert anomaly_score)
    anom = dim_scores.get("anomaly_score", 0.0)
    vis = dim_scores.get("infrastructure_visibility", 0.0)
    cond = dim_scores.get("equipment_condition", 0.0)
    sag = dim_scores.get("wire_sag_severity", 0.0)

    score = clamp((1.0 - anom) * 0.4 + vis * 0.2 + cond * 0.2 + (1.0 - sag) * 0.2)

    return {
        "score": round(score, 6),
        "dims": dim_scores,
        "sample_count": len(samples),
        "per_sample": per_sample,
        "alert_frames": alert_frames,
    }
