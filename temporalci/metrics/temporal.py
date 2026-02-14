from __future__ import annotations

from statistics import mean
from typing import Any

from temporalci.types import GeneratedSample
from temporalci.utils import clamp

ALL_DIMS = ["temporal_flicker", "motion_smoothness", "subject_consistency"]


def _score_stream(stream: list[float]) -> dict[str, float]:
    if len(stream) < 3:
        return {
            "temporal_flicker": 0.0,
            "motion_smoothness": 0.0,
            "subject_consistency": 0.0,
        }

    diffs = [abs(stream[i] - stream[i - 1]) for i in range(1, len(stream))]
    accel = [abs(diffs[i] - diffs[i - 1]) for i in range(1, len(diffs))]

    first = stream[: max(1, len(stream) // 4)]
    last = stream[-max(1, len(stream) // 4) :]

    temporal_flicker = clamp(1.0 - mean(diffs) * 3.0)
    motion_smoothness = clamp(1.0 - mean(accel) * 6.0)
    subject_consistency = clamp(1.0 - abs(mean(last) - mean(first)) * 3.0)

    return {
        "temporal_flicker": round(temporal_flicker, 6),
        "motion_smoothness": round(motion_smoothness, 6),
        "subject_consistency": round(subject_consistency, 6),
    }


def evaluate(samples: list[GeneratedSample], params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    requested_dims = params.get("dims", ALL_DIMS)
    if not isinstance(requested_dims, list):
        requested_dims = ALL_DIMS

    dims = [d for d in requested_dims if d in ALL_DIMS]
    if not dims:
        dims = ALL_DIMS

    if not samples:
        dim_scores = {dim: 0.0 for dim in dims}
        return {"score": 0.0, "dims": dim_scores, "sample_count": 0}

    per_dim_values: dict[str, list[float]] = {dim: [] for dim in dims}
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        scored = _score_stream(sample.evaluation_stream)
        sample_dims = {}
        for dim in dims:
            value = scored[dim]
            per_dim_values[dim].append(value)
            sample_dims[dim] = value
        per_sample.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "sample_id": sample.metadata.get("sample_id"),
                "dims": sample_dims,
            }
        )

    dim_scores = {dim: round(mean(values), 6) for dim, values in per_dim_values.items()}
    total = mean(dim_scores.values()) if dim_scores else 0.0

    return {
        "score": round(total, 6),
        "dims": dim_scores,
        "sample_count": len(samples),
        "per_sample": per_sample,
    }
