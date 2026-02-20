"""ML-based catenary vegetation proximity metric (stub).

Drop-in replacement for the HSV heuristic metric.  Register this metric
in ``metrics/__init__.py`` and switch the suite YAML ``metrics.name`` to
``catenary_vegetation_ml`` to upgrade from colour-space heuristics to a
segmentation model.

**Migration checklist**:

1. Install a segmentation backend::

       pip install torch torchvision   # or onnxruntime for lighter inference

2. Place or download a model checkpoint.  The ``params.model_path``
   field in the suite YAML tells the metric where to find it.

3. Swap the metric name in your suite YAML::

       metrics:
         - name: "catenary_vegetation_ml"
           params:
             model_path: "models/deeplabv3_catenary.pt"
             device: "cpu"          # or "cuda"
             confidence: 0.5

4. Gates, reports, heatmaps, and all other pipeline components work
   unchanged — the metric output schema (score, dims, per_sample,
   alert_frames) is identical.

This file is a **working stub** that falls back to the HSV metric when
no model is loaded.  Replace ``_segment_frame()`` with real inference to
activate ML-based detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.types import GeneratedSample

# Re-use the HSV metric as fallback and for shared utilities.
from temporalci.metrics.catenary_vegetation import (
    ALL_DIMS,
    _analyse_frame,
    _load_image,
    evaluate as _hsv_evaluate,
)

_HAS_DEPS = bool(__import__("importlib").util.find_spec("PIL"))


def _load_model(params: dict[str, Any]) -> Any:
    """Load a segmentation model from ``params['model_path']``.

    Returns ``None`` if no model path is configured, which signals the
    metric to fall back to the HSV heuristic.

    Replace this stub with real model loading (e.g. torchvision
    DeepLabV3, ONNX Runtime, or a custom checkpoint).
    """
    model_path = params.get("model_path")
    if not model_path or not Path(str(model_path)).exists():
        return None

    # --- STUB: replace with real model loading ---
    # Example for DeepLabV3:
    #
    #   import torch
    #   from torchvision.models.segmentation import deeplabv3_resnet50
    #   model = deeplabv3_resnet50(num_classes=2)
    #   model.load_state_dict(torch.load(model_path, map_location=device))
    #   model.eval()
    #   return model
    #
    return None


def _segment_frame(
    model: Any,
    pixels: "Any",
    params: dict[str, Any],
) -> dict[str, float]:
    """Run ML segmentation on a single frame.

    Should return the same dict shape as ``_analyse_frame()``::

        {
            "vegetation_proximity": float,  # 0-1, higher = more vegetation
            "green_coverage": float,        # 0-1
            "catenary_visibility": float,   # 0-1
        }

    Replace this stub with real inference.
    """
    # --- STUB: replace with real inference ---
    # Example for DeepLabV3:
    #
    #   import torch
    #   from torchvision import transforms
    #   transform = transforms.Compose([
    #       transforms.ToTensor(),
    #       transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225]),
    #   ])
    #   input_tensor = transform(Image.fromarray(pixels)).unsqueeze(0)
    #   with torch.no_grad():
    #       output = model(input_tensor)["out"]
    #   mask = output.argmax(1).squeeze().numpy()
    #   # class 1 = vegetation
    #   h = pixels.shape[0]
    #   quarter_veg = mask[:h//4].mean()
    #   half_veg = mask[:h//2].mean()
    #   return {
    #       "vegetation_proximity": float(quarter_veg),
    #       "green_coverage": float(half_veg),
    #       "catenary_visibility": ...,
    #   }
    #
    # Fallback to HSV heuristic:
    return _analyse_frame(pixels)


def evaluate(
    samples: list[GeneratedSample], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate vegetation proximity using ML segmentation.

    Falls back to the HSV heuristic if no model is configured or loaded.
    The output schema is identical to ``catenary_vegetation.evaluate()``.
    """
    params = params or {}
    model = _load_model(params)

    if model is None:
        # No model available — fall back to HSV heuristic.
        return _hsv_evaluate(samples, params)

    # ML path (active when model is loaded).
    if not _HAS_DEPS:
        raise RuntimeError("catenary_vegetation_ml requires Pillow")

    from statistics import mean

    from temporalci.utils import clamp

    proximity_threshold = float(params.get("proximity_threshold", 0.05))

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
        dims = _segment_frame(model, pixels, params)

        for dim in ALL_DIMS:
            per_dim_values[dim].append(dims.get(dim, 0.0))

        per_sample.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "sample_id": sample.metadata.get("sample_id"),
                "dims": dims,
            }
        )

        if dims.get("vegetation_proximity", 0.0) > proximity_threshold:
            alert_frames.append(
                {
                    "prompt": sample.prompt,
                    "frame": frame_path,
                    "vegetation_proximity": dims["vegetation_proximity"],
                }
            )

    dim_scores = {dim: round(mean(values), 6) for dim, values in per_dim_values.items()}
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
