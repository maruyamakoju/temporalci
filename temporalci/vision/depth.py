"""Monocular depth estimation using Depth Anything V2.

Produces a per-pixel relative depth map from a single image.
Uses ``depth-anything/Depth-Anything-V2-Small-hf`` (≈99 MB).
Depth values are normalised to 0–1 where 0 = nearest, 1 = farthest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class DepthResult:
    """Per-frame depth estimation output."""

    depth_map: np.ndarray  # (H, W) float32, 0 (near) – 1 (far)
    depth_raw: np.ndarray  # (H, W) float32, original model output
    min_depth: float
    max_depth: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_depth": round(float(self.min_depth), 6),
            "max_depth": round(float(self.max_depth), 6),
            "mean_depth": round(float(self.depth_map.mean()), 6),
        }


class DepthModel:
    """Thin wrapper around HuggingFace Depth Anything V2."""

    def __init__(self, *, device: str = "auto") -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            _MODEL_NAME,
        ).to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def estimate(self, image_or_path: str | Path | Image.Image) -> DepthResult:
        """Estimate depth for a single image."""
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        w, h = image.size
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        prediction = self._model(**inputs).predicted_depth  # (1, Hm, Wm)

        # Upsample to original resolution
        depth = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            normalised = (depth - d_min) / (d_max - d_min)
        else:
            normalised = np.zeros_like(depth)

        return DepthResult(
            depth_map=normalised,
            depth_raw=depth,
            min_depth=d_min,
            max_depth=d_max,
        )
