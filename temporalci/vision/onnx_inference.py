"""ONNX Runtime inference wrappers for segmentation and depth models.

Drop-in replacements for the PyTorch-based models.  Use these when
deploying without PyTorch (edge devices, CI runners, etc.).

Usage::

    from temporalci.vision.onnx_inference import OnnxSegmentationModel, OnnxDepthModel

    seg = OnnxSegmentationModel("models/onnx/segformer_b0_ade20k.int8.onnx")
    depth = OnnxDepthModel("models/onnx/depth_anything_v2_small.int8.onnx")

    seg_result = seg.segment("frame.jpg")    # same API as SegmentationModel
    depth_result = depth.estimate("frame.jpg")  # same API as DepthModel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from temporalci.vision.depth import DepthResult
from temporalci.vision.segmentation import (
    GROUND_IDS,
    INFRASTRUCTURE_IDS,
    SKY_IDS,
    VEGETATION_IDS,
    SegmentationResult,
)


def _select_providers(device: str) -> list[str]:
    """Select ONNX Runtime execution providers."""
    available = ort.get_available_providers()
    if device == "auto":
        # Prefer CUDA > TensorRT > CPU
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxSegmentationModel:
    """ONNX Runtime wrapper for SegFormer segmentation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        input_size: tuple[int, int] = (512, 512),
    ) -> None:
        providers = _select_providers(device)
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size  # (H, W)

        # SegFormer ImageNet normalisation
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self._input_size[1], self._input_size[0]), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        arr = (arr - self._mean) / self._std
        return arr

    def segment(self, image_or_path: str | Path | Image.Image) -> SegmentationResult:
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        w, h = image.size
        inputs = self._preprocess(image)
        (logits,) = self._session.run(None, {self._input_name: inputs})

        # Upsample logits to original size
        # logits shape: (1, 150, H/4, W/4)
        from scipy.ndimage import zoom

        _, n_classes, lh, lw = logits.shape
        scale_h = h / lh
        scale_w = w / lw
        upsampled = zoom(logits[0], (1, scale_h, scale_w), order=1)  # (150, H, W)
        seg_map = upsampled.argmax(axis=0).astype(np.int32)

        veg_mask = np.isin(seg_map, list(VEGETATION_IDS))
        sky_mask = np.isin(seg_map, list(SKY_IDS))
        infra_mask = np.isin(seg_map, list(INFRASTRUCTURE_IDS))
        ground_mask = np.isin(seg_map, list(GROUND_IDS))

        zone_h = int(h * 0.4)
        veg_upper = veg_mask[:zone_h, :]

        return SegmentationResult(
            seg_map=seg_map,
            vegetation_mask=veg_mask,
            sky_mask=sky_mask,
            infrastructure_mask=infra_mask,
            ground_mask=ground_mask,
            vegetation_ratio=float(veg_mask.mean()),
            vegetation_upper_ratio=float(veg_upper.mean()) if zone_h > 0 else 0.0,
            classes_found=set(np.unique(seg_map).tolist()),
        )


class OnnxDepthModel:
    """ONNX Runtime wrapper for Depth Anything V2 depth estimation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        input_size: tuple[int, int] = (518, 518),
    ) -> None:
        providers = _select_providers(device)
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self._input_size[1], self._input_size[0]), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None]
        arr = (arr - self._mean) / self._std
        return arr

    def estimate(self, image_or_path: str | Path | Image.Image) -> DepthResult:
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        w, h = image.size
        inputs = self._preprocess(image)
        (raw_depth,) = self._session.run(None, {self._input_name: inputs})

        # Upsample to original resolution
        from scipy.ndimage import zoom

        if raw_depth.ndim == 4:
            raw_depth = raw_depth[0, 0]
        elif raw_depth.ndim == 3:
            raw_depth = raw_depth[0]

        scale_h = h / raw_depth.shape[0]
        scale_w = w / raw_depth.shape[1]
        depth = zoom(raw_depth, (scale_h, scale_w), order=1).astype(np.float32)

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
