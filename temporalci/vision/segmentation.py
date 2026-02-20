"""Semantic segmentation using SegFormer (ADE20K).

Classifies each pixel into vegetation, sky, infrastructure, or ground.
Uses ``nvidia/segformer-b0-finetuned-ade-512-512`` (≈14 MB) for fast
inference on CPU or GPU.

ADE20K provides 150 fine-grained classes.  We group them into four
macro-categories relevant to catenary inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# ADE20K class groupings (0-indexed)
# Full list: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
# ---------------------------------------------------------------------------

# Vegetation: tree, grass, plant, flower, palm, bush/shrub, etc.
VEGETATION_IDS: frozenset[int] = frozenset(
    {
        4,  # tree
        9,  # grass
        17,  # plant / flora
        66,  # flower
        72,  # palm
        73,  # bush (some ADE20K variants)
    }
)

# Sky
SKY_IDS: frozenset[int] = frozenset({2})

# Infrastructure: building, wall, fence, pole, railing, signboard, etc.
INFRASTRUCTURE_IDS: frozenset[int] = frozenset(
    {
        0,  # wall
        1,  # building
        14,  # door
        25,  # cabinet / box
        32,  # fence
        43,  # signboard
        64,  # railing
        84,  # bridge
        87,  # streetlight
        93,  # pole
        136,  # pylon
    }
)

# Ground / road surface
GROUND_IDS: frozenset[int] = frozenset(
    {
        3,  # floor
        6,  # road / route
        11,  # sidewalk
        13,  # earth / ground
        29,  # path
        46,  # sand
        52,  # platform
        91,  # dirt track
    }
)

_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


@dataclass
class SegmentationResult:
    """Per-frame segmentation output."""

    seg_map: np.ndarray  # (H, W) int — raw class IDs
    vegetation_mask: np.ndarray  # (H, W) bool
    sky_mask: np.ndarray  # (H, W) bool
    infrastructure_mask: np.ndarray  # (H, W) bool
    ground_mask: np.ndarray  # (H, W) bool
    vegetation_ratio: float  # 0-1 over whole frame
    vegetation_upper_ratio: float  # 0-1 in catenary zone (upper 40%)
    classes_found: set[int] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vegetation_ratio": round(self.vegetation_ratio, 6),
            "vegetation_upper_ratio": round(self.vegetation_upper_ratio, 6),
            "classes_found": sorted(self.classes_found),
        }


class SegmentationModel:
    """Thin wrapper around HuggingFace SegFormer."""

    def __init__(self, *, device: str = "auto") -> None:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._processor = SegformerImageProcessor.from_pretrained(_MODEL_NAME)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            _MODEL_NAME,
        ).to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def segment(self, image_or_path: str | Path | Image.Image) -> SegmentationResult:
        """Run segmentation on a single image."""
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        w, h = image.size
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        logits = self._model(**inputs).logits  # (1, 150, H/4, W/4)

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        veg_mask = np.isin(seg_map, list(VEGETATION_IDS))
        sky_mask = np.isin(seg_map, list(SKY_IDS))
        infra_mask = np.isin(seg_map, list(INFRASTRUCTURE_IDS))
        ground_mask = np.isin(seg_map, list(GROUND_IDS))

        # Catenary zone = upper 40 % of frame
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
