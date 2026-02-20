"""Full 3-layer vision pipeline using ONNX Runtime — no PyTorch required.

Drop-in replacement for the PyTorch-based pipeline.  Designed for edge
deployment on railway vehicles (CPU-only, ARM, or GPU inference).

Usage::

    from temporalci.vision.onnx_pipeline import OnnxPipeline

    pipeline = OnnxPipeline(
        seg_model="models/onnx/segformer_b0_ade20k.int8.onnx",
        depth_model="models/onnx/depth_anything_v2_small.int8.onnx",
    )
    result = pipeline.analyze_frame("frame.jpg")
    result = pipeline.analyze_video("video.mp4", fps=1.0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from temporalci.vision.clearance import (
    ClearanceResult,
    WireDetection,
    calculate_clearance,
    detect_wires,
)
from temporalci.vision.depth import DepthResult
from temporalci.vision.onnx_inference import OnnxDepthModel, OnnxSegmentationModel
from temporalci.vision.segmentation import SegmentationResult


@dataclass
class FrameAnalysis:
    """Complete analysis result for a single frame."""

    frame_id: str
    frame_path: str
    segmentation: SegmentationResult
    depth: DepthResult | None
    wires: WireDetection
    clearance: ClearanceResult
    risk_level: str
    risk_score: float
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_path": self.frame_path,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 4),
            "vegetation_ratio": round(self.segmentation.vegetation_ratio, 4),
            "vegetation_upper_ratio": round(self.segmentation.vegetation_upper_ratio, 4),
            "clearance_px": round(self.clearance.min_clearance_px, 2),
            "vegetation_penetration": round(self.clearance.vegetation_penetration, 4),
            "wire_count": self.wires.wire_count,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "metadata": self.metadata,
        }


@dataclass
class PipelineStats:
    """Aggregate pipeline performance statistics."""

    total_frames: int = 0
    total_elapsed_ms: float = 0.0
    seg_elapsed_ms: float = 0.0
    depth_elapsed_ms: float = 0.0
    clearance_elapsed_ms: float = 0.0
    risk_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def avg_ms_per_frame(self) -> float:
        return self.total_elapsed_ms / max(self.total_frames, 1)

    @property
    def fps(self) -> float:
        avg = self.avg_ms_per_frame
        return 1000.0 / avg if avg > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "avg_ms_per_frame": round(self.avg_ms_per_frame, 1),
            "fps": round(self.fps, 2),
            "seg_elapsed_ms": round(self.seg_elapsed_ms, 1),
            "depth_elapsed_ms": round(self.depth_elapsed_ms, 1),
            "clearance_elapsed_ms": round(self.clearance_elapsed_ms, 1),
            "risk_distribution": self.risk_distribution,
        }


class OnnxPipeline:
    """Full 3-layer vision pipeline using ONNX Runtime."""

    def __init__(
        self,
        seg_model: str | Path,
        depth_model: str | Path | None = None,
        *,
        device: str = "auto",
    ) -> None:
        self._seg = OnnxSegmentationModel(seg_model, device=device)
        self._depth: OnnxDepthModel | None = None
        if depth_model is not None:
            self._depth = OnnxDepthModel(depth_model, device=device)
        self._stats = PipelineStats()

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    def analyze_frame(
        self,
        image_or_path: str | Path | Image.Image,
        *,
        frame_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FrameAnalysis:
        """Analyze a single frame through the full pipeline."""
        t0 = time.perf_counter()

        # Resolve image
        if isinstance(image_or_path, (str, Path)):
            frame_path = str(image_or_path)
            image = Image.open(frame_path).convert("RGB")
        else:
            frame_path = ""
            image = image_or_path.convert("RGB")

        if not frame_id:
            frame_id = Path(frame_path).stem if frame_path else "frame"

        # Layer 1: Segmentation
        t_seg = time.perf_counter()
        seg_result = self._seg.segment(image)
        seg_ms = (time.perf_counter() - t_seg) * 1000

        # Layer 2: Depth
        depth_result = None
        depth_ms = 0.0
        if self._depth is not None:
            t_depth = time.perf_counter()
            depth_result = self._depth.estimate(image)
            depth_ms = (time.perf_counter() - t_depth) * 1000

        # Wire detection + clearance
        t_cl = time.perf_counter()
        wires = detect_wires(image, seg=seg_result)
        clearance = calculate_clearance(seg_result, depth_result, wires)
        cl_ms = (time.perf_counter() - t_cl) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self._stats.total_frames += 1
        self._stats.total_elapsed_ms += total_ms
        self._stats.seg_elapsed_ms += seg_ms
        self._stats.depth_elapsed_ms += depth_ms
        self._stats.clearance_elapsed_ms += cl_ms
        level = clearance.risk_level
        self._stats.risk_distribution[level] = self._stats.risk_distribution.get(level, 0) + 1

        return FrameAnalysis(
            frame_id=frame_id,
            frame_path=frame_path,
            segmentation=seg_result,
            depth=depth_result,
            wires=wires,
            clearance=clearance,
            risk_level=level,
            risk_score=clearance.risk_score,
            elapsed_ms=total_ms,
            metadata=metadata or {},
        )

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        fps: float = 1.0,
        max_frames: int | None = None,
        progress_callback: Any | None = None,
    ) -> list[FrameAnalysis]:
        """Analyze a video file frame-by-frame."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))

        results: list[FrameAnalysis] = []
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)

                analysis = self.analyze_frame(
                    image,
                    frame_id=f"frame_{extracted:05d}",
                    metadata={
                        "video_path": str(video_path),
                        "video_frame_idx": frame_idx,
                        "video_timestamp_sec": round(frame_idx / video_fps, 2),
                    },
                )
                results.append(analysis)
                extracted += 1

                if progress_callback:
                    progress_callback(extracted, total_count // frame_interval)

                if max_frames and extracted >= max_frames:
                    break
            frame_idx += 1

        cap.release()
        return results
