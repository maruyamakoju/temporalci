"""Video processing pipeline — single command from video to analysis report."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int | None = None,
    quality: int = 95,
) -> list[Path]:
    """Extract frames from video at the specified FPS rate."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))

    extracted: list[Path] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{len(extracted):05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted.append(out_path)
            if max_frames and len(extracted) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return extracted


def process_video(
    video_path: str | Path,
    *,
    output_dir: str | Path = "inspection_output",
    fps: float = 1.0,
    max_frames: int | None = None,
    device: str = "auto",
    skip_depth: bool = False,
    generate_panels: bool = True,
) -> dict[str, Any]:
    """End-to-end pipeline: video -> frames -> analysis -> results.

    Returns the metric evaluation result dict.
    """
    from temporalci.metrics.catenary_clearance import evaluate
    from temporalci.types import GeneratedSample

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    panels_dir = output_dir / "panels" if generate_panels else None

    # Step 1: Extract frames
    print(f"Extracting frames at {fps} fps...")
    t0 = time.time()
    frame_paths = extract_frames(
        video_path,
        frames_dir,
        fps=fps,
        max_frames=max_frames,
    )
    print(f"  Extracted {len(frame_paths)} frames in {time.time() - t0:.1f}s")

    if not frame_paths:
        return {"score": 0.0, "dims": {}, "sample_count": 0, "error": "no frames extracted"}

    # Step 2: Build samples
    samples = [
        GeneratedSample(
            test_id="inspect",
            prompt=f.stem,
            seed=0,
            video_path=str(f),
            evaluation_stream=[],
        )
        for f in frame_paths
    ]

    # Step 3: Run pipeline
    print(f"Running 3-layer vision pipeline on {len(samples)} frames...")
    t1 = time.time()
    params: dict[str, Any] = {
        "device": device,
        "skip_depth": str(skip_depth).lower(),
    }
    if panels_dir:
        params["output_dir"] = str(panels_dir)

    result = evaluate(samples, params=params)
    elapsed = time.time() - t1
    print(f"  Analysis complete in {elapsed:.1f}s ({elapsed / len(samples):.2f}s/frame)")

    # Step 4: Summary
    result["_meta"] = {
        "video_path": str(video_path),
        "fps": fps,
        "frames_extracted": len(frame_paths),
        "output_dir": str(output_dir),
        "elapsed_extract": round(time.time() - t0 - elapsed, 1),
        "elapsed_analysis": round(elapsed, 1),
    }

    return result
