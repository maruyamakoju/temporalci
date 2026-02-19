#!/usr/bin/env python3
"""Extract frames from a video file at a fixed FPS.

Usage::

    python scripts/extract_frames.py \\
        --input jr23_720p.mp4 \\
        --output frames/left \\
        --fps 1 \\
        --max-frames 100 \\
        --quality 95

Requires ``opencv-python-headless`` (or ``opencv-python``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Output directory for frames")
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Frames per second to extract (default: 1)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Maximum frames to extract (0=unlimited)"
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    return parser.parse_args(argv)


def extract_frames(
    input_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = 0,
    quality: int = 95,
) -> int:
    """Extract frames and return the number of frames written."""
    try:
        import cv2
    except ImportError:
        print(
            "ERROR: opencv-python-headless is required. pip install opencv-python-headless",
            file=sys.stderr,
        )
        sys.exit(1)

    video = Path(input_path)
    if not video.is_file():
        print(f"ERROR: input file not found: {video}", file=sys.stderr)
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    written = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            name = f"frame_{written:05d}.jpg"
            cv2.imwrite(str(out / name), frame, encode_params)
            written += 1
            if max_frames > 0 and written >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"Extracted {written} frames to {out}")
    return written


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    extract_frames(
        input_path=args.input,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
