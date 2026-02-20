#!/usr/bin/env python3
"""End-to-end demo: video → full inspection pipeline → all reports.

Processes a video file through the complete catenary inspection pipeline
and generates every output artifact in a single demo_output/ directory.

Usage::

    python scripts/demo_full_pipeline.py --input jr23_720p.mp4
    python scripts/demo_full_pipeline.py --input jr23_720p.mp4 --fps 2 --max-frames 30

Generated outputs (in demo_output/):
    frames/             Extracted JPEG frames
    panels/             4-panel visualisation per frame
    dashboard.html      Interactive dark-themed dashboard
    route_map.html      Leaflet.js GPS route map
    anomaly_report.json Anomaly detection results
    km_report.html      Km-based maintenance report
    run.json            Raw pipeline results
    summary.txt         Human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end catenary inspection demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Path to input video")
    p.add_argument(
        "--output-dir",
        "-o",
        default="demo_output",
        help="Output directory (default: demo_output)",
    )
    p.add_argument("--fps", type=float, default=1.0, help="Extraction FPS (default: 1)")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    p.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation (faster)",
    )
    p.add_argument(
        "--skip-anomaly",
        action="store_true",
        help="Skip anomaly detection pass",
    )
    p.add_argument("--device", default="auto", help="Torch device (auto/cpu/cuda)")
    p.add_argument("--title", default="JR East Catenary Inspection", help="Report title")
    return p.parse_args(argv)


def _simulate_gps(n_frames: int, start_km: float = 12.0) -> list[dict[str, float]]:
    """Generate simulated GPS + km data for demo purposes.

    Returns one dict per frame with lat, lon, km.
    """
    # Simulate a straight route along a JR East line near Tokyo
    base_lat, base_lon = 35.6812, 139.7671
    result: list[dict[str, float]] = []
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        result.append(
            {
                "lat": round(base_lat + frac * 0.015, 6),
                "lon": round(base_lon + frac * 0.008, 6),
                "km": round(start_km + frac * 2.0, 3),
            }
        )
    return result


def run_demo(args: argparse.Namespace) -> int:
    """Execute the full demo pipeline."""
    t_start = time.time()
    video_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not video_path.is_file():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    panels_dir = output_dir / "panels"

    # ------------------------------------------------------------------
    # Step 1: Extract frames
    # ------------------------------------------------------------------
    print(f"[1/6] Extracting frames from {video_path.name} at {args.fps} fps ...")
    from temporalci.vision.video import extract_frames

    frame_paths = extract_frames(
        video_path,
        frames_dir,
        fps=args.fps,
        max_frames=args.max_frames or None,
    )
    n_frames = len(frame_paths)
    print(f"       {n_frames} frames extracted")

    if n_frames == 0:
        print("ERROR: no frames extracted", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 2: Run 3-layer clearance pipeline
    # ------------------------------------------------------------------
    print(f"[2/6] Running 3-layer vision pipeline on {n_frames} frames ...")
    from temporalci.metrics.catenary_clearance import evaluate as clearance_eval
    from temporalci.types import GeneratedSample

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

    params: dict[str, Any] = {
        "device": args.device,
        "skip_depth": str(args.skip_depth).lower(),
        "output_dir": str(panels_dir),
    }

    t_pipeline = time.time()
    clearance_results = clearance_eval(samples, params=params)
    pipeline_elapsed = time.time() - t_pipeline
    print(f"       Done in {pipeline_elapsed:.1f}s ({pipeline_elapsed / n_frames:.2f}s/frame)")

    # ------------------------------------------------------------------
    # Step 3: Anomaly detection (optional)
    # ------------------------------------------------------------------
    anomaly_results: dict[str, Any] | None = None
    if not args.skip_anomaly:
        print(f"[3/6] Running anomaly detection on {n_frames} frames ...")
        from temporalci.metrics.catenary_anomaly import evaluate as anomaly_eval

        t_anom = time.time()
        anomaly_results = anomaly_eval(samples, params=params)
        anom_elapsed = time.time() - t_anom
        print(f"       Done in {anom_elapsed:.1f}s")
    else:
        print("[3/6] Anomaly detection skipped")

    # ------------------------------------------------------------------
    # Step 4: Generate dashboard
    # ------------------------------------------------------------------
    print("[4/6] Generating dashboard ...")
    from temporalci.dashboard import generate_dashboard

    dashboard_path = output_dir / "dashboard.html"
    generate_dashboard(clearance_results, dashboard_path, title=args.title)
    print(f"       {dashboard_path}")

    # ------------------------------------------------------------------
    # Step 5: Generate route map + km report
    # ------------------------------------------------------------------
    print("[5/6] Generating route map and km report ...")
    from temporalci.fusion import aggregate_by_km, generate_km_report
    from temporalci.route_map import generate_route_map

    # Inject simulated GPS data
    gps_data = _simulate_gps(n_frames)
    per_sample = clearance_results.get("per_sample", [])

    route_items: list[dict[str, Any]] = []
    for i, sample in enumerate(per_sample):
        if i < len(gps_data):
            gps = gps_data[i]
            sample["lat"] = gps["lat"]
            sample["lon"] = gps["lon"]
            sample["km"] = gps["km"]
        route_items.append(
            {
                "prompt": sample.get("prompt", ""),
                "risk_level": sample.get("risk_level", "unknown"),
                "risk_score": sample.get("dims", {}).get("risk_score", 0.5),
                "lat": sample.get("lat"),
                "lon": sample.get("lon"),
                "km": sample.get("km"),
                "vegetation_zone": sample.get("dims", {}).get("vegetation_proximity_nn", 0),
                "clearance_px": sample.get("clearance_px", 0),
            }
        )

    route_map_path = output_dir / "route_map.html"
    generate_route_map(route_items, route_map_path, title=f"{args.title} — Route")
    print(f"       {route_map_path}")

    # Km report
    km_bins = aggregate_by_km(route_items, bin_size_km=0.5)
    if km_bins:
        km_report_path = output_dir / "km_report.html"
        generate_km_report(km_bins, km_report_path, title=f"{args.title} — Km Report")
        print(f"       {km_report_path}")

    # ------------------------------------------------------------------
    # Step 6: Save results and summary
    # ------------------------------------------------------------------
    print("[6/6] Saving results ...")
    run_data: dict[str, Any] = {
        "video_path": str(video_path),
        "n_frames": n_frames,
        "fps": args.fps,
        "pipeline_elapsed_s": round(pipeline_elapsed, 1),
        "clearance": clearance_results,
    }
    if anomaly_results:
        run_data["anomaly"] = anomaly_results

    run_json_path = output_dir / "run.json"
    run_json_path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")
    print(f"       {run_json_path}")

    # Anomaly report
    if anomaly_results:
        anomaly_path = output_dir / "anomaly_report.json"
        anomaly_path.write_text(
            json.dumps(anomaly_results, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"       {anomaly_path}")

    total_elapsed = time.time() - t_start

    # Summary
    score = clearance_results.get("score", 0)
    n_alerts = len(clearance_results.get("alert_frames", []))
    dist: dict[str, int] = {}
    for s in per_sample:
        level = s.get("risk_level", "unknown")
        dist[level] = dist.get(level, 0) + 1

    summary_lines = [
        f"{'=' * 60}",
        f"  {args.title}",
        f"{'=' * 60}",
        f"  Video:          {video_path.name}",
        f"  Frames:         {n_frames}",
        f"  FPS:            {args.fps}",
        f"  Pipeline time:  {pipeline_elapsed:.1f}s ({pipeline_elapsed / n_frames:.2f}s/frame)",
        f"  Total time:     {total_elapsed:.1f}s",
        "",
        f"  Composite Score: {score:.4f}",
        f"  Alerts:          {n_alerts}",
        "  Risk distribution:",
    ]
    for level in ["critical", "warning", "caution", "safe"]:
        count = dist.get(level, 0)
        bar = "#" * count
        summary_lines.append(f"    {level:10s} {count:3d}  {bar}")

    if anomaly_results:
        anom_alerts = len(anomaly_results.get("alert_frames", []))
        summary_lines.append(f"  Anomaly alerts:  {anom_alerts}")

    summary_lines.extend(
        [
            "",
            "  Outputs:",
            f"    {output_dir}/dashboard.html",
            f"    {output_dir}/route_map.html",
        ]
    )
    if km_bins:
        summary_lines.append(f"    {output_dir}/km_report.html")
    summary_lines.extend(
        [
            f"    {output_dir}/run.json",
            f"    {output_dir}/panels/  ({n_frames} panels)",
            f"{'=' * 60}",
        ]
    )

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
