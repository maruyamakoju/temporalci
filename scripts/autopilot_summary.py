from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from datetime import timezone
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.autopilot_utils import atomic_write_json
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import utc_now_iso


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _seconds_between(start_raw: Any, end_raw: Any) -> float | None:
    start = _parse_iso(start_raw)
    end = _parse_iso(end_raw)
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    if delta < 0:
        return None
    return float(delta)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    if left == right:
        return ordered[left]
    frac = pos - left
    return ordered[left] + (ordered[right] - ordered[left]) * frac


def _series_summary(values: list[tuple[datetime, float]]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "mean": None,
            "slope_per_hour": None,
        }

    ordered = sorted(values, key=lambda row: row[0])
    data = [row[1] for row in ordered]
    first_time, first_value = ordered[0]
    last_time, last_value = ordered[-1]
    span_sec = max(0.0, (last_time - first_time).total_seconds())
    slope_per_hour: float | None
    if span_sec > 0:
        slope_per_hour = (last_value - first_value) / (span_sec / 3600.0)
    else:
        slope_per_hour = 0.0
    return {
        "count": len(data),
        "first": first_value,
        "last": last_value,
        "min": min(data),
        "max": max(data),
        "mean": mean(data),
        "slope_per_hour": slope_per_hour,
    }


def _build_summary(
    *,
    artifacts_dir: Path,
    runs_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    status_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    cycle_end_rows = [row for row in runs_rows if str(row.get("event", "")) == "cycle_end"]
    cycle_status_counts = Counter(str(row.get("status", "unknown")) for row in cycle_end_rows)
    cycle_durations = [
        duration
        for duration in (
            _seconds_between(row.get("started_at_utc"), row.get("finished_at_utc"))
            for row in cycle_end_rows
        )
        if duration is not None
    ]

    telemetry_stop_counts = Counter(
        str(row.get("telemetry_stop_reason", ""))
        for row in telemetry_rows
        if row.get("telemetry_stop_reason")
    )
    rss_series: list[tuple[datetime, float]] = []
    gpu_mem_series: list[tuple[datetime, float]] = []
    gpu_util_series: list[tuple[datetime, float]] = []
    for row in telemetry_rows:
        timestamp = _parse_iso(row.get("timestamp_utc"))
        if timestamp is None:
            continue
        rss_value = row.get("rss_bytes")
        gpu_mem_value = row.get("gpu_mem_mb")
        gpu_util_value = row.get("gpu_util")
        if isinstance(rss_value, (int, float)):
            rss_series.append((timestamp, float(rss_value)))
        if isinstance(gpu_mem_value, (int, float)):
            gpu_mem_series.append((timestamp, float(gpu_mem_value)))
        if isinstance(gpu_util_value, (int, float)):
            gpu_util_series.append((timestamp, float(gpu_util_value)))

    telemetry_times = [
        _parse_iso(row.get("timestamp_utc"))
        for row in telemetry_rows
        if _parse_iso(row.get("timestamp_utc")) is not None
    ]
    telemetry_start = min(telemetry_times) if telemetry_times else None
    telemetry_end = max(telemetry_times) if telemetry_times else None
    telemetry_span_sec = (
        max(0.0, (telemetry_end - telemetry_start).total_seconds())
        if telemetry_start is not None and telemetry_end is not None
        else None
    )

    return {
        "generated_at_utc": utc_now_iso(),
        "artifacts_dir": str(artifacts_dir),
        "status_snapshot": status_payload or {},
        "runs": {
            "events_total": len(runs_rows),
            "cycle_end_total": len(cycle_end_rows),
            "cycle_status_counts": dict(cycle_status_counts),
            "cycle_duration_sec": {
                "count": len(cycle_durations),
                "mean": mean(cycle_durations) if cycle_durations else None,
                "p50": _quantile(cycle_durations, 0.5),
                "p95": _quantile(cycle_durations, 0.95),
                "max": max(cycle_durations) if cycle_durations else None,
            },
        },
        "telemetry": {
            "samples_total": len(telemetry_rows),
            "start_utc": telemetry_start.isoformat() if telemetry_start else None,
            "end_utc": telemetry_end.isoformat() if telemetry_end else None,
            "span_sec": telemetry_span_sec,
            "stop_reason_counts": dict(telemetry_stop_counts),
            "rss_bytes": _series_summary(rss_series),
            "gpu_mem_mb": _series_summary(gpu_mem_series),
            "gpu_util": _series_summary(gpu_util_series),
        },
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    status = summary.get("status_snapshot", {})
    runs = summary.get("runs", {})
    telemetry = summary.get("telemetry", {})
    cycle_duration = runs.get("cycle_duration_sec", {})

    lines = [
        "# Autopilot Summary",
        "",
        f"- generated_at_utc: {summary.get('generated_at_utc')}",
        f"- artifacts_dir: {summary.get('artifacts_dir')}",
        f"- state: {status.get('state')}",
        f"- cycle: {status.get('cycle')}",
        "",
        "## Runs",
        "",
        f"- events_total: {runs.get('events_total')}",
        f"- cycle_end_total: {runs.get('cycle_end_total')}",
        f"- cycle_status_counts: {runs.get('cycle_status_counts')}",
        f"- cycle_duration_mean_sec: {cycle_duration.get('mean')}",
        f"- cycle_duration_p50_sec: {cycle_duration.get('p50')}",
        f"- cycle_duration_p95_sec: {cycle_duration.get('p95')}",
        f"- cycle_duration_max_sec: {cycle_duration.get('max')}",
        "",
        "## Telemetry",
        "",
        f"- samples_total: {telemetry.get('samples_total')}",
        f"- start_utc: {telemetry.get('start_utc')}",
        f"- end_utc: {telemetry.get('end_utc')}",
        f"- span_sec: {telemetry.get('span_sec')}",
        f"- stop_reason_counts: {telemetry.get('stop_reason_counts')}",
        f"- rss_bytes: {telemetry.get('rss_bytes')}",
        f"- gpu_mem_mb: {telemetry.get('gpu_mem_mb')}",
        f"- gpu_util: {telemetry.get('gpu_util')}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a summary from autopilot logs.")
    parser.add_argument("--artifacts-dir", default="artifacts/autopilot-96h")
    parser.add_argument("--runs-file", default="autopilot_runs.jsonl")
    parser.add_argument("--telemetry-file", default="autopilot_telemetry.jsonl")
    parser.add_argument("--status-file", default="autopilot_status.json")
    parser.add_argument("--output-json", default="autopilot_summary.json")
    parser.add_argument("--output-md", default="autopilot_summary.md")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    runs_path = Path(args.runs_file)
    if not runs_path.is_absolute():
        runs_path = artifacts_dir / runs_path
    telemetry_path = Path(args.telemetry_file)
    if not telemetry_path.is_absolute():
        telemetry_path = artifacts_dir / telemetry_path
    status_path = Path(args.status_file)
    if not status_path.is_absolute():
        status_path = artifacts_dir / status_path
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = artifacts_dir / output_json
    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = artifacts_dir / output_md

    runs_rows = _read_jsonl(runs_path)
    telemetry_rows = _read_jsonl(telemetry_path)
    status_payload = read_json_dict(status_path) or {}

    summary = _build_summary(
        artifacts_dir=artifacts_dir,
        runs_rows=runs_rows,
        telemetry_rows=telemetry_rows,
        status_payload=status_payload,
    )
    atomic_write_json(output_json, summary)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(f"wrote_summary json={output_json} md={output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
