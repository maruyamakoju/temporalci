"""Cross-run trend report generator.

Reads run history from a model artifact directory and generates an HTML
trend report showing quality metrics over time.
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def _load_run_json(run_dir: Path) -> dict[str, Any] | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def load_model_runs(model_root: Path, *, last_n: int = 30) -> list[dict[str, Any]]:
    """Return the last *last_n* run payloads from *model_root*.

    Falls back to the minimal runs.jsonl entry when run.json is missing.
    """
    runs_jsonl = model_root / "runs.jsonl"
    if not runs_jsonl.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        for raw in runs_jsonl.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []

    entries = entries[-last_n:]

    runs: list[dict[str, Any]] = []
    for entry in entries:
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            continue
        payload = _load_run_json(model_root / run_id)
        if payload is not None:
            runs.append(payload)
        else:
            # Minimal stub from runs.jsonl
            runs.append({
                "run_id": run_id,
                "status": entry.get("status", "UNKNOWN"),
                "timestamp_utc": entry.get("timestamp_utc", ""),
                "sample_count": entry.get("sample_count", 0),
                "metrics": {},
                "gates": [],
            })
    return runs


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _resolve_dotted(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _extract_metric_series(
    runs: list[dict[str, Any]],
    metric_path: str,
) -> list[float | None]:
    values: list[float | None] = []
    for run in runs:
        metrics = run.get("metrics") or {}
        try:
            val = _resolve_dotted(metrics, metric_path)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                values.append(float(val))
            else:
                values.append(None)
        except (KeyError, TypeError, ValueError):
            values.append(None)
    return values


def _discover_metric_paths(runs: list[dict[str, Any]]) -> list[str]:
    """Return a de-duped list of useful scalar metric paths from any run."""
    paths: list[str] = []
    seen: set[str] = set()

    def _walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
            elif isinstance(v, dict):
                _walk(v, path)

    for run in runs:
        _walk(run.get("metrics") or {}, "")
    return paths


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_PASS_COLOR = "#15803d"
_FAIL_COLOR = "#b91c1c"
_LINE_COLOR = "#2563eb"
_GRID_COLOR = "#e2e8f0"
_AXIS_COLOR = "#94a3b8"
_LABEL_COLOR = "#64748b"


def _svg_trend_chart(
    *,
    values: list[float | None],
    statuses: list[str],
    metric_path: str,
    width: int = 700,
    height: int = 160,
) -> str:
    finite_vals = [v for v in values if v is not None]
    if not finite_vals:
        return f'<p style="color:#94a3b8;font-size:0.82rem">No data for {escape(metric_path)}</p>'

    PAD_L, PAD_R, PAD_T, PAD_B = 54, 16, 16, 38
    pw = width - PAD_L - PAD_R
    ph = height - PAD_T - PAD_B
    n = len(values)

    lo = min(finite_vals)
    hi = max(finite_vals)
    margin = (hi - lo) * 0.12 or 0.05
    y_min = lo - margin
    y_max = hi + margin
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_span = y_max - y_min

    def px(i: int) -> float:
        return PAD_L + (i / max(n - 1, 1)) * pw

    def py(val: float) -> float:
        return PAD_T + ph - ((val - y_min) / y_span) * ph

    parts: list[str] = []

    # background
    parts.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" '
        f'fill="#f8fafc" stroke="{_GRID_COLOR}" stroke-width="1" rx="4"/>'
    )

    # grid + y labels
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * y_span
        yp = py(yv)
        parts.append(
            f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{PAD_L + pw}" y2="{yp:.1f}" '
            f'stroke="{_GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 4}" y="{yp:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="10" fill="{_AXIS_COLOR}">{yv:.4f}</text>'
        )

    # line segments (skip None gaps)
    seg_start: int | None = None
    for i, v in enumerate(values):
        if v is not None:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None and i - seg_start >= 2:
                seg_vals = [(j, values[j]) for j in range(seg_start, i) if values[j] is not None]
                pts = " ".join(
                    f"{px(j):.1f},{py(float(val)):.1f}"
                    for j, val in seg_vals if val is not None
                )
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{_LINE_COLOR}" '
                    f'stroke-width="2" stroke-linejoin="round"/>'
                )
            seg_start = None
    if seg_start is not None:
        seg_vals = [(j, values[j]) for j in range(seg_start, n) if values[j] is not None]
        if len(seg_vals) >= 2:
            pts = " ".join(
                f"{px(j):.1f},{py(float(val)):.1f}"
                for j, val in seg_vals if val is not None
            )
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{_LINE_COLOR}" '
                f'stroke-width="2" stroke-linejoin="round"/>'
            )

    # data points (colored by status)
    for i, v in enumerate(values):
        if v is None:
            continue
        status = statuses[i] if i < len(statuses) else "UNKNOWN"
        dot_color = _PASS_COLOR if status == "PASS" else _FAIL_COLOR
        parts.append(
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" '
            f'fill="{dot_color}" stroke="#fff" stroke-width="1.5">'
            f'<title>{escape(status)} run #{i + 1}: {v:.6f}</title>'
            f"</circle>"
        )

    # axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ph}" '
        f'stroke="{_AXIS_COLOR}" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + ph}" x2="{PAD_L + pw}" y2="{PAD_T + ph}" '
        f'stroke="{_AXIS_COLOR}" stroke-width="1"/>'
    )

    # x ticks
    tick_count = min(n, 10)
    for t in range(tick_count + 1):
        i = round(t * (n - 1) / max(tick_count, 1))
        xp = px(i)
        parts.append(
            f'<line x1="{xp:.1f}" y1="{PAD_T + ph}" x2="{xp:.1f}" y2="{PAD_T + ph + 4}" '
            f'stroke="{_AXIS_COLOR}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xp:.1f}" y="{PAD_T + ph + 16}" text-anchor="middle" '
            f'font-size="10" fill="{_AXIS_COLOR}">#{i + 1}</text>'
        )

    # axis labels
    parts.append(
        f'<text x="{PAD_L + pw / 2:.0f}" y="{height - 2}" '
        f'text-anchor="middle" font-size="10" fill="{_LABEL_COLOR}">run index</text>'
    )

    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block">\n  {inner}\n</svg>'
    )


# ---------------------------------------------------------------------------
# Pass/fail history strip
# ---------------------------------------------------------------------------

def _svg_status_strip(statuses: list[str], *, width: int = 700, height: int = 28) -> str:
    if not statuses:
        return ""
    n = len(statuses)
    cell_w = width / n
    parts: list[str] = []
    for i, s in enumerate(statuses):
        color = _PASS_COLOR if s == "PASS" else _FAIL_COLOR
        x = i * cell_w
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{cell_w:.1f}" height="{height}" fill="{color}">'
            f'<title>Run #{i + 1}: {escape(s)}</title>'
            f"</rect>"
        )
    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block;border-radius:4px">\n  {inner}\n</svg>'
    )


# ---------------------------------------------------------------------------
# HTML report writer
# ---------------------------------------------------------------------------

_TREND_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
  }
  h1 { margin: 0 0 8px; font-size: 1.4rem; }
  h2 { font-size: 1.05rem; font-weight: 600; margin: 24px 0 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
  h3 { font-size: 0.92rem; font-weight: 600; margin: 0 0 8px; color: #475569; }
  section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 14px; }
  .subtitle { color: #64748b; font-size: 0.88rem; margin: 0 0 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { background: #f8fafc; font-weight: 600; text-align: left; padding: 7px 10px; border-bottom: 2px solid #e2e8f0; }
  td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  .tag-pass { display:inline-block; padding:1px 7px; border-radius:4px; font-weight:600; font-size:0.75rem; background:#dcfce7; color:#15803d; }
  .tag-fail { display:inline-block; padding:1px 7px; border-radius:4px; font-weight:600; font-size:0.75rem; background:#fee2e2; color:#b91c1c; }
  .chart-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .legend { display:flex; gap:12px; font-size:0.78rem; color:#64748b; margin-top:6px; }
  .legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:3px; }
  code { font-family: "SF Mono","Fira Code",monospace; font-size:0.85em; background:#f1f5f9; padding:1px 4px; border-radius:3px; }
"""


def write_trend_report(
    path: Path,
    runs: list[dict[str, Any]],
    *,
    title: str = "TemporalCI Trend Report",
) -> None:
    """Write an HTML trend report for *runs* to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not runs:
        path.write_text(
            "<!doctype html><html><body><p>No runs found.</p></body></html>",
            encoding="utf-8",
        )
        return

    statuses = [str(r.get("status", "UNKNOWN")) for r in runs]
    n_pass = statuses.count("PASS")
    n_fail = len(statuses) - n_pass

    # Metadata
    first_ts = str(runs[0].get("timestamp_utc", ""))
    last_ts = str(runs[-1].get("timestamp_utc", ""))
    project = str(runs[-1].get("project", ""))
    suite_name = str(runs[-1].get("suite_name", ""))
    model_name = str(runs[-1].get("model_name", ""))

    subtitle = (
        f"{escape(project)} / {escape(suite_name)} / {escape(model_name)}"
        f" &nbsp;·&nbsp; {len(runs)} runs"
        f" &nbsp;·&nbsp; {first_ts[:10]} → {last_ts[:10]}"
    )

    # Status strip
    strip_html = _svg_status_strip(statuses)

    # Run history table
    table_rows: list[str] = []
    for i, run in enumerate(runs):
        s = statuses[i]
        tag = f'<span class="tag-{"pass" if s == "PASS" else "fail"}">{escape(s)}</span>'
        run_id = str(run.get("run_id", ""))
        ts = str(run.get("timestamp_utc", ""))[:19]
        table_rows.append(
            f"<tr>"
            f"<td>#{i + 1}</td>"
            f"<td><code>{escape(run_id)}</code></td>"
            f"<td>{escape(ts)}</td>"
            f"<td>{tag}</td>"
            f"<td>{escape(str(run.get('sample_count', '')))}</td>"
            f"</tr>"
        )
    history_table = (
        "<table><thead><tr><th>#</th><th>Run ID</th><th>Timestamp</th><th>Status</th><th>Samples</th></tr></thead>"
        f'<tbody>{"".join(table_rows)}</tbody></table>'
    )

    # Metric charts
    metric_paths = _discover_metric_paths(runs)
    chart_sections: list[str] = []
    for mp in metric_paths:
        values = _extract_metric_series(runs, mp)
        finite = [v for v in values if v is not None]
        if len(finite) < 2:
            continue
        chart_svg = _svg_trend_chart(
            values=values,
            statuses=statuses,
            metric_path=mp,
        )
        last_val = finite[-1]
        first_val = finite[0]
        delta = last_val - first_val
        sign = "+" if delta >= 0 else ""
        delta_color = "#15803d" if delta >= 0 else "#b91c1c"
        chart_sections.append(
            f'<div class="chart-card">'
            f'<h3><code>{escape(mp)}</code>'
            f' &nbsp;<span style="font-size:0.82rem;color:{delta_color};font-weight:600">{sign}{delta:.4f}</span>'
            f' <span style="font-size:0.78rem;color:#94a3b8;font-weight:400">vs first run</span></h3>'
            f"{chart_svg}"
            f'<div class="legend">'
            f'<span><span class="legend-dot" style="background:#15803d"></span>PASS run</span>'
            f'<span><span class="legend-dot" style="background:#b91c1c"></span>FAIL run</span>'
            f"</div>"
            f"</div>"
        )

    charts_html = "".join(chart_sections) if chart_sections else '<p style="color:#94a3b8">No scalar metrics found.</p>'

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>{_TREND_CSS}</style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="subtitle">{subtitle}</p>

  <section>
    <h2>Pass / Fail History</h2>
    <p style="font-size:0.82rem;color:#475569;margin:0 0 8px">
      {n_pass} passed &nbsp;·&nbsp; {n_fail} failed &nbsp;·&nbsp; {len(runs)} total
    </p>
    {strip_html}
  </section>

  <section>
    <h2>Metric Trends</h2>
    {charts_html}
  </section>

  <section>
    <h2>Run History</h2>
    {history_table}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
