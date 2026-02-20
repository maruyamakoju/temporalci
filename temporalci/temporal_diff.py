"""Temporal comparison for catenary inspection runs.

Compares two inspection results (before / after) frame-by-frame to detect
vegetation growth or recession at each location along the rail line.

Public API
----------
temporal_diff(before, after) -> dict          (pure data)
write_temporal_diff_report(path, before, after) -> dict  (HTML report)
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def _match_frames(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match frames by prompt (frame identifier / location)."""
    after_map = {str(s.get("prompt", "")): s for s in after}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for b in before:
        prompt = str(b.get("prompt", ""))
        a = after_map.get(prompt)
        if a is not None:
            pairs.append((b, a))
    return pairs


def temporal_diff(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-frame temporal diff between two inspection runs.

    Both *before* and *after* should be metric evaluation results with
    ``per_sample`` entries containing ``dims`` (risk_score, vegetation_proximity_nn,
    vegetation_penetration, etc.).

    Returns a dict with:
    - ``matched_count``: number of paired frames
    - ``frames``: per-frame deltas sorted by worst degradation
    - ``summary``: aggregate statistics
    - ``hotspots``: frames where vegetation risk significantly increased
    """
    b_samples = before.get("per_sample") or []
    a_samples = after.get("per_sample") or []

    pairs = _match_frames(b_samples, a_samples)

    frames: list[dict[str, Any]] = []
    dim_deltas: dict[str, list[float]] = {}

    for b, a in pairs:
        b_dims = b.get("dims") or {}
        a_dims = a.get("dims") or {}

        all_dims = set(b_dims.keys()) | set(a_dims.keys())
        deltas: dict[str, float | None] = {}
        for dim in all_dims:
            bv = b_dims.get(dim)
            av = a_dims.get(dim)
            if (
                isinstance(bv, (int, float))
                and isinstance(av, (int, float))
                and not isinstance(bv, bool)
                and not isinstance(av, bool)
                and math.isfinite(float(bv))
                and math.isfinite(float(av))
            ):
                d = float(av) - float(bv)
                deltas[dim] = round(d, 6)
                dim_deltas.setdefault(dim, []).append(d)
            else:
                deltas[dim] = None

        # risk_score: higher = safer, so negative delta = worse
        b_risk = float(b_dims.get("risk_score", 0.5))
        a_risk = float(a_dims.get("risk_score", 0.5))
        risk_delta = a_risk - b_risk

        frames.append(
            {
                "prompt": str(b.get("prompt", "")),
                "before_risk": round(b_risk, 4),
                "after_risk": round(a_risk, 4),
                "risk_delta": round(risk_delta, 4),
                "before_level": b.get("risk_level", "unknown"),
                "after_level": a.get("risk_level", "unknown"),
                "dim_deltas": deltas,
            }
        )

    # Sort by risk degradation (most negative delta first = worst degradation)
    frames.sort(key=lambda f: f["risk_delta"])

    # Hotspots: frames where risk_score dropped by > 0.1
    hotspots = [f for f in frames if f["risk_delta"] < -0.1]

    # Summary
    risk_deltas = [f["risk_delta"] for f in frames]
    summary: dict[str, Any] = {
        "matched_count": len(pairs),
        "before_total": len(b_samples),
        "after_total": len(a_samples),
        "unmatched_before": len(b_samples) - len(pairs),
        "unmatched_after": len(a_samples) - len(pairs),
    }

    if risk_deltas:
        summary["avg_risk_delta"] = round(sum(risk_deltas) / len(risk_deltas), 4)
        summary["worst_risk_delta"] = round(min(risk_deltas), 4)
        summary["best_risk_delta"] = round(max(risk_deltas), 4)
        summary["degraded_count"] = sum(1 for d in risk_deltas if d < -0.05)
        summary["improved_count"] = sum(1 for d in risk_deltas if d > 0.05)
        summary["stable_count"] = sum(1 for d in risk_deltas if abs(d) <= 0.05)
    else:
        summary["avg_risk_delta"] = 0.0
        summary["degraded_count"] = 0
        summary["improved_count"] = 0
        summary["stable_count"] = 0

    # Per-dimension aggregate deltas
    dim_summary: dict[str, dict[str, float]] = {}
    for dim, values in dim_deltas.items():
        dim_summary[dim] = {
            "mean": round(sum(values) / len(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    summary["dim_summary"] = dim_summary

    return {
        "matched_count": len(pairs),
        "frames": frames,
        "hotspots": hotspots,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_DIFF_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
  }
  h1 { margin: 0 0 8px; font-size: 1.4rem; }
  h2 { font-size: 1.05rem; font-weight: 600; margin: 24px 0 8px;
       border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
  section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 20px; margin-bottom: 14px; }
  .subtitle { color: #64748b; font-size: 0.88rem; margin: 0 0 16px; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .stat-card {
    border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;
    background: #f8fafc; text-align: center;
  }
  .stat-value { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
  .stat-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
  .degraded { color: #b91c1c; }
  .improved { color: #15803d; }
  .stable { color: #64748b; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th { background: #f8fafc; font-weight: 600; text-align: left;
       padding: 7px 10px; border-bottom: 2px solid #e2e8f0; }
  td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafafa; }
  .bar-container { width: 100px; height: 16px; background: #f1f5f9;
                   border-radius: 3px; overflow: hidden; display: inline-block;
                   vertical-align: middle; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-weight: 600; font-size: 0.76rem; }
  .tag-critical { background: #fee2e2; color: #b91c1c; }
  .tag-warning  { background: #fff7ed; color: #c2410c; }
  .tag-caution  { background: #fef9c3; color: #92400e; }
  .tag-safe     { background: #dcfce7; color: #15803d; }
  .tag-unknown  { background: #f1f5f9; color: #94a3b8; }
  .delta-neg { color: #b91c1c; font-weight: 600; }
  .delta-pos { color: #15803d; font-weight: 600; }
  .delta-zero { color: #94a3b8; }
  code { font-family: "SF Mono","Fira Code",monospace; font-size: 0.85em;
         background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
  .hotspot-row td { background: #fef2f2 !important; }
"""


def _risk_tag(level: str) -> str:
    cls = {
        "critical": "tag-critical",
        "warning": "tag-warning",
        "caution": "tag-caution",
        "safe": "tag-safe",
    }.get(level, "tag-unknown")
    return f'<span class="tag {cls}">{escape(level.upper())}</span>'


def _delta_html(delta: float) -> str:
    if delta < -0.01:
        return f'<span class="delta-neg">{delta:+.4f}</span>'
    if delta > 0.01:
        return f'<span class="delta-pos">{delta:+.4f}</span>'
    return f'<span class="delta-zero">{delta:+.4f}</span>'


def _risk_bar(value: float, delta: float) -> str:
    pct = max(0, min(100, value * 100))
    if delta < -0.1:
        color = "#ef4444"
    elif delta < -0.05:
        color = "#fb923c"
    elif delta > 0.05:
        color = "#22c55e"
    else:
        color = "#3b82f6"
    return (
        f'<div class="bar-container">'
        f'<div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f"</div> {value:.2f}"
    )


def _svg_risk_timeline(
    frames: list[dict[str, Any]],
    *,
    width: int = 700,
    height: int = 200,
) -> str:
    """Generate an SVG chart showing before/after risk scores per frame."""
    if not frames:
        return '<p style="color:#94a3b8;font-size:0.82rem">No matched frames.</p>'

    PAD_L, PAD_R, PAD_T, PAD_B = 50, 16, 20, 38
    pw = width - PAD_L - PAD_R
    ph = height - PAD_T - PAD_B
    n = len(frames)

    def px(i: int) -> float:
        return PAD_L + (i / max(n - 1, 1)) * pw

    def py(val: float) -> float:
        return PAD_T + ph - val * ph

    parts: list[str] = []

    # Background
    parts.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" '
        f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="4"/>'
    )

    # Grid lines
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yp = py(frac)
        parts.append(
            f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{PAD_L + pw}" y2="{yp:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 4}" y="{yp:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="10" fill="#94a3b8">{frac:.2f}</text>'
        )

    # Before line (blue, dashed)
    before_pts = " ".join(f"{px(i):.1f},{py(f['before_risk']):.1f}" for i, f in enumerate(frames))
    parts.append(
        f'<polyline points="{before_pts}" fill="none" stroke="#3b82f6" '
        f'stroke-width="2" stroke-dasharray="6,4" stroke-linejoin="round"/>'
    )

    # After line (solid)
    after_pts = " ".join(f"{px(i):.1f},{py(f['after_risk']):.1f}" for i, f in enumerate(frames))
    parts.append(
        f'<polyline points="{after_pts}" fill="none" stroke="#ef4444" '
        f'stroke-width="2" stroke-linejoin="round"/>'
    )

    # Delta fill (green for improved, red for degraded)
    for i, f in enumerate(frames):
        if i == 0:
            continue
        prev = frames[i - 1]
        x0, x1 = px(i - 1), px(i)
        by0 = py(prev["before_risk"])
        by1 = py(f["before_risk"])
        ay0 = py(prev["after_risk"])
        ay1 = py(f["after_risk"])

        avg_delta = (prev["risk_delta"] + f["risk_delta"]) / 2
        fill_color = "rgba(239,68,68,0.15)" if avg_delta < 0 else "rgba(34,197,94,0.15)"

        parts.append(
            f'<polygon points="{x0:.1f},{by0:.1f} {x1:.1f},{by1:.1f} '
            f'{x1:.1f},{ay1:.1f} {x0:.1f},{ay0:.1f}" fill="{fill_color}"/>'
        )

    # Hotspot markers
    for i, f in enumerate(frames):
        if f["risk_delta"] < -0.1:
            parts.append(
                f'<circle cx="{px(i):.1f}" cy="{py(f["after_risk"]):.1f}" r="5" '
                f'fill="#ef4444" stroke="#fff" stroke-width="1.5">'
                f"<title>{escape(f['prompt'])}: {f['risk_delta']:+.4f}</title>"
                f"</circle>"
            )

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ph}" '
        f'stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + ph}" x2="{PAD_L + pw}" y2="{PAD_T + ph}" '
        f'stroke="#94a3b8" stroke-width="1"/>'
    )

    # X labels
    tick_count = min(n, 12)
    for t in range(tick_count + 1):
        idx = round(t * (n - 1) / max(tick_count, 1))
        xp = px(idx)
        parts.append(
            f'<text x="{xp:.1f}" y="{PAD_T + ph + 16}" text-anchor="middle" '
            f'font-size="9" fill="#94a3b8">#{idx + 1}</text>'
        )

    # Y label
    parts.append(
        f'<text x="{PAD_L + pw / 2:.0f}" y="{height - 2}" '
        f'text-anchor="middle" font-size="10" fill="#64748b">frame index</text>'
    )

    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block">\n  {inner}\n</svg>'
    )


def write_temporal_diff_report(
    path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    title: str = "Temporal Vegetation Change Report",
    before_label: str = "Before",
    after_label: str = "After",
) -> dict[str, Any]:
    """Write an HTML temporal diff report and return the diff data."""
    path.parent.mkdir(parents=True, exist_ok=True)

    diff = temporal_diff(before, after)
    summary = diff["summary"]
    frames = diff["frames"]
    hotspots = diff["hotspots"]

    # Sort frames by prompt for the timeline chart
    timeline_frames = sorted(frames, key=lambda f: f["prompt"])

    # Summary cards
    avg_delta = summary.get("avg_risk_delta", 0.0)
    avg_cls = "degraded" if avg_delta < -0.02 else ("improved" if avg_delta > 0.02 else "stable")

    # Dimension summary table
    dim_summary = summary.get("dim_summary", {})
    dim_rows = ""
    for dim, stats in dim_summary.items():
        mean_val = stats["mean"]
        cls = (
            "delta-neg"
            if mean_val > 0.01 and "vegetation" in dim
            else (
                "delta-pos"
                if mean_val < -0.01 and "vegetation" in dim
                else (
                    "delta-neg"
                    if mean_val < -0.01 and "risk" in dim
                    else ("delta-pos" if mean_val > 0.01 and "risk" in dim else "delta-zero")
                )
            )
        )
        dim_rows += (
            f"<tr><td><code>{escape(dim)}</code></td>"
            f'<td><span class="{cls}">{mean_val:+.6f}</span></td>'
            f"<td>{stats['min']:+.6f}</td>"
            f"<td>{stats['max']:+.6f}</td></tr>"
        )

    # Frame table (show all frames)
    frame_rows = ""
    for f in frames:
        is_hotspot = f["risk_delta"] < -0.1
        row_cls = ' class="hotspot-row"' if is_hotspot else ""
        frame_rows += (
            f"<tr{row_cls}>"
            f"<td><code>{escape(f['prompt'])}</code></td>"
            f"<td>{_risk_bar(f['before_risk'], 0)}</td>"
            f"<td>{_risk_tag(f['before_level'])}</td>"
            f"<td>{_risk_bar(f['after_risk'], f['risk_delta'])}</td>"
            f"<td>{_risk_tag(f['after_level'])}</td>"
            f"<td>{_delta_html(f['risk_delta'])}</td>"
            f"</tr>"
        )

    # SVG chart
    chart_svg = _svg_risk_timeline(timeline_frames)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>{_DIFF_CSS}</style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="subtitle">
    {escape(before_label)} vs {escape(after_label)}
    &nbsp;&middot;&nbsp;
    {summary["matched_count"]} frames matched
  </p>

  <section>
    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="stat-card">
        <div class="stat-value">{summary["matched_count"]}</div>
        <div class="stat-label">frames matched</div>
      </div>
      <div class="stat-card">
        <div class="stat-value {avg_cls}">{avg_delta:+.4f}</div>
        <div class="stat-label">avg risk delta</div>
      </div>
      <div class="stat-card">
        <div class="stat-value degraded">{summary.get("degraded_count", 0)}</div>
        <div class="stat-label">degraded</div>
      </div>
      <div class="stat-card">
        <div class="stat-value improved">{summary.get("improved_count", 0)}</div>
        <div class="stat-label">improved</div>
      </div>
      <div class="stat-card">
        <div class="stat-value stable">{summary.get("stable_count", 0)}</div>
        <div class="stat-label">stable</div>
      </div>
      <div class="stat-card">
        <div class="stat-value degraded">{len(hotspots)}</div>
        <div class="stat-label">hotspots (&Delta; &lt; -0.1)</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Risk Score Timeline</h2>
    <p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">
      <span style="color:#3b82f6">&#9644; &#9644;</span> {escape(before_label)}
      &nbsp;&nbsp;
      <span style="color:#ef4444">&#9644;&#9644;&#9644;</span> {escape(after_label)}
      &nbsp;&nbsp;
      <span style="color:#ef4444">&#9679;</span> hotspot (&Delta; &lt; -0.1)
    </p>
    {chart_svg}
  </section>

  <section>
    <h2>Dimension Deltas (aggregate)</h2>
    <table>
      <thead><tr><th>Dimension</th><th>Mean &Delta;</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>{dim_rows if dim_rows else '<tr><td colspan="4" style="color:#94a3b8">No dimension data</td></tr>'}</tbody>
    </table>
  </section>

  {"<section><h2>Hotspots</h2><p style='font-size:0.82rem;color:#64748b;margin:0 0 10px'>Locations where vegetation risk significantly increased (&Delta; &lt; -0.1)</p><table><thead><tr><th>Frame</th><th>Before</th><th>After</th><th>&Delta;</th></tr></thead><tbody>" + "".join(f"<tr><td><code>{escape(h['prompt'])}</code></td><td>{h['before_risk']:.4f} {_risk_tag(h['before_level'])}</td><td>{h['after_risk']:.4f} {_risk_tag(h['after_level'])}</td><td>{_delta_html(h['risk_delta'])}</td></tr>" for h in hotspots) + "</tbody></table></section>" if hotspots else ""}

  <section>
    <h2>Per-frame Comparison</h2>
    <table>
      <thead><tr>
        <th>Frame</th>
        <th>{escape(before_label)} Score</th>
        <th>Level</th>
        <th>{escape(after_label)} Score</th>
        <th>Level</th>
        <th>&Delta;</th>
      </tr></thead>
      <tbody>{frame_rows if frame_rows else '<tr><td colspan="6" style="color:#94a3b8">No matched frames</td></tr>'}</tbody>
    </table>
  </section>
</body>
</html>
"""
    path.write_text(html_content, encoding="utf-8")
    return diff
