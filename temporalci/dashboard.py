"""Interactive HTML dashboard for catenary inspection results.

Generates a single self-contained HTML file with:
- Summary statistics cards
- Risk distribution chart
- Risk timeline (per-frame scores)
- Alert table with severity indicators
- Dimension breakdown charts
- Performance metrics
- Embedded interactive map (if GPS data available)
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _risk_color(level: str) -> str:
    return {
        "critical": "#ef4444",
        "warning": "#fb923c",
        "caution": "#facc15",
        "safe": "#22c55e",
    }.get(level, "#94a3b8")


def _svg_risk_distribution(
    distribution: dict[str, int], *, width: int = 400, height: int = 32
) -> str:
    total = sum(distribution.values())
    if total == 0:
        return ""
    parts: list[str] = []
    x = 0.0
    order = ["critical", "warning", "caution", "safe"]
    for level in order:
        count = distribution.get(level, 0)
        if count == 0:
            continue
        w = (count / total) * width
        color = _risk_color(level)
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height}" '
            f'fill="{color}" rx="0">'
            f"<title>{level}: {count} ({count * 100 / total:.0f}%)</title>"
            f"</rect>"
        )
        if w > 30:
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{height / 2 + 1}" '
                f'text-anchor="middle" dominant-baseline="middle" '
                f'font-size="11" font-weight="600" fill="white">{count}</text>'
            )
        x += w
    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block;border-radius:6px;overflow:hidden">\n  {inner}\n</svg>'
    )


def _svg_timeline(
    frames: list[dict[str, Any]],
    *,
    width: int = 900,
    height: int = 200,
) -> str:
    if not frames:
        return ""
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

    # Danger zone (below 0.5)
    y_half = py(0.5)
    parts.append(
        f'<rect x="{PAD_L}" y="{y_half:.1f}" width="{pw}" '
        f'height="{PAD_T + ph - y_half:.1f}" fill="rgba(239,68,68,0.06)" rx="0"/>'
    )

    # Grid
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

    # Area fill
    area_pts = [f"{px(0):.1f},{PAD_T + ph:.1f}"]
    for i, f in enumerate(frames):
        score = f.get("risk_score", 0.5)
        area_pts.append(f"{px(i):.1f},{py(score):.1f}")
    area_pts.append(f"{px(n - 1):.1f},{PAD_T + ph:.1f}")
    parts.append(f'<polygon points="{" ".join(area_pts)}" fill="url(#riskGrad)" opacity="0.3"/>')

    # Gradient definition
    parts.insert(
        0,
        '<defs><linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#22c55e"/>'
        '<stop offset="50%" stop-color="#facc15"/>'
        '<stop offset="100%" stop-color="#ef4444"/>'
        "</linearGradient></defs>",
    )

    # Line
    line_pts = " ".join(
        f"{px(i):.1f},{py(f.get('risk_score', 0.5)):.1f}" for i, f in enumerate(frames)
    )
    parts.append(
        f'<polyline points="{line_pts}" fill="none" stroke="#3b82f6" '
        f'stroke-width="2" stroke-linejoin="round"/>'
    )

    # Data points
    for i, f in enumerate(frames):
        score = f.get("risk_score", 0.5)
        level = f.get("risk_level", "unknown")
        color = _risk_color(level)
        fid = escape(str(f.get("frame_id", f"#{i}")))
        parts.append(
            f'<circle cx="{px(i):.1f}" cy="{py(score):.1f}" r="4" '
            f'fill="{color}" stroke="#fff" stroke-width="1.5">'
            f"<title>{fid}: {score:.4f} ({level})</title>"
            f"</circle>"
        )

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ph}" '
        f'stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{PAD_L + pw / 2:.0f}" y="{height - 2}" text-anchor="middle" '
        f'font-size="10" fill="#64748b">frame index</text>'
    )

    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block">\n  {inner}\n</svg>'
    )


def _svg_dimension_bars(dims: dict[str, float], *, width: int = 500) -> str:
    if not dims:
        return ""
    parts: list[str] = []
    row_h = 28
    label_w = 200
    bar_w = width - label_w - 60
    h = row_h * len(dims) + 4

    for i, (name, value) in enumerate(dims.items()):
        y = i * row_h + 2
        val = max(0.0, min(1.0, value))

        # Determine color based on dimension semantics
        if "vegetation" in name or "penetration" in name:
            # Higher = worse for vegetation metrics
            color = (
                _risk_color("critical")
                if val > 0.5
                else (_risk_color("warning") if val > 0.3 else _risk_color("safe"))
            )
        elif "risk" in name or "clearance" in name:
            # Higher = better for risk_score / clearance
            color = (
                _risk_color("safe")
                if val > 0.5
                else (_risk_color("warning") if val > 0.3 else _risk_color("critical"))
            )
        else:
            color = "#3b82f6"

        parts.append(
            f'<text x="0" y="{y + row_h / 2 + 1}" '
            f'dominant-baseline="middle" font-size="11" fill="#475569">'
            f"{escape(name)}</text>"
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 4}" width="{bar_w}" height="{row_h - 8}" '
            f'fill="#f1f5f9" rx="3"/>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 4}" width="{bar_w * val:.1f}" '
            f'height="{row_h - 8}" fill="{color}" rx="3" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{label_w + bar_w + 4}" y="{y + row_h / 2 + 1}" '
            f'dominant-baseline="middle" font-size="11" fill="#64748b" '
            f'font-weight="600">{value:.4f}</text>'
        )

    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
        f'style="max-width:100%;display:block">\n  {inner}\n</svg>'
    )


def generate_dashboard(
    results: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "Catenary Inspection Dashboard",
    stats: dict[str, Any] | None = None,
) -> Path:
    """Generate a comprehensive HTML dashboard from inspection results.

    Parameters
    ----------
    results
        Metric evaluation result dict (from catenary_clearance.evaluate or
        OnnxPipeline output) containing: score, dims, per_sample, alert_frames.
    output_path
        Where to write the HTML file.
    title
        Dashboard title.
    stats
        Optional pipeline performance stats (from PipelineStats.to_dict()).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = results.get("score", 0.0)
    dims = results.get("dims", {})
    per_sample = results.get("per_sample", [])
    alert_frames = results.get("alert_frames", [])
    sample_count = results.get("sample_count", len(per_sample))

    # Risk distribution
    distribution: dict[str, int] = {}
    frame_data: list[dict[str, Any]] = []
    for s in per_sample:
        level = s.get("risk_level", "unknown")
        distribution[level] = distribution.get(level, 0) + 1
        frame_data.append(
            {
                "frame_id": s.get("prompt", s.get("test_id", "")),
                "risk_score": s.get("dims", {}).get("risk_score", 0.0),
                "risk_level": level,
            }
        )

    # Score color
    if score >= 0.7:
        score_color = "#22c55e"
        score_label = "GOOD"
    elif score >= 0.4:
        score_color = "#fb923c"
        score_label = "MODERATE"
    else:
        score_color = "#ef4444"
        score_label = "POOR"

    # Summary cards HTML
    n_critical = distribution.get("critical", 0)
    n_warning = distribution.get("warning", 0)
    n_safe = distribution.get("safe", 0) + distribution.get("caution", 0)

    # Alert table
    alert_rows = ""
    for a in alert_frames[:20]:
        color = _risk_color(a.get("risk_level", "unknown"))
        alert_rows += (
            f"<tr>"
            f"<td><code>{escape(str(a.get('prompt', '')))}</code></td>"
            f'<td style="color:{color};font-weight:700">'
            f"{escape(str(a.get('risk_level', '')).upper())}</td>"
            f"<td>{a.get('risk_score', 0):.4f}</td>"
            f"<td>{a.get('clearance_px', 0):.1f}px</td>"
            f"<td>{a.get('vegetation_zone', 0) * 100:.1f}%</td>"
            f"</tr>"
        )

    # Per-frame table
    frame_rows = ""
    for s in per_sample[:50]:
        level = s.get("risk_level", "unknown")
        color = _risk_color(level)
        s_dims = s.get("dims", {})
        frame_rows += (
            f"<tr>"
            f"<td><code>{escape(str(s.get('prompt', '')))}</code></td>"
            f'<td style="color:{color};font-weight:600">{escape(level.upper())}</td>'
            f"<td>{s_dims.get('risk_score', 0):.4f}</td>"
            f"<td>{s_dims.get('vegetation_proximity_nn', 0):.4f}</td>"
            f"<td>{s_dims.get('vegetation_penetration', 0):.4f}</td>"
            f"<td>{s.get('wire_count', 0)}</td>"
            f"<td>{s.get('clearance_px', 0):.1f}</td>"
            f"</tr>"
        )

    # SVGs
    dist_svg = _svg_risk_distribution(distribution)
    timeline_svg = _svg_timeline(frame_data)
    dim_svg = _svg_dimension_bars(dims)

    # Performance stats section
    perf_html = ""
    if stats:
        perf_html = f"""
  <section>
    <h2>Performance</h2>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">{stats.get("avg_ms_per_frame", 0):.0f}ms</div>
        <div class="stat-label">avg per frame</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats.get("fps", 0):.1f}</div>
        <div class="stat-label">frames/sec</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats.get("seg_elapsed_ms", 0) / max(stats.get("total_frames", 1), 1):.0f}ms</div>
        <div class="stat-label">segmentation</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats.get("depth_elapsed_ms", 0) / max(stats.get("total_frames", 1), 1):.0f}ms</div>
        <div class="stat-label">depth</div>
      </div>
    </div>
  </section>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(title)}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 32px 32px 24px; border-bottom: 1px solid #334155;
  }}
  .header h1 {{ margin: 0; font-size: 1.5rem; color: #f8fafc; }}
  .header .subtitle {{ color: #94a3b8; font-size: 0.88rem; margin: 4px 0 0; }}
  .content {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  section {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 24px; margin-bottom: 16px;
  }}
  h2 {{
    font-size: 1rem; font-weight: 600; margin: 0 0 16px;
    color: #f1f5f9; border-bottom: 1px solid #334155; padding-bottom: 8px;
  }}
  .stat-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
  }}
  .stat-card {{
    background: #0f172a; border: 1px solid #334155; border-radius: 10px;
    padding: 16px; text-align: center;
  }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; line-height: 1.2; }}
  .stat-label {{ font-size: 0.78rem; color: #64748b; margin-top: 4px; }}
  .score-ring {{
    width: 120px; height: 120px; margin: 0 auto 12px;
    position: relative;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th {{
    background: #0f172a; font-weight: 600; text-align: left;
    padding: 8px 10px; border-bottom: 2px solid #334155; color: #94a3b8;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b; color: #cbd5e1; }}
  tr:hover td {{ background: #334155; }}
  code {{
    font-family: "SF Mono","Fira Code",monospace; font-size: 0.85em;
    background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #93c5fd;
  }}
  .legend {{
    display: flex; gap: 16px; font-size: 0.8rem; color: #94a3b8;
    margin-top: 8px; flex-wrap: wrap;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block;
  }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <div class="header">
    <h1>{escape(title)}</h1>
    <p class="subtitle">
      {sample_count} frames analyzed &middot;
      {len(alert_frames)} alerts &middot;
      Score: {score:.4f}
    </p>
  </div>

  <div class="content">
    <section>
      <h2>Overview</h2>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value" style="color:{score_color}">{score:.4f}</div>
          <div class="stat-label">composite score ({score_label})</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{sample_count}</div>
          <div class="stat-label">frames analyzed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:#ef4444">{n_critical}</div>
          <div class="stat-label">critical</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:#fb923c">{n_warning}</div>
          <div class="stat-label">warning</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:#22c55e">{n_safe}</div>
          <div class="stat-label">safe / caution</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color:#ef4444">{len(alert_frames)}</div>
          <div class="stat-label">alerts</div>
        </div>
      </div>
    </section>

    <section>
      <h2>Risk Distribution</h2>
      {dist_svg}
      <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Critical</div>
        <div class="legend-item"><span class="legend-dot" style="background:#fb923c"></span>Warning</div>
        <div class="legend-item"><span class="legend-dot" style="background:#facc15"></span>Caution</div>
        <div class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>Safe</div>
      </div>
    </section>

    <section>
      <h2>Risk Score Timeline</h2>
      {timeline_svg}
      <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#3b82f6"></span>Risk Score</div>
        <div class="legend-item">
          <span style="width:30px;height:12px;display:inline-block;background:rgba(239,68,68,0.15);border-radius:2px"></span>
          Danger zone (&lt;0.5)
        </div>
      </div>
    </section>

    <div class="two-col">
      <section>
        <h2>Dimensions (aggregate)</h2>
        {dim_svg}
      </section>
      <section>
        <h2>Alerts ({len(alert_frames)})</h2>
        {"<table><thead><tr><th>Frame</th><th>Risk</th><th>Score</th><th>Clearance</th><th>Veg Zone</th></tr></thead><tbody>" + alert_rows + "</tbody></table>" if alert_rows else '<p style="color:#64748b;font-size:0.88rem">No alerts.</p>'}
      </section>
    </div>

    {perf_html}

    <section>
      <h2>Per-frame Analysis</h2>
      <table>
        <thead><tr>
          <th>Frame</th><th>Risk</th><th>Score</th>
          <th>Veg Prox</th><th>Penetration</th>
          <th>Wires</th><th>Clearance</th>
        </tr></thead>
        <tbody>{frame_rows if frame_rows else '<tr><td colspan="7" style="color:#64748b">No data</td></tr>'}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
