"""Self-contained HTML inspection report with embedded heatmap thumbnails.

Generates a single HTML file containing:
- Run summary (score, gates, pass/fail status)
- Per-frame table with base64-encoded heatmap thumbnails
- Alert frame highlights
- Zone boundary legend
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path
from typing import Any

try:
    from PIL import Image

    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from temporalci.heatmap import generate_heatmap

_CSS = """\
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
}
h1 { margin: 0 0 8px; font-size: 1.5rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 28px 0 10px;
     border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
.subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 16px; }
section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
          padding: 20px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 6px 14px; border-radius: 6px;
         color: #fff; font-weight: 700; font-size: 1rem; margin-bottom: 16px; }
.badge-pass { background: #15803d; }
.badge-fail { background: #b91c1c; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 8px; margin-bottom: 4px; }
.meta-item { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
             padding: 8px 12px; font-size: 0.82rem; }
.meta-item strong { display: block; color: #64748b; font-size: 0.75rem; margin-bottom: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #f8fafc; font-weight: 600; text-align: left; padding: 8px 10px;
     border-bottom: 2px solid #e2e8f0; }
td { padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
tr:hover td { background: #f8fafc; }
.tag-pass { display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-weight: 600; font-size: 0.78rem; background: #dcfce7; color: #15803d; }
.tag-fail { display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-weight: 600; font-size: 0.78rem; background: #fee2e2; color: #b91c1c; }
.tag-alert { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-weight: 600; font-size: 0.78rem; background: #fef3c7; color: #92400e; }
.thumb { border-radius: 4px; border: 1px solid #e2e8f0; cursor: pointer;
         transition: transform 0.15s; }
.thumb:hover { transform: scale(2.5); z-index: 10; position: relative; }
.bar { height: 8px; border-radius: 4px; display: inline-block; vertical-align: middle; }
.bar-green { background: #22c55e; }
.bar-gray { background: #e2e8f0; }
.legend-item { display: inline-flex; align-items: center; gap: 6px;
               margin-right: 16px; font-size: 0.8rem; color: #64748b; }
.legend-swatch { width: 16px; height: 16px; border-radius: 3px; display: inline-block; }
"""


def _img_to_base64(img: "Image.Image", max_width: int = 320) -> str:
    """Resize and encode a PIL image to a base64 data URI."""
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _proximity_bar(value: float, width: int = 80) -> str:
    """Render a small inline bar for vegetation proximity."""
    pct = min(max(value * 100, 0), 100)
    filled = int(pct / 100 * width)
    color = "#22c55e" if pct < 30 else ("#f59e0b" if pct < 60 else "#ef4444")
    return (
        f'<span class="bar" style="width:{filled}px;background:{color}"></span>'
        f'<span class="bar bar-gray" style="width:{width - filled}px"></span>'
        f' <span style="font-size:0.78rem">{pct:.1f}%</span>'
    )


def write_inspection_report(
    output_path: str | Path,
    *,
    run_data: dict[str, Any],
    frame_dir: str | Path,
    pattern: str = "*.jpg",
    thumbnail_width: int = 280,
) -> Path:
    """Generate an HTML inspection report with embedded heatmap thumbnails.

    Parameters
    ----------
    output_path:
        Where to write the HTML file.
    run_data:
        The run result dict (from run.json).
    frame_dir:
        Directory containing the source frames.
    pattern:
        Glob pattern for frame files.
    thumbnail_width:
        Max width for embedded thumbnails.

    Returns
    -------
    Path to the written HTML file.
    """
    if not _HAS_DEPS:
        raise RuntimeError("inspection report requires Pillow and numpy")

    output = Path(output_path)
    frame_dir = Path(frame_dir)

    status = str(run_data.get("status", "UNKNOWN"))
    badge_cls = "badge-pass" if status == "PASS" else "badge-fail"

    metrics = run_data.get("metrics", {})
    cv_metrics = metrics.get("catenary_vegetation", {})
    dims = cv_metrics.get("dims", {})
    per_sample = cv_metrics.get("per_sample", [])
    alert_frames_list = cv_metrics.get("alert_frames", [])
    alert_prompts = {a["prompt"] for a in alert_frames_list}
    gates = run_data.get("gates", [])

    # Meta info section
    meta_html = "".join(
        [
            _meta("Run ID", run_data.get("run_id")),
            _meta("Project", run_data.get("project")),
            _meta("Suite", run_data.get("suite_name")),
            _meta("Model", run_data.get("model_name")),
            _meta("Timestamp", run_data.get("timestamp_utc")),
            _meta("Samples", run_data.get("sample_count")),
            _meta("Score", f"{cv_metrics.get('score', 0):.4f}"),
            _meta("Alerts", str(len(alert_frames_list))),
        ]
    )

    # Dimension summary
    dims_html = "".join(
        f'<div class="meta-item"><strong>{escape(k)}</strong>{v:.4f}</div>' for k, v in dims.items()
    )

    # Gates table
    gates_rows = ""
    for g in gates:
        passed = g.get("passed", False)
        tag = (
            '<span class="tag-pass">PASS</span>' if passed else '<span class="tag-fail">FAIL</span>'
        )
        gates_rows += (
            f"<tr><td><code>{escape(str(g.get('metric', '')))}</code></td>"
            f"<td>{escape(str(g.get('op', '')))}</td>"
            f"<td>{escape(str(g.get('value', '')))}</td>"
            f"<td>{escape(str(g.get('actual', '—')))}</td>"
            f"<td>{tag}</td></tr>"
        )

    # Per-frame table with heatmap thumbnails
    frame_rows = _build_frame_rows(
        frame_dir=frame_dir,
        pattern=pattern,
        per_sample=per_sample,
        alert_prompts=alert_prompts,
        thumbnail_width=thumbnail_width,
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Inspection Report — {escape(run_data.get("project", ""))} — {escape(status)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Catenary Vegetation Inspection Report</h1>
  <p class="subtitle">{escape(run_data.get("project", ""))} / {escape(run_data.get("suite_name", ""))}</p>
  <div class="badge {badge_cls}">{escape(status)}</div>

  <section>
    <div class="meta-grid">{meta_html}</div>
  </section>

  <section>
    <h2>Dimension Averages</h2>
    <div class="meta-grid">{dims_html}</div>
  </section>

  <section>
    <h2>Gate Results</h2>
    <table>
      <thead><tr><th>Metric</th><th>Op</th><th>Target</th><th>Actual</th><th>Status</th></tr></thead>
      <tbody>{gates_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Per-Frame Analysis</h2>
    <div style="margin-bottom:12px">
      <span class="legend-item"><span class="legend-swatch" style="background:rgba(220,40,40,0.45)"></span> Detected vegetation</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#00dcff;opacity:0.7"></span> Catenary zone (upper 1/4)</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#64b4dc;opacity:0.5"></span> Upper half boundary</span>
    </div>
    <table>
      <thead><tr><th>Frame</th><th>Heatmap</th><th>Proximity</th><th>Coverage</th><th>Visibility</th><th>Status</th></tr></thead>
      <tbody>{frame_rows}</tbody>
    </table>
  </section>

  <footer style="text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:24px">
    Generated by TemporalCI Catenary Inspector
  </footer>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def _meta(label: str, value: object) -> str:
    return (
        f'<div class="meta-item"><strong>{escape(label)}</strong>'
        f"{escape(str(value) if value is not None else '—')}</div>"
    )


def _build_frame_rows(
    *,
    frame_dir: Path,
    pattern: str,
    per_sample: list[dict[str, Any]],
    alert_prompts: set[str],
    thumbnail_width: int,
) -> str:
    """Build HTML table rows with embedded heatmap thumbnails."""
    # Map prompt → per-sample data
    sample_map: dict[str, dict[str, Any]] = {}
    for s in per_sample:
        sample_map[s.get("prompt", "")] = s

    frames = sorted(p for p in frame_dir.glob(pattern) if p.is_file())
    rows: list[str] = []

    for frame_path in frames:
        prompt = frame_path.stem
        sample = sample_map.get(prompt, {})
        sample_dims = sample.get("dims", {})
        prox = sample_dims.get("vegetation_proximity", 0.0)
        cov = sample_dims.get("green_coverage", 0.0)
        vis = sample_dims.get("catenary_visibility", 0.0)

        is_alert = prompt in alert_prompts

        # Generate heatmap in memory
        heatmap_info = generate_heatmap(
            frame_path,
            frame_path.parent / f"_tmp_hm_{prompt}.png",
            overlay_alpha=0.45,
        )
        heatmap_img = Image.open(heatmap_info["output_path"])
        thumb_uri = _img_to_base64(heatmap_img, max_width=thumbnail_width)
        # Clean up temp file
        Path(heatmap_info["output_path"]).unlink(missing_ok=True)

        alert_tag = ' <span class="tag-alert">ALERT</span>' if is_alert else ""
        status_tag = (
            '<span class="tag-fail">HIGH</span>' if is_alert else '<span class="tag-pass">OK</span>'
        )

        rows.append(
            f"<tr>"
            f"<td><code>{escape(prompt)}</code>{alert_tag}</td>"
            f'<td><img src="{thumb_uri}" class="thumb" width="{thumbnail_width}" alt="{escape(prompt)}"/></td>'
            f"<td>{_proximity_bar(prox)}</td>"
            f"<td>{cov:.4f}</td>"
            f"<td>{vis:.4f}</td>"
            f"<td>{status_tag}</td>"
            f"</tr>"
        )

    return "".join(rows)
