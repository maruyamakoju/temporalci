"""Auto-generated project-level suite index HTML.

Scans a suite root directory (artifacts/project/suite/) for model
subdirectories and writes index.html summarising each model's latest
run status, recent pass/fail strip, and aggregate metrics.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_latest_run(model_root: Path) -> dict[str, Any] | None:
    """Return the most recent run payload from *model_root*, or None."""
    runs_jsonl = model_root / "runs.jsonl"
    if not runs_jsonl.exists():
        return None
    try:
        lines = [
            line.strip()
            for line in runs_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            run_id = str(entry.get("run_id", "")).strip()
            if not run_id:
                continue
            run_json = model_root / run_id / "run.json"
            if run_json.exists():
                payload = json.loads(run_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            # Minimal stub from jsonl entry
            return {
                "run_id": run_id,
                "status": entry.get("status", "UNKNOWN"),
                "timestamp_utc": entry.get("timestamp_utc", ""),
                "sample_count": entry.get("sample_count", 0),
                "metrics": {},
                "gates": [],
            }
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _recent_statuses(model_root: Path, *, last_n: int = 20) -> list[str]:
    """Return the last *last_n* run statuses from runs.jsonl."""
    runs_jsonl = model_root / "runs.jsonl"
    if not runs_jsonl.exists():
        return []
    try:
        lines = [
            line.strip()
            for line in runs_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return []
    statuses: list[str] = []
    for line in lines[-last_n:]:
        try:
            entry = json.loads(line)
            statuses.append(str(entry.get("status", "UNKNOWN")))
        except json.JSONDecodeError:
            continue
    return statuses


def _scalar_metrics(run: dict[str, Any]) -> dict[str, float]:
    """Return all scalar float metric values from *run*, keyed by dotted path."""
    out: dict[str, float] = {}

    def _walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                out[path] = float(v)
            elif isinstance(v, dict):
                _walk(v, path)

    _walk(run.get("metrics") or {}, "")
    return out


def discover_models(suite_root: Path) -> list[tuple[str, Path]]:
    """Return ``(model_name, model_root)`` pairs found under *suite_root*."""
    if not suite_root.exists():
        return []
    result: list[tuple[str, Path]] = []
    for child in sorted(suite_root.iterdir()):
        if child.is_dir() and (child / "runs.jsonl").exists():
            result.append((child.name, child))
    return result


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_PASS_COLOR = "#15803d"
_FAIL_COLOR = "#b91c1c"
_UNK_COLOR = "#94a3b8"


def _svg_mini_strip(statuses: list[str], *, width: int = 180, height: int = 18) -> str:
    """Compact SVG colored strip showing PASS/FAIL history."""
    if not statuses:
        return ""
    n = len(statuses)
    cell_w = width / n
    parts: list[str] = []
    for i, s in enumerate(statuses):
        color = _PASS_COLOR if s == "PASS" else (_FAIL_COLOR if s == "FAIL" else _UNK_COLOR)
        x = i * cell_w
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{cell_w:.1f}" height="{height}" fill="{color}">'
            f"<title>#{i + 1}: {escape(s)}</title></rect>"
        )
    inner = "".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="border-radius:3px;display:block">{inner}</svg>'
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_INDEX_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
  }
  h1 { margin: 0 0 4px; font-size: 1.5rem; }
  .subtitle { color: #64748b; font-size: 0.88rem; margin: 0 0 22px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }
  .card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px 18px; display: flex; flex-direction: column; gap: 10px;
  }
  .card-header { display: flex; align-items: center; gap: 10px; }
  .card-title { font-weight: 700; font-size: 0.97rem; flex: 1; word-break: break-word; }
  .badge { display:inline-block; padding:2px 9px; border-radius:5px; font-weight:700;
           font-size:0.78rem; color:#fff; white-space:nowrap; }
  .badge-pass { background:#15803d; }
  .badge-fail { background:#b91c1c; }
  .badge-unk  { background:#64748b; }
  .meta { font-size: 0.78rem; color: #64748b; }
  .metrics-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .metrics-table td { padding: 1px 0; border: none; }
  .metrics-table td:last-child { text-align: right; font-weight: 600;
      font-family: "SF Mono","Fira Code",monospace; }
  a.trend-link { font-size: 0.78rem; color: #2563eb; text-decoration: none; }
  a.trend-link:hover { text-decoration: underline; }
  .no-models { color: #94a3b8; font-style: italic; padding: 20px 0; }
  code { font-family: "SF Mono","Fira Code",monospace; font-size: 0.82em;
         background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
"""


def _status_badge(status: str) -> str:
    cls = "badge-pass" if status == "PASS" else ("badge-fail" if status == "FAIL" else "badge-unk")
    return f'<span class="badge {cls}">{escape(status)}</span>'


def _render_model_card(model_name: str, model_root: Path, suite_root: Path) -> str:
    latest = _load_latest_run(model_root)
    statuses = _recent_statuses(model_root, last_n=20)

    if latest is None:
        return (
            f'<div class="card">'
            f'<div class="card-header"><span class="card-title">{escape(model_name)}</span>'
            f"{_status_badge('UNKNOWN')}</div>"
            f'<p class="meta">No runs found.</p>'
            f"</div>"
        )

    status = str(latest.get("status", "UNKNOWN"))
    ts = str(latest.get("timestamp_utc", ""))[:19]
    run_id = str(latest.get("run_id", ""))
    sample_count = latest.get("sample_count", 0)

    # Prefer shallow paths (aggregate metrics first)
    all_scalars = _scalar_metrics(latest)
    sorted_metrics = sorted(all_scalars.items(), key=lambda kv: (kv[0].count("."), kv[0]))[:5]
    metric_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{v:.4f}</td></tr>" for k, v in sorted_metrics
    )
    metric_html = f'<table class="metrics-table">{metric_rows}</table>' if metric_rows else ""

    strip_svg = _svg_mini_strip(statuses)

    trend_path = model_root / "trend_report.html"
    trend_link = ""
    if trend_path.exists():
        try:
            rel = trend_path.relative_to(suite_root)
            href = str(rel).replace("\\", "/")
        except ValueError:
            href = str(trend_path).replace("\\", "/")
        trend_link = f'<a class="trend-link" href="{escape(href)}">→ trend report</a>'

    return (
        f'<div class="card">'
        f'<div class="card-header">'
        f'<span class="card-title">{escape(model_name)}</span>'
        f"{_status_badge(status)}"
        f"</div>"
        f"<div>{strip_svg}</div>"
        f'<div class="meta">'
        f"<code>{escape(run_id)}</code><br/>"
        f"{escape(ts)} &nbsp;·&nbsp; {sample_count} samples"
        f"</div>"
        f"{metric_html}"
        f"{trend_link}"
        f"</div>"
    )


def write_suite_index(
    suite_root: Path,
    *,
    project: str,
    suite_name: str,
) -> None:
    """Write ``suite_root/index.html`` summarising all models."""
    suite_root.mkdir(parents=True, exist_ok=True)
    models = discover_models(suite_root)

    if models:
        cards_html = (
            '<div class="grid">'
            + "".join(_render_model_card(name, root, suite_root) for name, root in models)
            + "</div>"
        )
    else:
        cards_html = '<p class="no-models">No model runs found yet.</p>'

    from temporalci.utils import utc_now

    generated_at = utc_now().isoformat()[:19]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TemporalCI — {escape(project)} / {escape(suite_name)}</title>
  <style>{_INDEX_CSS}</style>
</head>
<body>
  <h1>TemporalCI</h1>
  <p class="subtitle">
    {escape(project)} / {escape(suite_name)}
    &nbsp;·&nbsp; {len(models)} model{"s" if len(models) != 1 else ""}
    &nbsp;·&nbsp; generated {escape(generated_at)}
  </p>
  {cards_html}
</body>
</html>
"""
    (suite_root / "index.html").write_text(html, encoding="utf-8")
