from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

from temporalci.sprt import derive_sprt_metrics

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
  }
  h1 { margin: 0 0 16px; font-size: 1.5rem; }
  h2 { font-size: 1.1rem; font-weight: 600; margin: 28px 0 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
  h3 { font-size: 1rem; font-weight: 600; margin: 0 0 10px; }
  h4 { font-size: 0.9rem; font-weight: 600; margin: 16px 0 6px; color: #475569; }
  section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  .badge {
    display: inline-block; padding: 6px 14px; border-radius: 6px;
    color: #fff; font-weight: 700; font-size: 1rem; letter-spacing: 0.05em;
    margin-bottom: 16px;
  }
  .badge-pass { background: #15803d; }
  .badge-fail { background: #b91c1c; }
  .meta-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px;
    margin-bottom: 4px;
  }
  .meta-item {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 8px 12px; font-size: 0.82rem;
  }
  .meta-item strong { display: block; color: #64748b; font-size: 0.75rem; margin-bottom: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { background: #f8fafc; font-weight: 600; text-align: left; padding: 8px 10px; border-bottom: 2px solid #e2e8f0; }
  td { padding: 7px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f8fafc; }
  .tag-pass { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.78rem; background: #dcfce7; color: #15803d; }
  .tag-fail { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.78rem; background: #fee2e2; color: #b91c1c; }
  .tag-warn { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.78rem; background: #fef9c3; color: #92400e; }
  .delta-neg { color: #b91c1c; font-weight: 600; }
  .delta-pos { color: #15803d; font-weight: 600; }
  .delta-zero { color: #64748b; }
  .sprt-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 14px; }
  .sprt-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .sprt-metric { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.9rem; color: #334155; }
  .chart-wrap { margin: 12px 0; overflow-x: auto; }
  pre { margin: 0; white-space: pre-wrap; word-break: break-all; font-size: 0.8rem; color: #334155; }
  .stat-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  .stat-pill { background: #f1f5f9; border-radius: 6px; padding: 4px 10px; font-size: 0.8rem; }
  .stat-pill strong { margin-right: 4px; }
  .no-data { color: #94a3b8; font-style: italic; }
"""

# ---------------------------------------------------------------------------
# SVG: LLR trajectory chart
# ---------------------------------------------------------------------------

def _svg_llr_chart(
    *,
    llr_history: list[float],
    upper: float,
    lower: float,
    crossed_at: int | None,
    min_pairs: int,
    width: int = 640,
    height: int = 210,
) -> str:
    if not llr_history:
        return ""

    PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 16, 38
    pw = width - PAD_L - PAD_R
    ph = height - PAD_T - PAD_B
    n = len(llr_history)

    all_vals = llr_history + [upper, lower, 0.0]
    raw_min = min(all_vals)
    raw_max = max(all_vals)
    margin = (raw_max - raw_min) * 0.15 or 0.5
    y_min = raw_min - margin
    y_max = raw_max + margin
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_span = y_max - y_min

    def px(i: int) -> float:
        return PAD_L + ((i - 1) / max(n - 1, 1)) * pw

    def py(val: float) -> float:
        return PAD_T + ph - ((val - y_min) / y_span) * ph

    parts: list[str] = []

    # background
    parts.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" '
        f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="4"/>'
    )

    # horizontal grid lines + y-axis labels
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * y_span
        yp = py(yv)
        parts.append(
            f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{PAD_L + pw}" y2="{yp:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 4}" y="{yp:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="10" fill="#94a3b8">{yv:.3f}</text>'
        )

    # zero line
    if y_min < 0 < y_max:
        yp0 = py(0.0)
        parts.append(
            f'<line x1="{PAD_L}" y1="{yp0:.1f}" x2="{PAD_L + pw}" y2="{yp0:.1f}" '
            f'stroke="#cbd5e1" stroke-width="1"/>'
        )

    # upper threshold
    ypu = py(upper)
    parts.append(
        f'<line x1="{PAD_L}" y1="{ypu:.1f}" x2="{PAD_L + pw}" y2="{ypu:.1f}" '
        f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,3"/>'
    )
    parts.append(
        f'<text x="{PAD_L + 4}" y="{ypu - 4:.1f}" font-size="9" fill="#ef4444">H1 {upper:.3f}</text>'
    )

    # lower threshold
    ypl = py(lower)
    parts.append(
        f'<line x1="{PAD_L}" y1="{ypl:.1f}" x2="{PAD_L + pw}" y2="{ypl:.1f}" '
        f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,3"/>'
    )
    parts.append(
        f'<text x="{PAD_L + 4}" y="{ypl + 11:.1f}" font-size="9" fill="#ef4444">H0 {lower:.3f}</text>'
    )

    # min_pairs vertical line
    if 1 <= min_pairs <= n:
        xmp = px(min_pairs)
        parts.append(
            f'<line x1="{xmp:.1f}" y1="{PAD_T}" x2="{xmp:.1f}" y2="{PAD_T + ph}" '
            f'stroke="#94a3b8" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        parts.append(
            f'<text x="{xmp + 3:.1f}" y="{PAD_T + 10}" font-size="9" fill="#94a3b8">min={min_pairs}</text>'
        )

    # LLR path
    pts = " ".join(f"{px(i + 1):.1f},{py(v):.1f}" for i, v in enumerate(llr_history))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2" stroke-linejoin="round"/>')

    # crossing point
    if crossed_at is not None and 1 <= crossed_at <= n:
        cx = px(crossed_at)
        cy = py(llr_history[crossed_at - 1])
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#ef4444" stroke="#fff" stroke-width="1.5"/>'
        )

    # axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ph}" stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + ph}" x2="{PAD_L + pw}" y2="{PAD_T + ph}" stroke="#94a3b8" stroke-width="1"/>'
    )

    # x-axis ticks
    tick_count = min(n, 8)
    for t in range(tick_count + 1):
        i = max(1, round(1 + t * (n - 1) / max(tick_count, 1)))
        xp = px(i)
        parts.append(
            f'<line x1="{xp:.1f}" y1="{PAD_T + ph}" x2="{xp:.1f}" y2="{PAD_T + ph + 4}" stroke="#94a3b8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xp:.1f}" y="{PAD_T + ph + 16}" text-anchor="middle" font-size="10" fill="#94a3b8">{i}</text>'
        )

    # axis labels
    parts.append(
        f'<text x="{PAD_L + pw / 2:.0f}" y="{height - 2}" text-anchor="middle" font-size="10" fill="#64748b">pair index</text>'
    )
    parts.append(
        f'<text x="10" y="{PAD_T + ph / 2:.0f}" text-anchor="middle" '
        f'font-size="10" fill="#64748b" transform="rotate(-90,10,{PAD_T + ph / 2:.0f})">LLR</text>'
    )

    inner = "\n  ".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="max-width:100%;display:block">\n  {inner}\n</svg>'
    )


# ---------------------------------------------------------------------------
# SVG: mini trend sparkline (for metrics table)
# ---------------------------------------------------------------------------

def _svg_sparkline(values: list[float], *, width: int = 80, height: int = 28) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return ""
    lo, hi = min(finite), max(finite)
    span = hi - lo or 1.0
    pad = 3
    pw, ph = width - 2 * pad, height - 2 * pad
    pts = " ".join(
        f"{pad + i * pw / (len(finite) - 1):.1f},{pad + ph - (v - lo) / span * ph:.1f}"
        for i, v in enumerate(finite)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="vertical-align:middle">'
        f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# Sample lookup helper
# ---------------------------------------------------------------------------

def _build_sample_lookup(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for s in samples:
        meta = s.get("metadata") or {}
        sid = str(meta.get("sample_id", "")).strip()
        if sid:
            lookup[sid] = {
                "test_id": s.get("test_id", ""),
                "prompt": s.get("prompt", ""),
                "seed": s.get("seed", ""),
            }
    return lookup


def _pair_key_label(pair_key: str, sample_lookup: dict[str, dict[str, Any]]) -> str:
    if pair_key.startswith("sid:"):
        sid = pair_key[4:]
        info = sample_lookup.get(sid)
        if info:
            prompt = str(info.get("prompt", ""))
            prompt_short = prompt[:72] + "…" if len(prompt) > 72 else prompt
            return f"{info.get('test_id')} / seed={info.get('seed')} / {escape(prompt_short)}"
        return escape(pair_key)
    if pair_key.startswith("idx:"):
        return escape(pair_key)
    # legacy key: "test_id|seed|prompt"
    return escape(pair_key[:100])


def _delta_cell(delta: float) -> str:
    cls = "delta-neg" if delta < -1e-9 else ("delta-pos" if delta > 1e-9 else "delta-zero")
    sign = "+" if delta > 1e-9 else ""
    return f'<span class="{cls}">{sign}{delta:.6f}</span>'


# ---------------------------------------------------------------------------
# Section: Metrics
# ---------------------------------------------------------------------------

def _render_metrics_section(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<p class="no-data">No metrics.</p>'

    rows: list[str] = []
    for name, payload in metrics.items():
        if not isinstance(payload, dict):
            rows.append(
                f"<tr><td><code>{escape(name)}</code></td>"
                f"<td>{escape(str(payload))}</td><td></td></tr>"
            )
            continue
        score = payload.get("score")
        score_str = f"{score:.6f}" if isinstance(score, float) else str(score) if score is not None else "—"
        dims = payload.get("dims")
        dims_str = ""
        if isinstance(dims, dict):
            dims_str = " &nbsp;".join(
                f'<span class="stat-pill"><strong>{escape(k)}:</strong>{v:.4f}</span>'
                for k, v in dims.items()
                if isinstance(v, float)
            )
        violations = payload.get("violations")
        if violations is not None:
            score_str = f"violations={violations}"
        rows.append(
            f"<tr><td><code>{escape(name)}</code></td>"
            f"<td><strong>{score_str}</strong></td>"
            f"<td>{dims_str}</td></tr>"
        )

    return (
        '<table><thead><tr><th>Metric</th><th>Score</th><th>Dimensions</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Section: Gate summary
# ---------------------------------------------------------------------------

def _render_gates_section(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return '<p class="no-data">No gates defined.</p>'

    rows: list[str] = []
    for gate in gates:
        status = gate.get("passed", False)
        tag = '<span class="tag-pass">PASS</span>' if status else '<span class="tag-fail">FAIL</span>'
        method = gate.get("method", "threshold")
        method_badge = (
            '<span class="tag-warn" style="font-size:0.72rem">SPRT</span> '
            if method == "sprt_regression"
            else ""
        )
        error = gate.get("error", "")
        error_cell = f'<span style="color:#b91c1c;font-size:0.8rem">{escape(error)}</span>' if error else ""
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(gate.get('metric', '')))}</code></td>"
            f"<td>{escape(str(gate.get('op', '')))}</td>"
            f"<td>{escape(str(gate.get('value', '')))}</td>"
            f"<td>{escape(str(gate.get('actual', '—')))}</td>"
            f"<td>{method_badge}{tag}</td>"
            f"<td>{error_cell}</td>"
            f"</tr>"
        )

    return (
        '<table><thead><tr>'
        '<th>Metric</th><th>Op</th><th>Target</th><th>Actual</th><th>Status</th><th>Error</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Section: SPRT analysis
# ---------------------------------------------------------------------------

def _render_worst_deltas_table(
    worst_deltas: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
) -> str:
    if not worst_deltas:
        return '<p class="no-data">No paired delta data.</p>'

    rows: list[str] = []
    for item in worst_deltas:
        key = str(item.get("pair_key", ""))
        label = _pair_key_label(key, sample_lookup)
        baseline = item.get("baseline")
        current = item.get("current")
        delta = item.get("delta")
        baseline_str = f"{baseline:.6f}" if isinstance(baseline, float) else str(baseline)
        current_str = f"{current:.6f}" if isinstance(current, float) else str(current)
        delta_cell = _delta_cell(float(delta)) if isinstance(delta, (int, float)) else str(delta)
        rows.append(
            f"<tr>"
            f'<td style="font-size:0.82rem;max-width:400px;word-break:break-word">{label}</td>'
            f"<td>{baseline_str}</td>"
            f"<td>{current_str}</td>"
            f"<td>{delta_cell}</td>"
            f"</tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>Sample (prompt / seed)</th><th>Baseline</th><th>Current</th><th>Delta</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _render_sprt_gate_card(
    gate: dict[str, Any],
    sample_lookup: dict[str, dict[str, Any]],
) -> str:
    sprt = gate.get("sprt")
    if not isinstance(sprt, dict):
        return ""

    metric = escape(str(gate.get("metric", "")))
    decision = sprt.get("decision", "")
    decision_passed = bool(sprt.get("decision_passed", False))
    status_tag = (
        '<span class="tag-pass">PASS</span>'
        if decision_passed
        else '<span class="tag-fail">FAIL</span>'
    )

    # Stats pills
    stats: list[tuple[str, str]] = []
    paired_count = None
    pairing = sprt.get("pairing")
    if isinstance(pairing, dict):
        paired_count = pairing.get("paired_count")
        expected = pairing.get("expected_pairs")
        ratio = pairing.get("paired_ratio")
        if paired_count is not None:
            stats.append(("paired", str(paired_count)))
        if expected is not None:
            stats.append(("expected", str(expected)))
        if ratio is not None:
            stats.append(("ratio", f"{float(ratio):.3f}"))
    else:
        paired_count = sprt.get("paired_count")
        if paired_count is not None:
            stats.append(("paired", str(paired_count)))

    if sprt.get("sigma") is not None:
        stats.append(("σ", f"{float(sprt['sigma']):.6f}"))
    if sprt.get("llr") is not None:
        stats.append(("LLR", f"{float(sprt['llr']):.4f}"))
    if sprt.get("upper_threshold") is not None:
        stats.append(("H1 threshold", f"{float(sprt['upper_threshold']):.4f}"))
    if sprt.get("lower_threshold") is not None:
        stats.append(("H0 threshold", f"{float(sprt['lower_threshold']):.4f}"))
    if sprt.get("crossed_at") is not None:
        stats.append(("crossed at", str(sprt["crossed_at"])))

    derived = derive_sprt_metrics(
        effect_size=sprt.get("effect_size"),
        sigma=sprt.get("sigma"),
        llr=sprt.get("llr"),
        paired_count=paired_count,
        upper_threshold=sprt.get("upper_threshold"),
        lower_threshold=sprt.get("lower_threshold"),
        alpha=sprt.get("alpha"),
        beta=sprt.get("beta"),
    )
    if derived.get("drift_per_pair") is not None:
        stats.append(("drift/pair", f"{derived['drift_per_pair']:.6f}"))
    if derived.get("required_pairs_upper") is not None:
        stats.append(("req pairs", f"{math.ceil(derived['required_pairs_upper'])}"))

    stats_html = "".join(
        f'<span class="stat-pill"><strong>{escape(k)}:</strong> {escape(v)}</span>'
        for k, v in stats
    )

    # LLR chart
    llr_history = sprt.get("llr_history")
    chart_html = ""
    if (
        isinstance(llr_history, list)
        and llr_history
        and sprt.get("upper_threshold") is not None
        and sprt.get("lower_threshold") is not None
    ):
        chart_html = (
            '<div class="chart-wrap">'
            + _svg_llr_chart(
                llr_history=llr_history,
                upper=float(sprt["upper_threshold"]),
                lower=float(sprt["lower_threshold"]),
                crossed_at=sprt.get("crossed_at"),
                min_pairs=int(sprt.get("min_pairs", 2)),
            )
            + "</div>"
        )

    # Worst deltas
    worst_deltas: list[dict[str, Any]] = []
    if isinstance(pairing, dict):
        raw_wd = pairing.get("worst_deltas")
        if isinstance(raw_wd, list):
            worst_deltas = raw_wd
    worst_html = ""
    if worst_deltas:
        worst_html = (
            "<h4>Worst Sample Pairs (by delta)</h4>"
            + _render_worst_deltas_table(worst_deltas, sample_lookup)
        )

    # Reason / policy note
    reason = sprt.get("reason", "")
    reason_html = ""
    if reason:
        reason_html = (
            f'<p style="font-size:0.82rem;color:#92400e;margin:8px 0 0">'
            f'⚠ reason: <strong>{escape(reason)}</strong>'
        )
        for policy_key in ("pairing_mismatch_policy", "baseline_missing_policy", "inconclusive_policy"):
            if sprt.get(policy_key):
                reason_html += f" &nbsp;·&nbsp; {escape(policy_key)}: <strong>{escape(str(sprt[policy_key]))}</strong>"
        reason_html += "</p>"

    return (
        f'<div class="sprt-card">'
        f'<div class="sprt-header">'
        f'<code class="sprt-metric">{metric}</code>'
        f"{status_tag}"
        f'<span style="font-size:0.88rem;color:#475569">{escape(decision)}</span>'
        f"</div>"
        f'<div class="stat-row">{stats_html}</div>'
        f"{chart_html}"
        f"{reason_html}"
        f"{worst_html}"
        f"</div>"
    )


def _render_sprt_section(
    gates: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
) -> str:
    sprt_gates = [g for g in gates if isinstance(g.get("sprt"), dict)]
    if not sprt_gates:
        return '<p class="no-data">No SPRT gates in this run.</p>'
    return "".join(_render_sprt_gate_card(g, sample_lookup) for g in sprt_gates)


# ---------------------------------------------------------------------------
# Section: Regression
# ---------------------------------------------------------------------------

def _render_regressions_section(regressions: list[dict[str, Any]]) -> str:
    if not regressions:
        return '<p class="no-data">No baseline available or no directional gates.</p>'

    rows: list[str] = []
    for item in regressions:
        regressed = item.get("regressed", False)
        tag = '<span class="tag-fail">REGRESSED</span>' if regressed else '<span class="tag-pass">OK</span>'
        delta = item.get("delta")
        delta_cell = _delta_cell(float(delta)) if isinstance(delta, (int, float)) else str(delta)
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(item.get('metric', '')))}</code></td>"
            f"<td>{escape(str(item.get('baseline', '—')))}</td>"
            f"<td>{escape(str(item.get('current', '—')))}</td>"
            f"<td>{delta_cell}</td>"
            f"<td>{escape(str(item.get('direction', '')))}</td>"
            f"<td>{tag}</td>"
            f"</tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Baseline</th><th>Current</th>"
        "<th>Delta</th><th>Direction</th><th>Status</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Section: Samples
# ---------------------------------------------------------------------------

def _render_samples_section(samples: list[dict[str, Any]], limit: int = 200) -> str:
    if not samples:
        return '<p class="no-data">No samples recorded.</p>'

    rows: list[str] = []
    for s in samples[:limit]:
        retained = s.get("artifact_retained", True)
        vid_path = s.get("video_path") or ""
        vid_cell = (
            f'<span style="font-size:0.78rem;color:#94a3b8">deleted / not retained</span>'
            if not retained
            else f'<code style="font-size:0.78rem">{escape(str(vid_path))}</code>'
        )
        prompt = str(s.get("prompt", ""))
        prompt_display = (prompt[:80] + "…") if len(prompt) > 80 else prompt
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(s.get('test_id', '')))}</code></td>"
            f"<td>{escape(str(s.get('seed', '')))}</td>"
            f'<td style="max-width:360px">{escape(prompt_display)}</td>'
            f"<td>{vid_cell}</td>"
            f"</tr>"
        )

    overflow = ""
    if len(samples) > limit:
        overflow = f'<p style="font-size:0.8rem;color:#94a3b8">… and {len(samples) - limit} more samples (truncated)</p>'

    return (
        "<table><thead><tr>"
        "<th>Test ID</th><th>Seed</th><th>Prompt</th><th>Video</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
        + overflow
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_html_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    status = str(payload.get("status", "UNKNOWN"))
    badge_cls = "badge-pass" if status == "PASS" else "badge-fail"

    gates = payload.get("gates") or []
    if not isinstance(gates, list):
        gates = []
    regressions = payload.get("regressions") or []
    if not isinstance(regressions, list):
        regressions = []
    samples = payload.get("samples") or []
    if not isinstance(samples, list):
        samples = []
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    sample_lookup = _build_sample_lookup(samples)

    def _meta(label: str, value: object) -> str:
        return (
            f'<div class="meta-item"><strong>{escape(label)}</strong>'
            f"{escape(str(value) if value is not None else '—')}</div>"
        )

    meta_html = "".join([
        _meta("Run ID", payload.get("run_id")),
        _meta("Project", payload.get("project")),
        _meta("Suite", payload.get("suite_name")),
        _meta("Model", payload.get("model_name")),
        _meta("Timestamp (UTC)", payload.get("timestamp_utc")),
        _meta("Baseline Run", payload.get("baseline_run_id")),
        _meta("Baseline Mode", payload.get("baseline_mode")),
        _meta("Samples", payload.get("sample_count")),
    ])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TemporalCI Report — {escape(status)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>TemporalCI Run Report</h1>
  <div class="badge {badge_cls}">{escape(status)}</div>

  <section>
    <div class="meta-grid">{meta_html}</div>
  </section>

  <section>
    <h2>Metrics</h2>
    {_render_metrics_section(metrics)}
  </section>

  <section>
    <h2>Gate Results</h2>
    {_render_gates_section(gates)}
  </section>

  <section>
    <h2>SPRT Analysis</h2>
    {_render_sprt_section(gates, sample_lookup)}
  </section>

  <section>
    <h2>Regression vs Baseline</h2>
    {_render_regressions_section(regressions)}
  </section>

  <section>
    <h2>Samples</h2>
    {_render_samples_section(samples)}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
