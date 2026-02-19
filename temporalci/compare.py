"""Side-by-side comparison of two TemporalCI run payloads.

Public API
----------
compare_runs(baseline, candidate)  ->  dict          (pure data, no I/O)
write_compare_report(path, baseline, candidate)       (writes HTML)
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

from temporalci.utils import resolve_dotted_path as _resolve_dotted

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_sample_info(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map sample_id → {test_id, prompt, seed} from the samples list."""
    info: dict[str, dict[str, Any]] = {}
    for s in payload.get("samples") or []:
        if not isinstance(s, dict):
            continue
        sid = str((s.get("metadata") or {}).get("sample_id", "")).strip()
        if sid:
            info[sid] = {
                "test_id": s.get("test_id", ""),
                "prompt": s.get("prompt", ""),
                "seed": s.get("seed", ""),
            }
    return info


def _per_sample_scores(metrics: dict[str, Any], metric_name: str) -> dict[str, float]:
    """Return sample_id → aggregate score for one metric's per_sample data."""
    metric_payload = metrics.get(metric_name)
    if not isinstance(metric_payload, dict):
        return {}
    per_sample = metric_payload.get("per_sample")
    if not isinstance(per_sample, list):
        return {}
    scores: dict[str, float] = {}
    for row in per_sample:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            continue
        score = row.get("score")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        ):
            scores[sid] = float(score)
        else:
            dims = row.get("dims")
            if isinstance(dims, dict):
                vals = [
                    float(v)
                    for v in dims.values()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(float(v))
                ]
                if vals:
                    scores[sid] = sum(vals) / len(vals)
    return scores


def _scalar_metric_paths(metrics: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    def _walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool) and path not in seen:
                seen.add(path)
                paths.append(path)
            elif isinstance(v, dict):
                _walk(v, path)

    _walk(metrics, "")
    return paths


# ---------------------------------------------------------------------------
# Comparison computations
# ---------------------------------------------------------------------------


def _compare_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare scalar aggregate metric values. Returns sorted by |delta| desc."""
    b_metrics = baseline.get("metrics") or {}
    c_metrics = candidate.get("metrics") or {}

    all_paths: list[str] = []
    seen: set[str] = set()
    for p in _scalar_metric_paths(b_metrics) + _scalar_metric_paths(c_metrics):
        if p not in seen:
            seen.add(p)
            all_paths.append(p)

    results: list[dict[str, Any]] = []
    for path in all_paths:
        b_val: float | None = None
        c_val: float | None = None
        try:
            raw = _resolve_dotted(b_metrics, path)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                b_val = float(raw)
        except (KeyError, TypeError):
            pass
        try:
            raw = _resolve_dotted(c_metrics, path)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                c_val = float(raw)
        except (KeyError, TypeError):
            pass
        delta = (c_val - b_val) if (b_val is not None and c_val is not None) else None
        results.append({"metric": path, "baseline": b_val, "candidate": c_val, "delta": delta})

    results.sort(key=lambda x: abs(x["delta"]) if x["delta"] is not None else 0.0, reverse=True)
    return results


def _compare_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    b_gate_map: dict[str, dict[str, Any]] = {
        str(g.get("metric", "")): g for g in (baseline.get("gates") or []) if isinstance(g, dict)
    }
    results: list[dict[str, Any]] = []
    for cg in candidate.get("gates") or []:
        if not isinstance(cg, dict):
            continue
        metric = str(cg.get("metric", ""))
        bg = b_gate_map.get(metric, {})
        b_passed = bg.get("passed")
        c_passed = cg.get("passed")
        if b_passed is True and c_passed is False:
            change = "regression"
        elif b_passed is False and c_passed is True:
            change = "improvement"
        elif b_passed is None:
            change = "new"
        else:
            change = "unchanged"
        results.append(
            {
                "metric": metric,
                "op": cg.get("op"),
                "value": cg.get("value"),
                "method": cg.get("method", "threshold"),
                "baseline_actual": bg.get("actual"),
                "candidate_actual": cg.get("actual"),
                "baseline_passed": b_passed,
                "candidate_passed": c_passed,
                "change": change,
                "sprt_baseline": bg.get("sprt"),
                "sprt_candidate": cg.get("sprt"),
            }
        )
    return results


def _compare_samples(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Per-sample delta across all shared metrics. Returns worst/best lists."""
    b_metrics = baseline.get("metrics") or {}
    c_metrics = candidate.get("metrics") or {}
    b_info = _build_sample_info(baseline)
    c_info = _build_sample_info(candidate)
    sample_info = {**b_info, **c_info}

    metric_names = set(b_metrics.keys()) & set(c_metrics.keys())
    agg_deltas: dict[str, list[float]] = {}
    per_metric: dict[str, dict[str, float]] = {}

    for mname in metric_names:
        b_scores = _per_sample_scores(b_metrics, mname)
        c_scores = _per_sample_scores(c_metrics, mname)
        for sid in set(b_scores) & set(c_scores):
            delta = c_scores[sid] - b_scores[sid]
            agg_deltas.setdefault(sid, []).append(delta)
            per_metric.setdefault(sid, {})[mname] = delta

    rows: list[dict[str, Any]] = []
    for sid, deltas in agg_deltas.items():
        avg = sum(deltas) / len(deltas)
        info = sample_info.get(sid, {})
        rows.append(
            {
                "sample_id": sid,
                "test_id": str(info.get("test_id", "")),
                "prompt": str(info.get("prompt", "")),
                "seed": info.get("seed", ""),
                "avg_delta": avg,
                "metric_deltas": per_metric.get(sid, {}),
            }
        )

    rows.sort(key=lambda x: x["avg_delta"])
    return {
        "worst": rows[:top_n],
        "best": list(reversed(rows[-top_n:])) if len(rows) > top_n else [],
        "total_matched": len(rows),
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compute a full side-by-side comparison. Returns a pure-data dict."""
    gate_changes = _compare_gates(baseline, candidate)
    return {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_status": baseline.get("status"),
        "candidate_status": candidate.get("status"),
        "baseline_timestamp": str(baseline.get("timestamp_utc", ""))[:19],
        "candidate_timestamp": str(candidate.get("timestamp_utc", ""))[:19],
        "project": candidate.get("project") or baseline.get("project"),
        "suite_name": candidate.get("suite_name") or baseline.get("suite_name"),
        "model_name": candidate.get("model_name") or baseline.get("model_name"),
        "gate_changes": gate_changes,
        "gate_regressions": [g for g in gate_changes if g["change"] == "regression"],
        "gate_improvements": [g for g in gate_changes if g["change"] == "improvement"],
        "metric_deltas": _compare_metrics(baseline, candidate),
        "sample_analysis": _compare_samples(baseline, candidate),
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; line-height: 1.5;
  }
  h1 { margin: 0 0 8px; font-size: 1.4rem; }
  h2 { font-size: 1.05rem; font-weight: 600; margin: 26px 0 10px;
       border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; }
  h3 { font-size: 0.92rem; font-weight: 600; margin: 14px 0 6px; color: #475569; }
  section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 20px; margin-bottom: 14px; }
  .subtitle { color: #64748b; font-size: 0.88rem; margin: 0 0 16px; }
  .run-pair { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .run-card {
    border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px;
    background: #f8fafc; min-width: 200px;
  }
  .run-card small { display: block; color: #94a3b8; font-size: 0.75rem; margin-bottom: 4px; }
  .run-card .rid { font-family: "SF Mono","Fira Code",monospace; font-size: 0.82rem; }
  .arrow { font-size: 1.4rem; color: #94a3b8; }
  .badge { display:inline-block; padding:3px 10px; border-radius:5px; font-weight:700;
           font-size:0.82rem; color:#fff; }
  .badge-pass { background:#15803d; }
  .badge-fail { background:#b91c1c; }
  .badge-unk  { background:#64748b; }
  .summary-pills { display:flex; gap:10px; flex-wrap:wrap; }
  .pill { border-radius:6px; padding:5px 12px; font-size:0.82rem; font-weight:600; }
  .pill-reg  { background:#fee2e2; color:#b91c1c; }
  .pill-imp  { background:#dcfce7; color:#15803d; }
  .pill-same { background:#f1f5f9; color:#64748b; }
  table { width:100%; border-collapse:collapse; font-size:0.84rem; }
  th { background:#f8fafc; font-weight:600; text-align:left;
       padding:7px 10px; border-bottom:2px solid #e2e8f0; }
  td { padding:6px 10px; border-bottom:1px solid #f1f5f9; vertical-align:top; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#fafafa; }
  .tag-reg  { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#fee2e2; color:#b91c1c; }
  .tag-imp  { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#dcfce7; color:#15803d; }
  .tag-same { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#f1f5f9; color:#64748b; }
  .tag-new  { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#fef9c3; color:#92400e; }
  .tag-pass { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#dcfce7; color:#15803d; }
  .tag-fail { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#fee2e2; color:#b91c1c; }
  .tag-na   { display:inline-block; padding:1px 7px; border-radius:4px;
              font-weight:600; font-size:0.76rem; background:#f1f5f9; color:#94a3b8; }
  .delta-neg { color:#b91c1c; font-weight:600; }
  .delta-pos { color:#15803d; font-weight:600; }
  .delta-zero { color:#94a3b8; }
  .prompt-cell { max-width:340px; font-size:0.8rem; word-break:break-word; }
  code { font-family:"SF Mono","Fira Code",monospace; font-size:0.83em;
         background:#f1f5f9; padding:1px 4px; border-radius:3px; }
  .no-data { color:#94a3b8; font-style:italic; font-size:0.88rem; }
"""


def _status_badge(status: str | None) -> str:
    s = str(status or "UNKNOWN")
    cls = "badge-pass" if s == "PASS" else ("badge-fail" if s == "FAIL" else "badge-unk")
    return f'<span class="badge {cls}">{escape(s)}</span>'


def _gate_tag(passed: bool | None) -> str:
    if passed is True:
        return '<span class="tag-pass">PASS</span>'
    if passed is False:
        return '<span class="tag-fail">FAIL</span>'
    return '<span class="tag-na">N/A</span>'


def _change_tag(change: str) -> str:
    if change == "regression":
        return '<span class="tag-reg">REGRESSED</span>'
    if change == "improvement":
        return '<span class="tag-imp">IMPROVED</span>'
    if change == "new":
        return '<span class="tag-new">NEW</span>'
    return '<span class="tag-same">unchanged</span>'


def _delta_cell(delta: float | None) -> str:
    if delta is None:
        return '<span class="delta-zero">—</span>'
    if delta < -1e-9:
        return f'<span class="delta-neg">{delta:+.6f}</span>'
    if delta > 1e-9:
        return f'<span class="delta-pos">{delta:+.6f}</span>'
    return '<span class="delta-zero">±0</span>'


def _render_gate_changes(gate_changes: list[dict[str, Any]]) -> str:
    if not gate_changes:
        return '<p class="no-data">No gates.</p>'
    rows: list[str] = []
    for g in gate_changes:
        b_act = g.get("baseline_actual")
        c_act = g.get("candidate_actual")
        b_str = (
            f"{b_act:.6f}" if isinstance(b_act, float) else str(b_act) if b_act is not None else "—"
        )
        c_str = (
            f"{c_act:.6f}" if isinstance(c_act, float) else str(c_act) if c_act is not None else "—"
        )
        method = g.get("method", "threshold")
        sprt_badge = (
            ' <span style="font-size:0.72rem;background:#fef9c3;color:#92400e;padding:1px 5px;border-radius:3px">SPRT</span>'
            if method == "sprt_regression"
            else ""
        )
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(g.get('metric', '')))}</code>{sprt_badge}</td>"
            f"<td>{escape(str(g.get('op', '')))}</td>"
            f"<td>{escape(str(g.get('value', '')))}</td>"
            f"<td>{b_str} {_gate_tag(g.get('baseline_passed'))}</td>"
            f"<td>{c_str} {_gate_tag(g.get('candidate_passed'))}</td>"
            f"<td>{_change_tag(g.get('change', 'unchanged'))}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Op</th><th>Target</th>"
        "<th>Baseline</th><th>Candidate</th><th>Change</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_sprt_comparison(gate_changes: list[dict[str, Any]]) -> str:
    sprt_gates = [g for g in gate_changes if g.get("method") == "sprt_regression"]
    if not sprt_gates:
        return '<p class="no-data">No SPRT gates.</p>'

    cards: list[str] = []
    for g in sprt_gates:
        metric = escape(str(g.get("metric", "")))
        sb = g.get("sprt_baseline") or {}
        sc = g.get("sprt_candidate") or {}

        def _sprt_row(label: str, key: str) -> str:
            bv = sb.get(key)
            cv = sc.get(key)
            bvs = f"{bv:.6f}" if isinstance(bv, float) else str(bv) if bv is not None else "—"
            cvs = f"{cv:.6f}" if isinstance(cv, float) else str(cv) if cv is not None else "—"
            return f"<tr><td><strong>{escape(label)}</strong></td><td>{bvs}</td><td>{cvs}</td></tr>"

        pairs_b = (sb.get("pairing") or {}).get("paired_count") or sb.get("paired_count")
        pairs_c = (sc.get("pairing") or {}).get("paired_count") or sc.get("paired_count")

        cards.append(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-bottom:10px">'
            f"<h3><code>{metric}</code></h3>"
            f"<table><thead><tr><th></th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>"
            f"<tr><td><strong>decision</strong></td>"
            f"<td>{escape(str(sb.get('decision', '—')))}</td>"
            f"<td>{escape(str(sc.get('decision', '—')))}</td></tr>"
            f"<tr><td><strong>paired</strong></td>"
            f"<td>{pairs_b if pairs_b is not None else '—'}</td>"
            f"<td>{pairs_c if pairs_c is not None else '—'}</td></tr>"
            + _sprt_row("LLR", "llr")
            + _sprt_row("sigma", "sigma")
            + _sprt_row("upper threshold", "upper_threshold")
            + _sprt_row("lower threshold", "lower_threshold")
            + "</tbody></table></div>"
        )
    return "".join(cards)


def _render_metric_deltas(deltas: list[dict[str, Any]]) -> str:
    if not deltas:
        return '<p class="no-data">No shared metrics.</p>'
    rows: list[str] = []
    for d in deltas:
        b = d.get("baseline")
        c = d.get("candidate")
        bstr = f"{b:.6f}" if isinstance(b, float) else "—"
        cstr = f"{c:.6f}" if isinstance(c, float) else "—"
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(d['metric']))}</code></td>"
            f"<td>{bstr}</td>"
            f"<td>{cstr}</td>"
            f"<td>{_delta_cell(d.get('delta'))}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Baseline</th><th>Candidate</th><th>Δ</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_sample_table(samples: list[dict[str, Any]], label: str) -> str:
    if not samples:
        return ""
    rows: list[str] = []
    for s in samples:
        prompt = str(s.get("prompt", ""))
        prompt_short = (prompt[:75] + "…") if len(prompt) > 75 else prompt
        deltas_html = " ".join(
            f'<span style="font-size:0.78rem"><code>{escape(mname)}</code> {_delta_cell(v)}</span>'
            for mname, v in (s.get("metric_deltas") or {}).items()
        )
        rows.append(
            f"<tr>"
            f"<td><code>{escape(str(s.get('test_id', '')))}</code></td>"
            f"<td>{escape(str(s.get('seed', '')))}</td>"
            f'<td class="prompt-cell">{escape(prompt_short)}</td>'
            f"<td>{_delta_cell(s.get('avg_delta'))}</td>"
            f"<td>{deltas_html}</td>"
            f"</tr>"
        )
    return (
        f"<h3>{escape(label)}</h3>"
        "<table><thead><tr>"
        "<th>Test ID</th><th>Seed</th><th>Prompt</th><th>Avg Δ</th><th>Per-metric Δ</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_sample_analysis(analysis: dict[str, Any]) -> str:
    total = analysis.get("total_matched", 0)
    if total == 0:
        return '<p class="no-data">No matched samples (sample_id pairing required).</p>'
    worst = analysis.get("worst") or []
    best = analysis.get("best") or []
    out = f'<p style="font-size:0.82rem;color:#64748b;margin:0 0 10px">{total} samples matched across runs.</p>'
    out += _render_sample_table(worst, f"Worst degraded (top {len(worst)})")
    if best:
        out += _render_sample_table(best, f"Most improved (top {len(best)})")
    return out


def format_compare_text(cmp: dict[str, Any]) -> str:
    """Return a concise terminal-friendly summary of a compare_runs result."""
    lines: list[str] = []
    b_id = str(cmp.get("baseline_run_id") or "—")
    c_id = str(cmp.get("candidate_run_id") or "—")
    b_status = str(cmp.get("baseline_status") or "UNKNOWN")
    c_status = str(cmp.get("candidate_status") or "UNKNOWN")
    lines.append(f"baseline : {b_id}  [{b_status}]")
    lines.append(f"candidate: {c_id}  [{c_status}]")
    lines.append("")

    gate_changes = cmp.get("gate_changes") or []
    if gate_changes:
        lines.append("gates:")
        for g in gate_changes:
            metric = str(g.get("metric", ""))
            op = str(g.get("op", ""))
            target = str(g.get("value", ""))
            b_act = g.get("baseline_actual")
            c_act = g.get("candidate_actual")
            change = str(g.get("change", "unchanged"))
            b_str = (
                f"{b_act:.6f}"
                if isinstance(b_act, float)
                else (str(b_act) if b_act is not None else "—")
            )
            c_str = (
                f"{c_act:.6f}"
                if isinstance(c_act, float)
                else (str(c_act) if c_act is not None else "—")
            )
            tag = {"regression": "REGRESSED", "improvement": "IMPROVED", "new": "NEW"}.get(
                change, "ok"
            )
            lines.append(f"  {metric} {op} {target}  {b_str} → {c_str}  [{tag}]")
        lines.append("")

    deltas = cmp.get("metric_deltas") or []
    top_deltas = [d for d in deltas if d.get("delta") is not None][:8]
    if top_deltas:
        lines.append("metric deltas (top by |Δ|):")
        w = max(len(str(d["metric"])) for d in top_deltas)
        for d in top_deltas:
            b = d.get("baseline")
            c = d.get("candidate")
            delta = d.get("delta")
            b_str = f"{b:.6f}" if isinstance(b, float) else "—"
            c_str = f"{c:.6f}" if isinstance(c, float) else "—"
            sign = "+" if delta > 0 else ""
            lines.append(f"  {str(d['metric']):<{w}}  {b_str} → {c_str}  ({sign}{delta:.6f})")

    return "\n".join(lines)


def write_compare_report(
    path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Write a side-by-side comparison HTML report. Returns the comparison data dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmp = compare_runs(baseline, candidate)

    b_status = str(cmp["baseline_status"] or "UNKNOWN")
    c_status = str(cmp["candidate_status"] or "UNKNOWN")

    gate_regressions = cmp["gate_regressions"]
    gate_improvements = cmp["gate_improvements"]
    n_unchanged = len(cmp["gate_changes"]) - len(gate_regressions) - len(gate_improvements)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TemporalCI Compare — {escape(b_status)} → {escape(c_status)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>TemporalCI Run Comparison</h1>
  <p class="subtitle">
    {escape(str(cmp.get("project", "")))} / {escape(str(cmp.get("suite_name", "")))} /
    {escape(str(cmp.get("model_name", "")))}
  </p>

  <section>
    <div class="run-pair">
      <div class="run-card">
        <small>BASELINE</small>
        <div class="rid">{escape(str(cmp["baseline_run_id"] or "—"))}</div>
        <div style="margin-top:4px">
          {_status_badge(cmp["baseline_status"])}
          <span style="font-size:0.78rem;color:#94a3b8;margin-left:6px">{escape(cmp["baseline_timestamp"])}</span>
        </div>
      </div>
      <div class="arrow">→</div>
      <div class="run-card">
        <small>CANDIDATE</small>
        <div class="rid">{escape(str(cmp["candidate_run_id"] or "—"))}</div>
        <div style="margin-top:4px">
          {_status_badge(cmp["candidate_status"])}
          <span style="font-size:0.78rem;color:#94a3b8;margin-left:6px">{escape(cmp["candidate_timestamp"])}</span>
        </div>
      </div>
    </div>

    <div class="summary-pills">
      <span class="pill pill-reg">{len(gate_regressions)} gate regression{"s" if len(gate_regressions) != 1 else ""}</span>
      <span class="pill pill-imp">{len(gate_improvements)} gate improvement{"s" if len(gate_improvements) != 1 else ""}</span>
      <span class="pill pill-same">{n_unchanged} unchanged</span>
      <span class="pill pill-same">{cmp["sample_analysis"].get("total_matched", 0)} samples matched</span>
    </div>
  </section>

  <section>
    <h2>Gate Changes</h2>
    {_render_gate_changes(cmp["gate_changes"])}
  </section>

  <section>
    <h2>Metric Deltas</h2>
    {_render_metric_deltas(cmp["metric_deltas"])}
  </section>

  <section>
    <h2>SPRT Comparison</h2>
    {_render_sprt_comparison(cmp["gate_changes"])}
  </section>

  <section>
    <h2>Per-sample Analysis</h2>
    {_render_sample_analysis(cmp["sample_analysis"])}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return cmp
