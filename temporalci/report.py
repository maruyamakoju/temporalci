from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _render_metrics(metrics: dict[str, Any]) -> str:
    rows: list[str] = []
    for metric_name, payload in metrics.items():
        payload_dict = payload if isinstance(payload, dict) else {"value": payload}
        score = payload_dict.get("score", "")
        if score == "" and "violations" in payload_dict:
            score = f"violations={payload_dict.get('violations')}"
        rows.append(
            "<tr>"
            f"<td>{escape(str(metric_name))}</td>"
            f"<td><pre>{escape(str(score))}</pre></td>"
            f"<td><pre>{escape(str(payload_dict))}</pre></td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_gates(gates: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for gate in gates:
        status = "PASS" if gate.get("passed") else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{escape(str(gate.get('metric')))}</td>"
            f"<td>{escape(str(gate.get('op')))}</td>"
            f"<td>{escape(str(gate.get('value')))}</td>"
            f"<td>{escape(str(gate.get('actual')))}</td>"
            f"<td>{escape(status)}</td>"
            f"<td>{escape(str(gate.get('error', '')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_sprt_gates(gates: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for gate in gates:
        sprt = gate.get("sprt")
        if not isinstance(sprt, dict):
            continue
        pairing = sprt.get("pairing")
        pairing_dict = pairing if isinstance(pairing, dict) else {}
        paired_count = pairing_dict.get("paired_count", sprt.get("paired_count", ""))
        paired_ratio = pairing_dict.get("paired_ratio", sprt.get("paired_ratio", ""))
        worst_deltas = pairing_dict.get("worst_deltas")

        details: dict[str, Any] = {}
        for key in (
            "reason",
            "pairing_mismatch_policy",
            "baseline_missing_policy",
            "require_baseline",
            "inconclusive_policy",
            "sigma_mode",
            "sigma",
            "effect_size",
            "min_pairs",
            "min_paired_ratio",
        ):
            value = sprt.get(key)
            if value is None or value == "":
                continue
            details[key] = value
        if isinstance(worst_deltas, list) and worst_deltas:
            details["worst_deltas"] = worst_deltas
        for key in (
            "pairing",
            "expected_pairs",
            "current_series_count",
            "baseline_series_count",
            "current_missing_sample_id_count",
            "baseline_missing_sample_id_count",
        ):
            if key in pairing_dict:
                details[f"pairing.{key}"] = pairing_dict[key]

        status = "PASS" if sprt.get("decision_passed") else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{escape(str(gate.get('metric')))}</td>"
            f"<td>{escape(str(sprt.get('decision', '')))}</td>"
            f"<td>{escape(status)}</td>"
            f"<td>{escape(str(paired_count))}</td>"
            f"<td>{escape(str(paired_ratio))}</td>"
            f"<td>{escape(str(sprt.get('llr', '')))}</td>"
            f"<td>{escape(str(sprt.get('upper_threshold', '')))}</td>"
            f"<td>{escape(str(sprt.get('lower_threshold', '')))}</td>"
            f"<td>{escape(str(sprt.get('crossed_at', '')))}</td>"
            f"<td><pre>{escape(str(details))}</pre></td>"
            "</tr>"
        )
    if not rows:
        return "<tr><td colspan='10'>No SPRT gates.</td></tr>"
    return "\n".join(rows)


def _render_regressions(regressions: list[dict[str, Any]]) -> str:
    if not regressions:
        return "<tr><td colspan='6'>No baseline or no directional gates.</td></tr>"
    rows: list[str] = []
    for item in regressions:
        status = "REGRESSED" if item.get("regressed") else "OK"
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('metric')))}</td>"
            f"<td>{escape(str(item.get('baseline')))}</td>"
            f"<td>{escape(str(item.get('current')))}</td>"
            f"<td>{escape(str(item.get('delta')))}</td>"
            f"<td>{escape(str(item.get('direction')))}</td>"
            f"<td>{escape(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_samples(samples: list[dict[str, Any]], limit: int = 200) -> str:
    rows: list[str] = []
    for sample in samples[:limit]:
        rows.append(
            "<tr>"
            f"<td>{escape(str(sample.get('test_id')))}</td>"
            f"<td>{escape(str(sample.get('seed')))}</td>"
            f"<td>{escape(str(sample.get('prompt')))}</td>"
            f"<td>{escape(str(sample.get('video_path')))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4'>No samples.</td></tr>")
    return "\n".join(rows)


def write_html_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = payload.get("status", "UNKNOWN")
    status_color = "#14804a" if status == "PASS" else "#a52222"

    metrics = payload.get("metrics", {})
    metrics_render_input = metrics if isinstance(metrics, dict) else {}
    gates = payload.get("gates", [])
    gates_render_input = gates if isinstance(gates, list) else []
    regressions = payload.get("regressions", [])
    regressions_render_input = regressions if isinstance(regressions, list) else []
    samples = payload.get("samples", [])
    samples_render_input = samples if isinstance(samples, list) else []

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TemporalCI Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.4; }}
    .status {{ display: inline-block; padding: 8px 12px; border-radius: 6px; color: white; background: {status_color}; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f6f6f6; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fff; }}
  </style>
</head>
<body>
  <h1>TemporalCI Run Report</h1>
  <div class="status">{escape(str(status))}</div>
  <div class="grid">
    <div class="card"><strong>Run ID:</strong> {escape(str(payload.get("run_id")))}</div>
    <div class="card"><strong>Project:</strong> {escape(str(payload.get("project")))}</div>
    <div class="card"><strong>Suite:</strong> {escape(str(payload.get("suite_name")))}</div>
    <div class="card"><strong>Model:</strong> {escape(str(payload.get("model_name")))}</div>
    <div class="card"><strong>Timestamp:</strong> {escape(str(payload.get("timestamp_utc")))}</div>
    <div class="card"><strong>Baseline:</strong> {escape(str(payload.get("baseline_run_id")))}</div>
    <div class="card"><strong>Baseline Mode:</strong> {escape(str(payload.get("baseline_mode")))}</div>
    <div class="card"><strong>Artifacts Policy:</strong> {escape(str(payload.get("artifacts_policy")))}</div>
  </div>

  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Primary</th><th>Payload</th></tr></thead>
    <tbody>{_render_metrics(metrics_render_input)}</tbody>
  </table>

  <h2>Gates</h2>
  <table>
    <thead><tr><th>Metric</th><th>Op</th><th>Target</th><th>Actual</th><th>Status</th><th>Error</th></tr></thead>
    <tbody>{_render_gates(gates_render_input)}</tbody>
  </table>

  <h2>SPRT Analysis</h2>
  <table>
    <thead><tr><th>Metric</th><th>Decision</th><th>Status</th><th>Paired</th><th>Paired Ratio</th><th>LLR</th><th>Upper</th><th>Lower</th><th>Crossed At</th><th>Details</th></tr></thead>
    <tbody>{_render_sprt_gates(gates_render_input)}</tbody>
  </table>

  <h2>Regression vs Baseline</h2>
  <table>
    <thead><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Direction</th><th>Status</th></tr></thead>
    <tbody>{_render_regressions(regressions_render_input)}</tbody>
  </table>

  <h2>Samples</h2>
  <table>
    <thead><tr><th>Test ID</th><th>Seed</th><th>Prompt</th><th>Video Path</th></tr></thead>
    <tbody>{_render_samples(samples_render_input)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
