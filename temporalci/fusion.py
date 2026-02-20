"""Multi-camera fusion engine for combining inspection results.

Railway inspection cars typically have 6 cameras (front, back, left, right,
up, down).  Each camera produces per-frame results with risk scores,
vegetation metrics, clearance dims, etc.  This module combines those into
unified per-location assessments and supports km-based aggregation for
maintenance prioritization.

Public API
----------
fuse_cameras(results_by_camera, weights)  -> dict
aggregate_by_km(per_sample, bin_size_km)  -> list[dict]
generate_km_report(km_bins, output_path)  -> Path
prioritize_maintenance(km_bins, budget_km) -> list[dict]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraWeight:
    """Per-camera weight factors for different risk types.

    All weights are in [0, 1]; they do **not** need to sum to 1 across cameras
    -- normalisation is applied at fusion time.
    """

    position: str
    vegetation_weight: float = 0.0
    equipment_weight: float = 0.0
    visibility_weight: float = 0.0


@dataclass
class FusionResult:
    """Per-location fused result combining all available camera angles."""

    location_id: str
    fused_risk_score: float
    fused_vegetation_score: float = 0.0
    fused_equipment_score: float = 0.0
    fused_visibility_score: float = 0.0
    camera_count: int = 0
    camera_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_level: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default camera weights
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_WEIGHTS: dict[str, CameraWeight] = {
    "up": CameraWeight("up", vegetation_weight=0.3, equipment_weight=0.4, visibility_weight=0.1),
    "left": CameraWeight(
        "left", vegetation_weight=0.25, equipment_weight=0.15, visibility_weight=0.2
    ),
    "right": CameraWeight(
        "right", vegetation_weight=0.25, equipment_weight=0.15, visibility_weight=0.2
    ),
    "front": CameraWeight(
        "front", vegetation_weight=0.1, equipment_weight=0.15, visibility_weight=0.2
    ),
    "back": CameraWeight(
        "back", vegetation_weight=0.05, equipment_weight=0.1, visibility_weight=0.2
    ),
    "down": CameraWeight(
        "down", vegetation_weight=0.05, equipment_weight=0.05, visibility_weight=0.1
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_risk(score: float) -> str:
    """Translate a 0-1 risk score into a human-readable level.

    Higher score = safer (consistent with the rest of TemporalCI).
    """
    if score >= 0.8:
        return "safe"
    if score >= 0.6:
        return "caution"
    if score >= 0.4:
        return "warning"
    return "critical"


def _weighted_average(
    values: list[float],
    weights: list[float],
) -> float:
    """Compute a weighted average, falling back to simple mean on zero weight."""
    total_w = sum(weights)
    if total_w == 0:
        return mean(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce *val* to float, returning *default* on failure."""
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return default


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------


def fuse_cameras(
    results_by_camera: dict[str, dict[str, Any]],
    weights: dict[str, CameraWeight] | None = None,
) -> dict[str, Any]:
    """Fuse per-camera metric results into a single unified assessment.

    Parameters
    ----------
    results_by_camera:
        Mapping of camera position name (e.g. ``"up"``, ``"front"``) to a
        metric evaluation result dict.  Each result should contain at least
        ``risk_score`` (float 0-1).  Optional keys: ``vegetation_proximity_nn``,
        ``vegetation_penetration``, ``equipment_score``, ``visibility_score``,
        ``clearance_px``, ``dims`` (sub-dict).
    weights:
        Camera weight overrides.  Defaults to :data:`DEFAULT_CAMERA_WEIGHTS`.

    Returns
    -------
    dict with:
    - ``risk_score`` (float): weighted-average risk
    - ``vegetation_score`` (float): weighted-average vegetation metric
    - ``equipment_score`` (float): weighted-average equipment metric
    - ``visibility_score`` (float): weighted-average visibility metric
    - ``risk_level`` (str): classified risk level
    - ``camera_count`` (int)
    - ``cameras`` (dict): per-camera breakdown
    """
    if weights is None:
        weights = DEFAULT_CAMERA_WEIGHTS

    if not results_by_camera:
        return {
            "risk_score": 0.0,
            "vegetation_score": 0.0,
            "equipment_score": 0.0,
            "visibility_score": 0.0,
            "risk_level": "critical",
            "camera_count": 0,
            "cameras": {},
        }

    risk_vals: list[float] = []
    risk_weights: list[float] = []

    veg_vals: list[float] = []
    veg_weights: list[float] = []

    equip_vals: list[float] = []
    equip_weights: list[float] = []

    vis_vals: list[float] = []
    vis_weights: list[float] = []

    cameras: dict[str, dict[str, Any]] = {}

    for cam_name, result in results_by_camera.items():
        cw = weights.get(cam_name, CameraWeight(cam_name))

        dims = result.get("dims", {})
        risk = _safe_float(result.get("risk_score", dims.get("risk_score")), default=0.5)
        veg = _safe_float(
            result.get(
                "vegetation_proximity_nn",
                dims.get("vegetation_proximity_nn"),
            ),
            default=0.0,
        )
        equip = _safe_float(
            result.get("equipment_score", dims.get("equipment_score")),
            default=0.0,
        )
        vis = _safe_float(
            result.get("visibility_score", dims.get("visibility_score")),
            default=0.0,
        )

        # Compute an average risk weight across all types for the overall score
        avg_w = (cw.vegetation_weight + cw.equipment_weight + cw.visibility_weight) / 3.0
        risk_vals.append(risk)
        risk_weights.append(avg_w)

        veg_vals.append(veg)
        veg_weights.append(cw.vegetation_weight)

        equip_vals.append(equip)
        equip_weights.append(cw.equipment_weight)

        vis_vals.append(vis)
        vis_weights.append(cw.visibility_weight)

        cameras[cam_name] = {
            "risk_score": round(risk, 4),
            "vegetation_score": round(veg, 4),
            "equipment_score": round(equip, 4),
            "visibility_score": round(vis, 4),
            "weight": {
                "vegetation": cw.vegetation_weight,
                "equipment": cw.equipment_weight,
                "visibility": cw.visibility_weight,
            },
        }

    fused_risk = _weighted_average(risk_vals, risk_weights)
    fused_veg = _weighted_average(veg_vals, veg_weights)
    fused_equip = _weighted_average(equip_vals, equip_weights)
    fused_vis = _weighted_average(vis_vals, vis_weights)

    return {
        "risk_score": round(fused_risk, 4),
        "vegetation_score": round(fused_veg, 4),
        "equipment_score": round(fused_equip, 4),
        "visibility_score": round(fused_vis, 4),
        "risk_level": _classify_risk(fused_risk),
        "camera_count": len(results_by_camera),
        "cameras": cameras,
    }


# ---------------------------------------------------------------------------
# Km-based aggregation
# ---------------------------------------------------------------------------


def aggregate_by_km(
    per_sample: list[dict[str, Any]],
    bin_size_km: float = 0.5,
) -> list[dict[str, Any]]:
    """Group per-frame results by km marker and aggregate risk.

    Each entry in *per_sample* should contain:
    - ``km`` (float): distance marker in kilometres
    - ``risk_score`` (float): 0-1 score (higher = safer)
    - ``prompt`` (str, optional): frame identifier
    - ``risk_level`` (str, optional)

    Any sample missing a valid ``km`` value is silently skipped.

    Returns a list of km-bin dicts sorted by km_start, each containing:
    - ``km_start``, ``km_end``: bin boundaries
    - ``avg_risk``, ``min_risk``, ``max_risk``
    - ``worst_frame``: prompt of the frame with the lowest risk
    - ``frame_count``: number of frames in the bin
    - ``frames``: list of contributing frame summaries
    """
    if bin_size_km <= 0:
        bin_size_km = 0.5

    # Collect samples with valid km
    valid: list[tuple[float, dict[str, Any]]] = []
    for sample in per_sample:
        km_raw = sample.get("km")
        if km_raw is None:
            continue
        km = _safe_float(km_raw, default=float("nan"))
        if not math.isfinite(km):
            continue
        valid.append((km, sample))

    if not valid:
        return []

    # Assign to bins
    bins: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for km, sample in valid:
        bin_idx = int(km / bin_size_km)
        bins.setdefault(bin_idx, []).append((km, sample))

    result: list[dict[str, Any]] = []
    for bin_idx in sorted(bins):
        entries = bins[bin_idx]
        km_start = round(bin_idx * bin_size_km, 4)
        km_end = round((bin_idx + 1) * bin_size_km, 4)

        risks = [_safe_float(s.get("risk_score"), 0.5) for _, s in entries]
        worst_idx = risks.index(min(risks))
        worst_sample = entries[worst_idx][1]

        frames: list[dict[str, Any]] = []
        for km_val, s in entries:
            frames.append(
                {
                    "prompt": str(s.get("prompt", "")),
                    "km": round(km_val, 4),
                    "risk_score": round(_safe_float(s.get("risk_score"), 0.5), 4),
                    "risk_level": s.get(
                        "risk_level", _classify_risk(_safe_float(s.get("risk_score"), 0.5))
                    ),
                }
            )

        result.append(
            {
                "km_start": km_start,
                "km_end": km_end,
                "avg_risk": round(mean(risks), 4),
                "min_risk": round(min(risks), 4),
                "max_risk": round(max(risks), 4),
                "worst_frame": str(worst_sample.get("prompt", "")),
                "frame_count": len(entries),
                "frames": frames,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Maintenance prioritization
# ---------------------------------------------------------------------------


def _urgency_label(risk: float) -> str:
    """Map a risk score to an urgency label (inverse of risk)."""
    if risk < 0.4:
        return "critical"
    if risk < 0.6:
        return "high"
    if risk < 0.8:
        return "medium"
    return "low"


def prioritize_maintenance(
    km_bins: list[dict[str, Any]],
    budget_km: float = 5.0,
) -> list[dict[str, Any]]:
    """Select highest-priority track segments within a maintenance budget.

    Parameters
    ----------
    km_bins:
        Output from :func:`aggregate_by_km`.
    budget_km:
        Total km of track that can be serviced in this maintenance window.

    Returns
    -------
    List of selected bins (subset of *km_bins*) sorted by ascending risk
    (worst first), each augmented with:
    - ``urgency``: "critical" / "high" / "medium" / "low"
    - ``cumulative_km``: running total of km covered
    """
    if not km_bins:
        return []

    # Sort by avg_risk ascending (worst segments first)
    ranked = sorted(km_bins, key=lambda b: b.get("avg_risk", 1.0))

    selected: list[dict[str, Any]] = []
    cumulative = 0.0

    for km_bin in ranked:
        bin_len = km_bin.get("km_end", 0.0) - km_bin.get("km_start", 0.0)
        if bin_len <= 0:
            continue
        if cumulative + bin_len > budget_km:
            break
        cumulative += bin_len
        entry = dict(km_bin)
        entry["urgency"] = _urgency_label(km_bin.get("avg_risk", 1.0))
        entry["cumulative_km"] = round(cumulative, 4)
        selected.append(entry)

    return selected


# ---------------------------------------------------------------------------
# HTML km report
# ---------------------------------------------------------------------------

_KM_REPORT_CSS = """
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
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }
  .stat-card {
    border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px;
    background: #f8fafc; text-align: center;
  }
  .stat-value { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
  .stat-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th { background: #f8fafc; font-weight: 600; text-align: left;
       padding: 7px 10px; border-bottom: 2px solid #e2e8f0; }
  td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafafa; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-weight: 600; font-size: 0.76rem; }
  .tag-critical { background: #fee2e2; color: #b91c1c; }
  .tag-high     { background: #fff7ed; color: #c2410c; }
  .tag-medium   { background: #fef9c3; color: #92400e; }
  .tag-low      { background: #dcfce7; color: #15803d; }
  .tag-safe     { background: #dcfce7; color: #15803d; }
  .tag-caution  { background: #fef9c3; color: #92400e; }
  .tag-warning  { background: #fff7ed; color: #c2410c; }
  code { font-family: "SF Mono","Fira Code",monospace; font-size: 0.85em;
         background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
  .heatmap-bar {
    display: flex; height: 28px; border-radius: 6px; overflow: hidden;
    margin-bottom: 8px;
  }
  .heatmap-seg {
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 600; color: #fff; min-width: 2px;
  }
  .detail-card {
    border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;
    margin-bottom: 10px; background: #f8fafc;
  }
  .detail-card h3 { margin: 0 0 8px; font-size: 0.92rem; }
  .detail-card .frame-list { font-size: 0.82rem; color: #475569; }
"""


def _risk_color(score: float) -> str:
    """Map a risk score to a CSS colour."""
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.6:
        return "#facc15"
    if score >= 0.4:
        return "#fb923c"
    return "#ef4444"


def _urgency_tag(label: str) -> str:
    """Render an urgency tag span."""
    cls_map = {
        "critical": "tag-critical",
        "high": "tag-high",
        "medium": "tag-medium",
        "low": "tag-low",
    }
    cls = cls_map.get(label, "tag-low")
    return f'<span class="tag {cls}">{escape(label.upper())}</span>'


def generate_km_report(
    km_bins: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Km-based Inspection Report",
) -> Path:
    """Generate an HTML report with risk heatmap, priority table, and detail cards.

    Parameters
    ----------
    km_bins:
        Output from :func:`aggregate_by_km`.
    output_path:
        File path to write the HTML report to.
    title:
        Report heading.

    Returns
    -------
    The resolved output :class:`Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not km_bins:
        output_path.write_text(
            f"<!doctype html><html><head><title>{escape(title)}</title></head>"
            f"<body><h1>{escape(title)}</h1><p>No km data available.</p></body></html>",
            encoding="utf-8",
        )
        return output_path

    # Summary statistics
    all_risks = [b["avg_risk"] for b in km_bins]
    total_km = sum(b["km_end"] - b["km_start"] for b in km_bins)
    total_frames = sum(b["frame_count"] for b in km_bins)
    overall_avg_risk = round(mean(all_risks), 4) if all_risks else 0.0
    critical_count = sum(1 for r in all_risks if r < 0.4)
    warning_count = sum(1 for r in all_risks if 0.4 <= r < 0.6)

    # Heatmap bar
    heatmap_segments = ""
    for km_bin in km_bins:
        width_pct = (km_bin["km_end"] - km_bin["km_start"]) / total_km * 100 if total_km > 0 else 0
        color = _risk_color(km_bin["avg_risk"])
        label = f"{km_bin['km_start']:.1f}"
        heatmap_segments += (
            f'<div class="heatmap-seg" style="width:{width_pct:.1f}%;background:{color}"'
            f' title="km {km_bin["km_start"]:.1f}-{km_bin["km_end"]:.1f}: '
            f'risk={km_bin["avg_risk"]:.2f}">{label}</div>'
        )

    # Priority table (sorted by avg_risk ascending = worst first)
    priority_sorted = sorted(km_bins, key=lambda b: b.get("avg_risk", 1.0))
    priority_rows = ""
    for km_bin in priority_sorted:
        urgency = _urgency_label(km_bin["avg_risk"])
        priority_rows += (
            f"<tr>"
            f"<td>{km_bin['km_start']:.1f} - {km_bin['km_end']:.1f}</td>"
            f"<td>{km_bin['avg_risk']:.4f}</td>"
            f"<td>{km_bin['min_risk']:.4f}</td>"
            f"<td>{_urgency_tag(urgency)}</td>"
            f"<td><code>{escape(km_bin['worst_frame'])}</code></td>"
            f"<td>{km_bin['frame_count']}</td>"
            f"</tr>"
        )

    # Detail cards
    detail_cards = ""
    for km_bin in km_bins:
        frame_items = ""
        for frm in km_bin.get("frames", []):
            level = frm.get("risk_level", "unknown")
            level_cls = {
                "critical": "tag-critical",
                "warning": "tag-warning",
                "caution": "tag-caution",
                "safe": "tag-safe",
            }.get(level, "tag-low")
            frame_items += (
                f'<div style="margin:2px 0">'
                f"<code>{escape(frm['prompt'])}</code> "
                f"km={frm['km']:.2f} "
                f"risk={frm['risk_score']:.4f} "
                f'<span class="tag {level_cls}">{escape(level.upper())}</span>'
                f"</div>"
            )
        detail_cards += (
            f'<div class="detail-card">'
            f"<h3>km {km_bin['km_start']:.1f} &ndash; {km_bin['km_end']:.1f}</h3>"
            f'<div style="font-size:0.84rem;margin-bottom:6px">'
            f"avg_risk={km_bin['avg_risk']:.4f} &middot; "
            f"frames={km_bin['frame_count']}</div>"
            f'<div class="frame-list">{frame_items}</div>'
            f"</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>{_KM_REPORT_CSS}</style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="subtitle">
    {len(km_bins)} km segments &middot; {total_frames} frames &middot;
    {total_km:.1f} km total
  </p>

  <section>
    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="stat-card">
        <div class="stat-value">{len(km_bins)}</div>
        <div class="stat-label">segments</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{total_frames}</div>
        <div class="stat-label">frames</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{overall_avg_risk:.4f}</div>
        <div class="stat-label">avg risk</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:#b91c1c">{critical_count}</div>
        <div class="stat-label">critical bins</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:#c2410c">{warning_count}</div>
        <div class="stat-label">warning bins</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Risk Heatmap</h2>
    <div class="heatmap-bar">{heatmap_segments}</div>
    <div style="font-size:0.78rem;color:#94a3b8;display:flex;justify-content:space-between">
      <span>{km_bins[0]["km_start"]:.1f} km</span>
      <span>{km_bins[-1]["km_end"]:.1f} km</span>
    </div>
  </section>

  <section>
    <h2>Priority Maintenance Table</h2>
    <table>
      <thead><tr>
        <th>Km Range</th><th>Avg Risk</th><th>Min Risk</th>
        <th>Urgency</th><th>Worst Frame</th><th>Frames</th>
      </tr></thead>
      <tbody>{priority_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Per-bin Details</h2>
    {detail_cards}
  </section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
