"""Generate interactive route risk map using Leaflet.js."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_RISK_COLORS = {
    "critical": "#ef4444",
    "warning": "#fb923c",
    "caution": "#facc15",
    "safe": "#22c55e",
}

_LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
_LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"


def generate_route_map(
    results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Catenary Inspection Route Map",
    default_center: tuple[float, float] = (35.68, 139.77),
    default_zoom: int = 14,
) -> Path:
    """Generate an interactive HTML route map from inspection results.

    Each result dict should have:
    - prompt: frame identifier
    - risk_level: "critical" | "warning" | "caution" | "safe"
    - risk_score: float 0-1
    - lat: float (optional, from sidecar metadata)
    - lon: float (optional, from sidecar metadata)
    - km: float (optional, km marker)
    - vegetation_zone: float (optional)
    - clearance_px: float (optional)

    Results without lat/lon are skipped.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter results with GPS data
    geo_results = [r for r in results if r.get("lat") is not None and r.get("lon") is not None]

    if not geo_results:
        # Generate a placeholder map with a message
        markers_js = "[]"
        center = list(default_center)
        zoom = default_zoom
    else:
        markers_data = []
        for r in geo_results:
            markers_data.append(
                {
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "prompt": r.get("prompt", "unknown"),
                    "risk_level": r.get("risk_level", "safe"),
                    "risk_score": round(r.get("risk_score", 0), 2),
                    "km": r.get("km"),
                    "vegetation_zone": round(r.get("vegetation_zone", 0), 4),
                    "clearance_px": round(r.get("clearance_px", 0), 1),
                }
            )
        markers_js = json.dumps(markers_data, indent=2)
        lats = [r["lat"] for r in geo_results]
        lons = [r["lon"] for r in geo_results]
        center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
        zoom = default_zoom

    risk_colors_js = json.dumps(_RISK_COLORS)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{_LEAFLET_CSS}">
<script src="{_LEAFLET_JS}"></script>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  #map {{ width: 100%; height: 100vh; }}
  .legend {{
    background: white; padding: 10px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); line-height: 1.8;
    font-size: 13px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .legend-dot {{
    width: 14px; height: 14px; border-radius: 50%;
    display: inline-block; border: 1px solid #666;
  }}
  .popup-table {{ border-collapse: collapse; font-size: 12px; }}
  .popup-table td {{ padding: 2px 8px; }}
  .popup-table td:first-child {{ font-weight: bold; color: #555; }}
  .stats-bar {{
    position: fixed; top: 10px; left: 60px; z-index: 1000;
    background: rgba(0,0,0,0.8); color: white; padding: 8px 16px;
    border-radius: 8px; font-size: 13px;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div class="stats-bar" id="stats"></div>
<script>
const markers = {markers_js};
const riskColors = {risk_colors_js};
const center = {json.dumps(center)};

const map = L.map("map").setView(center, {zoom});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 19,
}}).addTo(map);

// Legend
const legend = L.control({{position: "bottomright"}});
legend.onAdd = function() {{
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML = "<b>Risk Level</b><br>" +
    Object.entries(riskColors).map(([level, color]) =>
      `<div class="legend-item"><span class="legend-dot" style="background:${{color}}"></span>${{level}}</div>`
    ).join("");
  return div;
}};
legend.addTo(map);

// Markers
const layerGroup = L.layerGroup().addTo(map);
const latLngs = [];

markers.forEach(m => {{
  const color = riskColors[m.risk_level] || "#888";
  const circle = L.circleMarker([m.lat, m.lon], {{
    radius: 8, fillColor: color, color: "#333", weight: 1,
    fillOpacity: 0.85,
  }}).addTo(layerGroup);

  let popup = `<table class="popup-table">
    <tr><td>Frame</td><td>${{m.prompt}}</td></tr>
    <tr><td>Risk</td><td style="color:${{color}};font-weight:bold">${{m.risk_level.toUpperCase()}}</td></tr>
    <tr><td>Score</td><td>${{m.risk_score}}</td></tr>`;
  if (m.km !== null) popup += `<tr><td>km</td><td>${{m.km}}</td></tr>`;
  popup += `<tr><td>Veg zone</td><td>${{(m.vegetation_zone * 100).toFixed(1)}}%</td></tr>
    <tr><td>Clearance</td><td>${{m.clearance_px}}px</td></tr>
  </table>`;
  circle.bindPopup(popup);
  latLngs.push([m.lat, m.lon]);
}});

// Polyline connecting markers
if (latLngs.length > 1) {{
  L.polyline(latLngs, {{color: "#3b82f6", weight: 3, opacity: 0.6, dashArray: "8,6"}}).addTo(map);
  map.fitBounds(latLngs, {{padding: [40, 40]}});
}}

// Stats bar
const total = markers.length;
const critical = markers.filter(m => m.risk_level === "critical").length;
const warning = markers.filter(m => m.risk_level === "warning").length;
const safe = markers.filter(m => m.risk_level === "safe" || m.risk_level === "caution").length;
document.getElementById("stats").innerHTML =
  `<b>{html.escape(title)}</b> &mdash; ${{total}} points | ` +
  `<span style="color:#ef4444">${{critical}} critical</span> | ` +
  `<span style="color:#fb923c">${{warning}} warning</span> | ` +
  `<span style="color:#22c55e">${{safe}} safe/caution</span>`;
</script>
</body>
</html>'''

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
