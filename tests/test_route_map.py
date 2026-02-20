"""Tests for the route map generator."""

from __future__ import annotations

from pathlib import Path

from temporalci.route_map import generate_route_map


class TestRouteMap:
    def test_generates_html(self, tmp_path: Path) -> None:
        results = [
            {
                "prompt": "frame_001",
                "risk_level": "critical",
                "risk_score": 0.2,
                "lat": 35.681,
                "lon": 139.767,
                "vegetation_zone": 0.5,
                "clearance_px": 12.0,
            },
            {
                "prompt": "frame_002",
                "risk_level": "safe",
                "risk_score": 0.9,
                "lat": 35.682,
                "lon": 139.768,
                "vegetation_zone": 0.1,
                "clearance_px": 85.0,
            },
        ]
        out = tmp_path / "map.html"
        result_path = generate_route_map(results, out)
        assert result_path == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "leaflet" in content.lower()
        assert "frame_001" in content
        assert "35.681" in content

    def test_no_gps_data(self, tmp_path: Path) -> None:
        results = [
            {"prompt": "f1", "risk_level": "safe", "risk_score": 0.8},
        ]
        out = tmp_path / "map.html"
        generate_route_map(results, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "[]" in content  # empty markers

    def test_mixed_gps_data(self, tmp_path: Path) -> None:
        results = [
            {"prompt": "f1", "risk_level": "safe", "risk_score": 0.8, "lat": 35.68, "lon": 139.77},
            {"prompt": "f2", "risk_level": "warning", "risk_score": 0.4},
        ]
        out = tmp_path / "map.html"
        generate_route_map(results, out)
        content = out.read_text(encoding="utf-8")
        # Only f1 should be in markers (has GPS)
        assert "f1" in content

    def test_custom_title(self, tmp_path: Path) -> None:
        out = tmp_path / "map.html"
        generate_route_map(
            [
                {
                    "prompt": "f1",
                    "lat": 35.68,
                    "lon": 139.77,
                    "risk_level": "safe",
                    "risk_score": 0.9,
                }
            ],
            out,
            title="JR East Line 23",
        )
        content = out.read_text(encoding="utf-8")
        assert "JR East Line 23" in content

    def test_polyline_with_multiple_points(self, tmp_path: Path) -> None:
        results = [
            {
                "prompt": f"f{i}",
                "lat": 35.68 + i * 0.001,
                "lon": 139.77 + i * 0.001,
                "risk_level": "safe",
                "risk_score": 0.8,
            }
            for i in range(5)
        ]
        out = tmp_path / "map.html"
        generate_route_map(results, out)
        content = out.read_text(encoding="utf-8")
        assert "L.polyline" in content

    def test_empty_results(self, tmp_path: Path) -> None:
        out = tmp_path / "map.html"
        generate_route_map([], out)
        assert out.exists()
