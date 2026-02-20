"""Tests for the FastAPI server module."""

from __future__ import annotations


import pytest

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


class TestPipelineState:
    def test_initial_state(self) -> None:
        from temporalci.server import PipelineState

        state = PipelineState()
        assert state.status == "idle"
        assert state.progress == 0
        assert state.total_frames == 0

    def test_to_status_dict(self) -> None:
        from temporalci.server import PipelineState

        state = PipelineState()
        state.status = "running"
        state.progress = 5
        state.total_frames = 10
        d = state.to_status_dict()
        assert d["status"] == "running"
        assert d["progress"] == 5
        assert d["total_frames"] == 10

    def test_subscribe_unsubscribe(self) -> None:
        from temporalci.server import PipelineState

        state = PipelineState()
        q = state.subscribe()
        assert q in state._subscribers
        state.unsubscribe(q)
        assert q not in state._subscribers


class TestCreateApp:
    def test_creates_app(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4")
        assert app is not None

    def test_dashboard_route(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4")
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TemporalCI" in resp.text

    def test_status_route(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4")
        client = TestClient(app)
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_results_route(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4")
        client = TestClient(app)
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_frame_not_found(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4", output_dir="/tmp/nonexistent_serve")
        client = TestClient(app)
        resp = client.get("/api/frames/0")
        assert resp.status_code == 404

    def test_start_no_video(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="")
        client = TestClient(app)
        resp = client.post("/api/start")
        assert resp.status_code == 400

    def test_stop_when_idle(self) -> None:
        from temporalci.server import create_app

        app = create_app(video_path="test.mp4")
        client = TestClient(app)
        resp = client.post("/api/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


class TestDashboardHtml:
    def test_html_structure(self) -> None:
        from temporalci.server import _DASHBOARD_HTML

        assert "<!doctype html>" in _DASHBOARD_HTML
        assert "TemporalCI" in _DASHBOARD_HTML
        assert "WebSocket" in _DASHBOARD_HTML
        assert "/ws" in _DASHBOARD_HTML
        assert "/api/start" in _DASHBOARD_HTML

    def test_has_risk_table(self) -> None:
        from temporalci.server import _DASHBOARD_HTML

        assert "frameTable" in _DASHBOARD_HTML
        assert "Score" in _DASHBOARD_HTML

    def test_has_controls(self) -> None:
        from temporalci.server import _DASHBOARD_HTML

        assert "startBtn" in _DASHBOARD_HTML
        assert "stopBtn" in _DASHBOARD_HTML
