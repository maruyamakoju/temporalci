"""FastAPI server for real-time catenary inspection dashboard.

Provides a live web dashboard that processes video frames and streams
analysis results via WebSocket.

Usage::

    # From CLI
    temporalci serve --video jr23_720p.mp4 --port 8421

    # Programmatic
    from temporalci.server import create_app, run_server
    app = create_app(video_path="jr23_720p.mp4")
    run_server(app, port=8421)

Endpoints
---------
GET  /                  Live dashboard (HTML)
GET  /api/status        Pipeline status
GET  /api/results       Current results snapshot
GET  /api/frames/{idx}  Serve extracted frame JPEG
WS   /ws                WebSocket stream of per-frame results
POST /api/start         Start/restart analysis
POST /api/stop          Stop running analysis
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class PipelineState:
    """Mutable server state shared across requests."""

    def __init__(self) -> None:
        self.video_path: str = ""
        self.fps: float = 1.0
        self.max_frames: int = 0
        self.device: str = "auto"
        self.skip_depth: bool = False
        self.output_dir: str = "serve_output"
        self.title: str = "Live Catenary Inspection"

        self.status: str = "idle"  # idle / running / done / error
        self.progress: int = 0
        self.total_frames: int = 0
        self.current_frame: str = ""
        self.results: dict[str, Any] = {}
        self.per_frame: list[dict[str, Any]] = []
        self.error: str = ""
        self.elapsed_s: float = 0.0

        self._task: asyncio.Task[None] | None = None
        self._subscribers: list[asyncio.Queue[str]] = []

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
        }

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all WebSocket subscribers."""
        text = json.dumps(message, default=str)
        dead: list[asyncio.Queue[str]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(text)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


# ---------------------------------------------------------------------------
# Analysis runner (runs in thread)
# ---------------------------------------------------------------------------


def _run_analysis_sync(state: PipelineState) -> None:
    """Synchronous analysis — called from a thread via asyncio."""
    from temporalci.vision.video import extract_frames

    video_path = Path(state.video_path)
    output_dir = Path(state.output_dir)
    frames_dir = output_dir / "frames"

    # Extract frames
    frame_paths = extract_frames(
        video_path,
        frames_dir,
        fps=state.fps,
        max_frames=state.max_frames or None,
    )

    state.total_frames = len(frame_paths)
    if not frame_paths:
        state.error = "No frames extracted"
        state.status = "error"
        return

    # Build samples
    from temporalci.metrics.catenary_clearance import evaluate
    from temporalci.types import GeneratedSample

    samples = [
        GeneratedSample(
            test_id="inspect",
            prompt=f.stem,
            seed=0,
            video_path=str(f),
            evaluation_stream=[],
        )
        for f in frame_paths
    ]

    params: dict[str, Any] = {
        "device": state.device,
        "skip_depth": str(state.skip_depth).lower(),
        "output_dir": str(output_dir / "panels"),
    }

    state.results = evaluate(samples, params=params)
    state.per_frame = state.results.get("per_sample", [])
    state.status = "done"


async def _run_analysis_async(state: PipelineState) -> None:
    """Async wrapper that runs analysis in a thread and broadcasts progress."""
    try:
        state.status = "running"
        state.progress = 0
        state.error = ""
        t0 = time.time()

        await state.broadcast(
            {
                "type": "status",
                "status": "running",
                "message": "Extracting frames...",
            }
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_analysis_sync, state)

        state.elapsed_s = time.time() - t0

        if state.status == "done":
            await state.broadcast(
                {
                    "type": "status",
                    "status": "done",
                    "total_frames": state.total_frames,
                    "elapsed_s": round(state.elapsed_s, 1),
                }
            )
            # Send all frame results
            for i, frame in enumerate(state.per_frame):
                await state.broadcast(
                    {
                        "type": "frame",
                        "index": i,
                        "total": state.total_frames,
                        "data": frame,
                    }
                )
            await state.broadcast(
                {
                    "type": "results",
                    "data": {
                        "score": state.results.get("score", 0),
                        "dims": state.results.get("dims", {}),
                        "sample_count": state.results.get("sample_count", 0),
                        "alert_count": len(state.results.get("alert_frames", [])),
                    },
                }
            )
        else:
            await state.broadcast(
                {
                    "type": "error",
                    "message": state.error,
                }
            )
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        await state.broadcast({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TemporalCI — Live Inspection</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; background: #0f172a; color: #e2e8f0; line-height: 1.5;
  }
  .header {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px 24px; border-bottom: 1px solid #334155;
    display: flex; align-items: center; gap: 16px;
  }
  .header h1 { margin: 0; font-size: 1.3rem; color: #f8fafc; }
  .header .badge {
    padding: 3px 10px; border-radius: 12px; font-size: 0.75rem;
    font-weight: 600; text-transform: uppercase;
  }
  .badge-idle { background: #334155; color: #94a3b8; }
  .badge-running { background: #1d4ed8; color: #dbeafe; animation: pulse 1.5s infinite; }
  .badge-done { background: #15803d; color: #dcfce7; }
  .badge-error { background: #b91c1c; color: #fee2e2; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }
  .content { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .controls { display: flex; gap: 10px; margin-bottom: 16px; }
  .btn {
    padding: 8px 20px; border: none; border-radius: 8px; cursor: pointer;
    font-weight: 600; font-size: 0.85rem; transition: opacity .15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn-primary { background: #3b82f6; color: white; }
  .btn-danger { background: #ef4444; color: white; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  section {
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 20px; margin-bottom: 14px;
  }
  h2 {
    font-size: 0.95rem; font-weight: 600; margin: 0 0 14px;
    color: #f1f5f9; border-bottom: 1px solid #334155; padding-bottom: 6px;
  }
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
  }
  .stat-card {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 14px; text-align: center;
  }
  .stat-value { font-size: 1.5rem; font-weight: 700; }
  .stat-label { font-size: 0.72rem; color: #64748b; margin-top: 2px; }
  #log {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 12px; font-family: "SF Mono","Fira Code",monospace;
    font-size: 0.78rem; max-height: 300px; overflow-y: auto;
    color: #94a3b8; white-space: pre-wrap;
  }
  .log-entry { margin: 2px 0; }
  .log-frame { color: #93c5fd; }
  .log-error { color: #fca5a5; }
  .log-done { color: #86efac; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th {
    background: #0f172a; font-weight: 600; text-align: left;
    padding: 6px 8px; border-bottom: 2px solid #334155; color: #94a3b8;
  }
  td { padding: 5px 8px; border-bottom: 1px solid #1e293b; color: #cbd5e1; }
  tr:hover td { background: #334155; }
  .risk-critical { color: #ef4444; font-weight: 600; }
  .risk-warning { color: #fb923c; font-weight: 600; }
  .risk-caution { color: #facc15; font-weight: 600; }
  .risk-safe { color: #22c55e; font-weight: 600; }
  code {
    font-family: "SF Mono","Fira Code",monospace; font-size: 0.85em;
    background: #0f172a; padding: 2px 5px; border-radius: 3px; color: #93c5fd;
  }
  .progress-bar {
    width: 100%; height: 6px; background: #334155; border-radius: 3px;
    margin: 8px 0; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: #3b82f6; border-radius: 3px;
    transition: width .3s ease;
  }
</style>
</head>
<body>
  <div class="header">
    <h1>TemporalCI — Live Inspection</h1>
    <span class="badge badge-idle" id="badge">IDLE</span>
  </div>
  <div class="content">
    <div class="controls">
      <button class="btn btn-primary" id="startBtn" onclick="startAnalysis()">Start Analysis</button>
      <button class="btn btn-danger" id="stopBtn" onclick="stopAnalysis()" disabled>Stop</button>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%"></div></div>

    <section>
      <h2>Overview</h2>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value" id="score">—</div>
          <div class="stat-label">composite score</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="frames">0</div>
          <div class="stat-label">frames</div>
        </div>
        <div class="stat-card">
          <div class="stat-value risk-critical" id="critical">0</div>
          <div class="stat-label">critical</div>
        </div>
        <div class="stat-card">
          <div class="stat-value risk-warning" id="warnings">0</div>
          <div class="stat-label">warning</div>
        </div>
        <div class="stat-card">
          <div class="stat-value risk-safe" id="safe">0</div>
          <div class="stat-label">safe</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="elapsed">—</div>
          <div class="stat-label">elapsed</div>
        </div>
      </div>
    </section>

    <section>
      <h2>Per-frame Results</h2>
      <table>
        <thead><tr>
          <th>#</th><th>Frame</th><th>Risk</th><th>Score</th>
          <th>Veg Prox</th><th>Clearance</th><th>Wires</th>
        </tr></thead>
        <tbody id="frameTable"></tbody>
      </table>
    </section>

    <section>
      <h2>Log</h2>
      <div id="log"></div>
    </section>
  </div>

<script>
const RISK_CLASSES = {critical:'risk-critical', warning:'risk-warning', caution:'risk-caution', safe:'risk-safe'};
let ws = null;
let frameCount = 0;
const dist = {critical:0, warning:0, caution:0, safe:0};

function log(msg, cls='') {
  const el = document.getElementById('log');
  const div = document.createElement('div');
  div.className = 'log-entry ' + cls;
  div.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function setBadge(status) {
  const b = document.getElementById('badge');
  b.textContent = status.toUpperCase();
  b.className = 'badge badge-' + status;
}

function startAnalysis() {
  fetch('/api/start', {method:'POST'}).then(r => r.json()).then(d => {
    log('Analysis started');
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    frameCount = 0;
    Object.keys(dist).forEach(k => dist[k] = 0);
    document.getElementById('frameTable').innerHTML = '';
    connectWS();
  });
}

function stopAnalysis() {
  fetch('/api/stop', {method:'POST'}).then(r => r.json()).then(d => {
    log('Analysis stopped');
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
  });
}

function connectWS() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');

  ws.onmessage = function(e) {
    const msg = JSON.parse(e.data);

    if (msg.type === 'status') {
      setBadge(msg.status);
      log(msg.message || msg.status, msg.status === 'done' ? 'log-done' : '');
      if (msg.status === 'done') {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('elapsed').textContent = msg.elapsed_s + 's';
        document.getElementById('frames').textContent = msg.total_frames;
      }
    }

    if (msg.type === 'frame') {
      frameCount++;
      const d = msg.data;
      const dims = d.dims || {};
      const level = d.risk_level || 'unknown';
      const cls = RISK_CLASSES[level] || '';

      dist[level] = (dist[level] || 0) + 1;
      document.getElementById('critical').textContent = dist.critical || 0;
      document.getElementById('warnings').textContent = dist.warning || 0;
      document.getElementById('safe').textContent = (dist.safe||0) + (dist.caution||0);

      const pct = msg.total > 0 ? (msg.index + 1) / msg.total * 100 : 0;
      document.getElementById('progress').style.width = pct + '%';

      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${msg.index+1}</td>` +
        `<td><code>${d.prompt || ''}</code></td>` +
        `<td class="${cls}">${(level||'').toUpperCase()}</td>` +
        `<td>${(dims.risk_score||0).toFixed(4)}</td>` +
        `<td>${(dims.vegetation_proximity_nn||0).toFixed(4)}</td>` +
        `<td>${(d.clearance_px||0).toFixed(1)}px</td>` +
        `<td>${d.wire_count||0}</td>`;
      document.getElementById('frameTable').appendChild(tr);

      log(`Frame ${msg.index+1}/${msg.total}: ${d.prompt} [${level}] score=${(dims.risk_score||0).toFixed(4)}`, 'log-frame');
    }

    if (msg.type === 'results') {
      const r = msg.data;
      const color = r.score >= 0.7 ? '#22c55e' : (r.score >= 0.4 ? '#fb923c' : '#ef4444');
      const el = document.getElementById('score');
      el.textContent = r.score.toFixed(4);
      el.style.color = color;
    }

    if (msg.type === 'error') {
      log('ERROR: ' + msg.message, 'log-error');
      setBadge('error');
      document.getElementById('startBtn').disabled = false;
      document.getElementById('stopBtn').disabled = true;
    }
  };

  ws.onclose = function() { log('WebSocket disconnected'); };
}

// Auto-connect WS
connectWS();
setBadge('idle');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    video_path: str = "",
    fps: float = 1.0,
    max_frames: int = 0,
    device: str = "auto",
    skip_depth: bool = False,
    output_dir: str = "serve_output",
    title: str = "Live Catenary Inspection",
) -> Any:
    """Create a FastAPI application for live inspection."""
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the serve command. "
            "Install with: pip install 'fastapi[standard]'"
        )

    app = FastAPI(title="TemporalCI Live Inspection")
    state = PipelineState()
    state.video_path = video_path
    state.fps = fps
    state.max_frames = max_frames
    state.device = device
    state.skip_depth = skip_depth
    state.output_dir = output_dir
    state.title = title

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(state.to_status_dict())

    @app.get("/api/results")
    async def api_results() -> JSONResponse:
        return JSONResponse(
            {
                "status": state.status,
                "results": state.results,
            }
        )

    @app.get("/api/frames/{idx}", response_model=None)
    async def api_frame(idx: int) -> FileResponse | JSONResponse:
        frames_dir = Path(state.output_dir) / "frames"
        frame_path = frames_dir / f"frame_{idx:05d}.jpg"
        if frame_path.is_file():
            return FileResponse(str(frame_path), media_type="image/jpeg")
        return JSONResponse({"error": "frame not found"}, status_code=404)

    @app.post("/api/start")
    async def api_start() -> JSONResponse:
        if state.status == "running":
            return JSONResponse({"error": "already running"}, status_code=409)
        if not state.video_path:
            return JSONResponse({"error": "no video configured"}, status_code=400)
        state._task = asyncio.create_task(_run_analysis_async(state))
        return JSONResponse({"status": "started"})

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        if state._task and not state._task.done():
            state._task.cancel()
            state.status = "idle"
        return JSONResponse({"status": "stopped"})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = state.subscribe()
        try:
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)
        except WebSocketDisconnect:
            pass
        finally:
            state.unsubscribe(queue)

    return app


def run_server(
    app: Any,
    host: str = "0.0.0.0",
    port: int = 8421,
) -> None:
    """Run the FastAPI server using uvicorn."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install uvicorn") from None
    uvicorn.run(app, host=host, port=port)
