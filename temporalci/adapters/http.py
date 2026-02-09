from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from temporalci.adapters.base import ModelAdapter
from temporalci.types import GeneratedSample


class HttpAdapter(ModelAdapter):
    """
    Minimal remote inference adapter.

    Expected endpoint request body:
      {
        "model": "...",
        "prompt": "...",
        "seed": 0,
        "video_cfg": {...},
        "params": {...}
      }

    Expected response body (JSON):
      {
        "video_path": "/path/to/video.mp4",        # optional
        "evaluation_stream": [0.1, 0.2, ...],      # optional
        "metadata": {...}                           # optional
      }
    """

    def __init__(self, model_name: str, params: dict[str, object] | None = None) -> None:
        super().__init__(model_name, params)
        endpoint = str(self.params.get("endpoint", "")).strip()
        if not endpoint:
            raise ValueError("http adapter requires 'endpoint' in model params")
        self.endpoint = endpoint
        self.timeout_sec = float(self.params.get("timeout_sec", 120.0))

    def generate(
        self,
        *,
        test_id: str,
        prompt: str,
        seed: int,
        video_cfg: dict[str, object],
        output_dir: Path,
    ) -> GeneratedSample:
        output_dir.mkdir(parents=True, exist_ok=True)

        body = {
            "model": self.model_name,
            "prompt": prompt,
            "seed": seed,
            "video_cfg": video_cfg,
            "params": self.params,
        }
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            text = response.read().decode("utf-8")
        payload = json.loads(text)

        token = hashlib.sha1(f"{test_id}|{prompt}|{seed}".encode("utf-8")).hexdigest()[:12]
        response_copy = output_dir / f"http_response_{token}.json"
        response_copy.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        stream_raw = payload.get("evaluation_stream")
        if isinstance(stream_raw, list):
            stream = [float(x) for x in stream_raw]
        else:
            stream = self._fallback_stream(prompt=prompt, seed=seed, video_cfg=video_cfg)

        video_path = payload.get("video_path")
        if not isinstance(video_path, str) or not video_path.strip():
            video_path = str(response_copy)

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["adapter"] = "http"
        metadata["endpoint"] = self.endpoint

        return GeneratedSample(
            test_id=test_id,
            prompt=prompt,
            seed=seed,
            video_path=video_path,
            evaluation_stream=stream,
            metadata=metadata,
        )

    def _fallback_stream(
        self,
        *,
        prompt: str,
        seed: int,
        video_cfg: dict[str, object],
    ) -> list[float]:
        num_frames = int(video_cfg.get("num_frames", 25))
        fingerprint = hashlib.sha256(
            f"{prompt}|{seed}|{self.model_name}|fallback".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(fingerprint[:8], 16))
        return [round(0.5 + (rng.random() - 0.5) * 0.2, 6) for _ in range(num_frames)]

