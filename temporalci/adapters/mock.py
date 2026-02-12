from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from temporalci.adapters.base import ModelAdapter
from temporalci.types import GeneratedSample
from temporalci.utils import clamp


class MockAdapter(ModelAdapter):
    """Deterministic synthetic generator for CI plumbing."""

    def generate(
        self,
        *,
        test_id: str,
        prompt: str,
        seed: int,
        video_cfg: dict[str, Any],
        output_dir: Path,
    ) -> GeneratedSample:
        output_dir.mkdir(parents=True, exist_ok=True)
        num_frames = int(video_cfg.get("num_frames", 25))
        quality_shift = float(self.params.get("quality_shift", 0.0))
        noise_scale = float(self.params.get("noise_scale", 0.06))

        stream = self._build_stream(
            prompt=prompt,
            seed=seed,
            num_frames=num_frames,
            quality_shift=quality_shift,
            noise_scale=noise_scale,
        )

        sample_hash = hashlib.sha1(
            f"{test_id}|{prompt}|{seed}|{self.model_name}".encode("utf-8")
        ).hexdigest()[:12]
        video_path = output_dir / f"{sample_hash}.json"
        payload = {
            "test_id": test_id,
            "model": self.model_name,
            "prompt": prompt,
            "seed": seed,
            "num_frames": num_frames,
            "evaluation_stream": stream,
        }
        video_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return GeneratedSample(
            test_id=test_id,
            prompt=prompt,
            seed=seed,
            video_path=str(video_path),
            evaluation_stream=stream,
            metadata={"adapter": "mock"},
        )

    def _build_stream(
        self,
        *,
        prompt: str,
        seed: int,
        num_frames: int,
        quality_shift: float,
        noise_scale: float,
    ) -> list[float]:
        fingerprint = hashlib.sha256(
            f"{prompt}|{seed}|{self.model_name}".encode("utf-8")
        ).hexdigest()
        rng_seed = int(fingerprint[:8], 16)
        rng = random.Random(rng_seed)

        freq = 0.1 + rng.random() * 0.4
        phase = rng.random() * (math.pi * 2.0)
        motion_amp = 0.18 + rng.random() * 0.2

        # Negative quality_shift increases noise; positive shift stabilizes.
        noise_multiplier = 1.0 + max(-quality_shift, 0.0) * 2.0
        drift = quality_shift * 0.05
        effective_noise = noise_scale * noise_multiplier

        stream: list[float] = []
        for frame_idx in range(num_frames):
            smooth_signal = 0.5 + motion_amp * math.sin(frame_idx * freq + phase)
            noise = rng.gauss(0.0, effective_noise)
            value = clamp(smooth_signal + drift + noise)
            stream.append(round(value, 6))
        return stream
