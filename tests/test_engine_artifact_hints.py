from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from temporalci.adapters import register_adapter
from temporalci.adapters.base import ModelAdapter
from temporalci.config import load_suite
from temporalci.engine import run_suite
from temporalci.metrics import register_metric
from temporalci.types import GeneratedSample


def test_artifact_hints_are_forwarded_to_adapter_and_metric(tmp_path: Path) -> None:
    captured_video_cfg: dict[str, Any] = {}
    captured_metric_params: dict[str, Any] = {}

    class CaptureAdapter(ModelAdapter):
        def generate(
            self,
            *,
            test_id: str,
            prompt: str,
            seed: int,
            video_cfg: dict[str, Any],
            output_dir: Path,
        ) -> GeneratedSample:
            captured_video_cfg.update(video_cfg)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "sample.json"
            video_path.write_text("{}", encoding="utf-8")
            return GeneratedSample(
                test_id=test_id,
                prompt=prompt,
                seed=seed,
                video_path=str(video_path),
                evaluation_stream=[0.5, 0.5, 0.5],
            )

    def capture_metric(
        samples: list[GeneratedSample],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured_metric_params.update(params or {})
        return {"score": 0.9, "sample_count": len(samples)}

    adapter_name = "capture_artifact_hints_adapter"
    metric_name = "capture_artifact_hints_metric"
    register_adapter(adapter_name, CaptureAdapter)
    register_metric(metric_name, capture_metric)

    suite_payload = {
        "version": 1,
        "project": "demo",
        "suite_name": "artifact-hints",
        "models": [{"name": "m1", "adapter": adapter_name}],
        "tests": [
            {
                "id": "t1",
                "type": "generation",
                "prompts": ["a prompt"],
                "seeds": [0],
                "video": {"num_frames": 8},
            }
        ],
        "metrics": [{"name": metric_name}],
        "gates": [{"metric": f"{metric_name}.score", "op": ">=", "value": 0.1}],
        "artifacts": {"encode": "h265", "keep_workdir": True},
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    suite = load_suite(suite_path)
    result = run_suite(suite=suite, artifacts_dir=tmp_path / "artifacts")

    assert result["status"] == "PASS"
    assert captured_video_cfg["encode"] == "h265"
    assert captured_metric_params["keep_workdir"] is True
