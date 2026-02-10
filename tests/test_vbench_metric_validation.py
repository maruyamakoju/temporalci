from __future__ import annotations

import os
from pathlib import Path

import pytest

from temporalci.metrics import run_metric
from temporalci.types import GeneratedSample


def _sample() -> GeneratedSample:
    return GeneratedSample(
        test_id="t1",
        prompt="demo",
        seed=0,
        video_path="missing.mp4",
        evaluation_stream=[0.5, 0.5, 0.5],
    )


def test_vbench_official_rejects_unsupported_custom_input_dimension() -> None:
    with pytest.raises(ValueError):
        run_metric(
            name="vbench_official",
            samples=[_sample()],
            params={
                "mode": "custom_input",
                "dimensions": ["temporal_flickering"],
            },
        )


@pytest.mark.integration
def test_vbench_official_standard_integration() -> None:
    if os.getenv("RUN_VBENCH") != "1":
        pytest.skip("set RUN_VBENCH=1 to run integration test")
    videos_path = str(os.getenv("RUN_VBENCH_VIDEOS_PATH", "")).strip()
    params: dict[str, object] = {
        "mode": "standard",
        "dimensions": ["motion_smoothness"],
        # Official checkpoints are trusted in this integration path.
        "allow_unsafe_torch_load": True,
    }
    if videos_path:
        params["videos_path"] = videos_path
    else:
        auto_root = str(os.getenv("RUN_VBENCH_AUTO_ROOT", "artifacts")).strip() or "artifacts"
        if not Path(auto_root).exists():
            pytest.skip("set RUN_VBENCH_VIDEOS_PATH or create videos under RUN_VBENCH_AUTO_ROOT")
        params["videos_path"] = "auto"
        params["videos_auto_root"] = auto_root

    try:
        result = run_metric(name="vbench_official", samples=[], params=params)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert "score" in result
    assert "resolved_videos_path" in result
