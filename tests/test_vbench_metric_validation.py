from __future__ import annotations

import os

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
    videos_path = os.getenv("RUN_VBENCH_VIDEOS_PATH")
    if not videos_path:
        pytest.skip("set RUN_VBENCH_VIDEOS_PATH for integration test")

    result = run_metric(
        name="vbench_official",
        samples=[],
        params={
            "mode": "standard",
            "videos_path": videos_path,
            "dimensions": ["motion_smoothness"],
        },
    )
    assert "score" in result
