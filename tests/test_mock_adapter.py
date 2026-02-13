from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.adapters.mock import MockAdapter


def test_mock_adapter_honors_sleep_sec(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[float] = []

    def fake_sleep(value: float) -> None:
        calls.append(value)

    monkeypatch.setattr("temporalci.adapters.mock.time.sleep", fake_sleep)
    adapter = MockAdapter(model_name="mock-delayed", params={"sleep_sec": "0.25"})
    sample = adapter.generate(
        test_id="t1",
        prompt="hello",
        seed=0,
        video_cfg={"num_frames": 3},
        output_dir=tmp_path,
    )

    assert sample.test_id == "t1"
    assert len(calls) == 1
    assert calls[0] == 0.25
