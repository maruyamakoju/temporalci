from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from temporalci.adapters.http import HttpAdapter


def _make_adapter(endpoint: str = "http://localhost:8000/generate") -> HttpAdapter:
    return HttpAdapter(
        model_name="test-model",
        params={"endpoint": endpoint, "timeout_sec": 5},
    )


def _make_response_payload(
    *,
    video_path: str | None = None,
    evaluation_stream: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if video_path is not None:
        payload["video_path"] = video_path
    if evaluation_stream is not None:
        payload["evaluation_stream"] = evaluation_stream
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


class _FakeHTTPResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def test_http_adapter_requires_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        HttpAdapter(model_name="test", params={})


def test_http_adapter_requires_endpoint_nonempty() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        HttpAdapter(model_name="test", params={"endpoint": "  "})


@patch("temporalci.adapters.http.urllib.request.urlopen")
def test_http_adapter_generate_with_stream(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    response_data = _make_response_payload(
        video_path="/remote/video.mp4",
        evaluation_stream=[0.1, 0.2, 0.3],
        metadata={"model_version": "v2"},
    )
    mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(response_data).encode("utf-8"))

    adapter = _make_adapter()
    sample = adapter.generate(
        test_id="t1",
        prompt="a cat",
        seed=42,
        video_cfg={"num_frames": 3},
        output_dir=tmp_path,
    )

    assert sample.test_id == "t1"
    assert sample.prompt == "a cat"
    assert sample.seed == 42
    assert sample.video_path == "/remote/video.mp4"
    assert sample.evaluation_stream == [0.1, 0.2, 0.3]
    assert sample.metadata["adapter"] == "http"
    assert sample.metadata["model_version"] == "v2"
    mock_urlopen.assert_called_once()


@patch("temporalci.adapters.http.urllib.request.urlopen")
def test_http_adapter_fallback_stream(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    response_data = _make_response_payload(video_path="/remote/video.mp4")
    mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(response_data).encode("utf-8"))

    adapter = _make_adapter()
    sample = adapter.generate(
        test_id="t1",
        prompt="a dog",
        seed=0,
        video_cfg={"num_frames": 5},
        output_dir=tmp_path,
    )

    assert len(sample.evaluation_stream) == 5
    for val in sample.evaluation_stream:
        assert 0.0 <= val <= 1.0


@patch("temporalci.adapters.http.urllib.request.urlopen")
def test_http_adapter_missing_video_path_uses_response_copy(
    mock_urlopen: MagicMock, tmp_path: Path
) -> None:
    response_data = _make_response_payload(evaluation_stream=[0.5])
    mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(response_data).encode("utf-8"))

    adapter = _make_adapter()
    sample = adapter.generate(
        test_id="t1", prompt="test", seed=0, video_cfg={}, output_dir=tmp_path
    )

    # When no video_path in response, uses the response copy path
    assert Path(sample.video_path).name.startswith("http_response_")


@patch("temporalci.adapters.http.urllib.request.urlopen")
def test_http_adapter_invalid_metadata_becomes_empty(
    mock_urlopen: MagicMock, tmp_path: Path
) -> None:
    response_data = {"evaluation_stream": [0.5], "metadata": "not-a-dict"}
    mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(response_data).encode("utf-8"))

    adapter = _make_adapter()
    sample = adapter.generate(
        test_id="t1", prompt="test", seed=0, video_cfg={}, output_dir=tmp_path
    )

    assert sample.metadata["adapter"] == "http"
    assert "not-a-dict" not in str(sample.metadata)


@patch("temporalci.adapters.http.urllib.request.urlopen")
def test_http_adapter_saves_response_copy(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    response_data = _make_response_payload(evaluation_stream=[0.1])
    mock_urlopen.return_value = _FakeHTTPResponse(json.dumps(response_data).encode("utf-8"))

    adapter = _make_adapter()
    adapter.generate(test_id="t1", prompt="test", seed=0, video_cfg={}, output_dir=tmp_path)

    response_files = list(tmp_path.glob("http_response_*.json"))
    assert len(response_files) == 1
    saved = json.loads(response_files[0].read_text(encoding="utf-8"))
    assert saved == response_data
