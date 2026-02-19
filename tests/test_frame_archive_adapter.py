from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from temporalci.adapters.frame_archive import FrameArchiveAdapter
from temporalci.errors import AdapterError
from temporalci.types import GeneratedSample


def _make_adapter(params: dict[str, Any]) -> FrameArchiveAdapter:
    return FrameArchiveAdapter(model_name="test-cam", params=params)


def _generate(adapter: FrameArchiveAdapter, prompt: str, output_dir: Path) -> GeneratedSample:
    return adapter.generate(
        test_id="t1",
        prompt=prompt,
        seed=0,
        video_cfg={},
        output_dir=output_dir,
    )


class TestFrameArchiveAdapter:
    def test_resolve_jpg(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "frame_001.jpg").write_bytes(b"\xff\xd8")
        adapter = _make_adapter({"archive_dir": str(archive)})
        sample = _generate(adapter, "frame_001", tmp_path / "out")
        assert sample.prompt == "frame_001"
        assert sample.video_path == str(archive / "frame_001.jpg")
        assert sample.evaluation_stream == []
        assert sample.metadata["adapter"] == "frame_archive"

    def test_extension_fallback(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "frame_001.png").write_bytes(b"\x89PNG")
        adapter = _make_adapter({"archive_dir": str(archive)})
        sample = _generate(adapter, "frame_001", tmp_path / "out")
        assert sample.video_path == str(archive / "frame_001.png")

    def test_explicit_extension(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "frame_001.bmp").write_bytes(b"BM")
        adapter = _make_adapter({"archive_dir": str(archive), "extension": "bmp"})
        sample = _generate(adapter, "frame_001", tmp_path / "out")
        assert sample.video_path == str(archive / "frame_001.bmp")

    def test_missing_frame_raises(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        adapter = _make_adapter({"archive_dir": str(archive)})
        with pytest.raises(AdapterError, match="frame not found"):
            _generate(adapter, "no_such_frame", tmp_path / "out")

    def test_missing_archive_dir_raises(self, tmp_path: Path) -> None:
        adapter = _make_adapter({"archive_dir": str(tmp_path / "nope")})
        with pytest.raises(AdapterError, match="not a directory"):
            _generate(adapter, "x", tmp_path / "out")

    def test_no_archive_dir_param_raises(self, tmp_path: Path) -> None:
        adapter = _make_adapter({})
        with pytest.raises(AdapterError, match="requires 'archive_dir'"):
            _generate(adapter, "x", tmp_path / "out")

    def test_copy_to_output(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "f.jpg").write_bytes(b"\xff\xd8data")
        out = tmp_path / "out"
        adapter = _make_adapter({"archive_dir": str(archive), "copy_to_output": "true"})
        sample = _generate(adapter, "f", out)
        assert Path(sample.video_path) == out / "f.jpg"
        assert (out / "f.jpg").read_bytes() == b"\xff\xd8data"

    def test_camera_metadata(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "f.jpg").write_bytes(b"\xff")
        adapter = _make_adapter({"archive_dir": str(archive), "camera": "left"})
        sample = _generate(adapter, "f", tmp_path / "out")
        assert sample.metadata["camera"] == "left"

    def test_explicit_extension_missing_raises(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "f.jpg").write_bytes(b"\xff")
        adapter = _make_adapter({"archive_dir": str(archive), "extension": "png"})
        with pytest.raises(AdapterError, match="frame not found"):
            _generate(adapter, "f", tmp_path / "out")
