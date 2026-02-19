from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.errors import ConfigError
from temporalci.prompt_sources import expand_prompt_source


@pytest.fixture()
def frame_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    for name in ["frame_003.jpg", "frame_001.jpg", "frame_002.png", "frame_004.bmp"]:
        (d / name).write_bytes(b"\x00")
    return d


class TestDirectoryPromptSource:
    def test_basic_listing(self, frame_dir: Path) -> None:
        prompts = expand_prompt_source(
            {"kind": "directory", "path": str(frame_dir)},
            suite_dir=frame_dir.parent,
        )
        assert prompts == ["frame_001", "frame_002", "frame_003", "frame_004"]

    def test_sorted_deterministic(self, frame_dir: Path) -> None:
        prompts = expand_prompt_source(
            {"kind": "directory", "path": str(frame_dir)},
            suite_dir=frame_dir.parent,
        )
        assert prompts == sorted(prompts)

    def test_pattern_filter(self, frame_dir: Path) -> None:
        prompts = expand_prompt_source(
            {"kind": "directory", "path": str(frame_dir), "pattern": "*.jpg"},
            suite_dir=frame_dir.parent,
        )
        assert prompts == ["frame_001", "frame_003"]

    def test_limit(self, frame_dir: Path) -> None:
        prompts = expand_prompt_source(
            {"kind": "directory", "path": str(frame_dir), "limit": 2},
            suite_dir=frame_dir.parent,
        )
        assert len(prompts) == 2
        assert prompts == ["frame_001", "frame_002"]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ConfigError, match="zero prompts"):
            expand_prompt_source(
                {"kind": "directory", "path": str(empty)},
                suite_dir=tmp_path,
            )

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a directory"):
            expand_prompt_source(
                {"kind": "directory", "path": str(tmp_path / "nonexistent")},
                suite_dir=tmp_path,
            )

    def test_path_required(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="path is required"):
            expand_prompt_source(
                {"kind": "directory"},
                suite_dir=tmp_path,
            )

    def test_relative_path(self, tmp_path: Path) -> None:
        d = tmp_path / "sub" / "frames"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"\x00")
        prompts = expand_prompt_source(
            {"kind": "directory", "path": "sub/frames"},
            suite_dir=tmp_path,
        )
        assert prompts == ["img"]

    def test_subdirectories_excluded(self, tmp_path: Path) -> None:
        d = tmp_path / "frames"
        d.mkdir()
        (d / "file.jpg").write_bytes(b"\x00")
        (d / "subdir").mkdir()
        prompts = expand_prompt_source(
            {"kind": "directory", "path": str(d)},
            suite_dir=tmp_path,
        )
        assert prompts == ["file"]
