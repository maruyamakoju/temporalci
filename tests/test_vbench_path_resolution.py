from __future__ import annotations

import os
from pathlib import Path

import pytest

from temporalci.metrics.vbench_official import _resolve_standard_videos_path


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00")


def test_resolve_standard_videos_path_explicit(tmp_path: Path) -> None:
    videos_dir = tmp_path / "manual" / "videos"
    _touch_video(videos_dir / "sample.mp4")

    resolved, source = _resolve_standard_videos_path(params={"videos_path": str(videos_dir)})

    assert resolved == videos_dir.resolve()
    assert source == "explicit"


def test_resolve_standard_videos_path_auto_selects_latest(tmp_path: Path) -> None:
    old_videos = tmp_path / "run_old" / "videos"
    new_videos = tmp_path / "run_new" / "videos"
    _touch_video(old_videos / "old.mp4")
    _touch_video(new_videos / "new.mp4")

    os.utime(old_videos / "old.mp4", (100, 100))
    os.utime(new_videos / "new.mp4", (200, 200))

    resolved, source = _resolve_standard_videos_path(
        params={
            "videos_path": "auto",
            "videos_auto_root": str(tmp_path),
        }
    )

    assert resolved == new_videos.resolve()
    assert source == "auto"


def test_resolve_standard_videos_path_auto_root_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing-root"
    with pytest.raises(FileNotFoundError):
        _resolve_standard_videos_path(
            params={
                "videos_path": "auto",
                "videos_auto_root": str(missing),
            }
        )


def test_resolve_standard_videos_path_auto_no_candidates(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        _resolve_standard_videos_path(
            params={
                "videos_path": "auto",
                "videos_auto_root": str(tmp_path),
            }
        )
