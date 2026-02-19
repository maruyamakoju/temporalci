from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from temporalci.adapters.base import ModelAdapter
from temporalci.errors import AdapterError
from temporalci.types import GeneratedSample

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class FrameArchiveAdapter(ModelAdapter):
    """Adapter that resolves pre-existing image frames from a directory.

    The *prompt* is treated as a filename stem.  The adapter locates the
    corresponding file inside ``archive_dir`` and returns it as the sample
    artefact.  No generation occurs — this adapter is designed for
    inspection pipelines where images already exist on disk.
    """

    def generate(
        self,
        *,
        test_id: str,
        prompt: str,
        seed: int,
        video_cfg: dict[str, Any],
        output_dir: Path,
    ) -> GeneratedSample:
        archive_dir = self._resolve_archive_dir()
        frame_path = self._find_frame(archive_dir, prompt)

        copy_to_output = str(self.params.get("copy_to_output", "false")).lower() in (
            "true",
            "1",
            "yes",
        )
        if copy_to_output:
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / frame_path.name
            shutil.copy2(frame_path, dest)
            resolved_path = str(dest)
        else:
            resolved_path = str(frame_path)

        metadata: dict[str, Any] = {"adapter": "frame_archive"}
        camera = self.params.get("camera")
        if camera:
            metadata["camera"] = str(camera)
        metadata["archive_dir"] = str(archive_dir)

        return GeneratedSample(
            test_id=test_id,
            prompt=prompt,
            seed=seed,
            video_path=resolved_path,
            evaluation_stream=[],
            metadata=metadata,
        )

    def _resolve_archive_dir(self) -> Path:
        raw = self.params.get("archive_dir")
        if not raw:
            raise AdapterError("frame_archive adapter requires 'archive_dir' param")
        archive = Path(str(raw))
        if not archive.is_dir():
            raise AdapterError(f"archive_dir is not a directory: {archive}")
        return archive

    def _find_frame(self, archive_dir: Path, prompt: str) -> Path:
        extension = self.params.get("extension")
        if extension:
            ext = str(extension).strip()
            if not ext.startswith("."):
                ext = f".{ext}"
            candidate = archive_dir / f"{prompt}{ext}"
            if candidate.is_file():
                return candidate
            raise AdapterError(f"frame not found: {candidate}")

        for ext in _IMAGE_EXTENSIONS:
            candidate = archive_dir / f"{prompt}{ext}"
            if candidate.is_file():
                return candidate

        raise AdapterError(
            f"frame not found for prompt '{prompt}' in {archive_dir} "
            f"(tried extensions: {', '.join(_IMAGE_EXTENSIONS)})"
        )
