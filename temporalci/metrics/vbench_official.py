from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.types import GeneratedSample

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".gif", ".mov", ".mkv", ".avi", ".webm"}
CUSTOM_INPUT_SUPPORTED_DIMS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_dimension_score(payload: Any) -> float | None:
    if _is_number(payload):
        return float(payload)
    if isinstance(payload, (list, tuple)) and payload:
        head = payload[0]
        if _is_number(head):
            return float(head)
    if isinstance(payload, dict):
        score = payload.get("score")
        if _is_number(score):
            return float(score)
    return None


def _resolve_full_info_json() -> str:
    import vbench

    base = Path(vbench.__file__).resolve().parent
    candidate = base / "VBench_full_info.json"
    if not candidate.exists():
        raise FileNotFoundError(
            "VBench_full_info.json was not found in installed vbench package. "
            "Set params.full_info_json explicitly."
        )
    return str(candidate)


def _materialize_video_inputs(
    samples: list[GeneratedSample],
    target_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    prompt_map: dict[str, str] = {}

    for idx, sample in enumerate(samples):
        source = Path(sample.video_path)
        if not source.exists() or source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue
        destination = target_dir / f"sample_{idx:05d}{source.suffix.lower()}"
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

        destination_abs = str(destination.resolve())
        copied.append(destination)
        prompt_map[destination_abs] = sample.prompt
    return copied, prompt_map


def _count_video_files(path: Path) -> int:
    count = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
            count += 1
    return count


def evaluate(samples: list[GeneratedSample], params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}

    mode = str(params.get("mode", "custom_input")).strip().lower()
    if mode not in {"custom_input", "standard"}:
        raise ValueError("vbench_official params.mode must be 'custom_input' or 'standard'")

    dimensions = params.get(
        "dimensions",
        ["subject_consistency", "motion_smoothness", "dynamic_degree"],
    )
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("vbench_official requires params.dimensions as non-empty list")
    dimensions = [str(dim).strip() for dim in dimensions if str(dim).strip()]
    if not dimensions:
        raise ValueError("vbench_official params.dimensions resolved to empty")

    if mode == "custom_input":
        unsupported = sorted(set(dimensions) - CUSTOM_INPUT_SUPPORTED_DIMS)
        if unsupported:
            allowed = ", ".join(sorted(CUSTOM_INPUT_SUPPORTED_DIMS))
            raise ValueError(
                "vbench_official custom_input only supports specific dimensions. "
                f"unsupported={unsupported}, allowed=[{allowed}]"
            )

    try:
        from vbench import VBench
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "vbench is not installed. Install official dependency with `pip install vbench`."
        ) from exc

    output_root = Path(str(params.get("output_dir", "artifacts/vbench_official"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="temporalci_vbench_", dir=str(output_root)))
    videos_dir = temp_dir / "videos"
    keep_workdir = bool(params.get("keep_workdir", False))

    try:
        if mode == "custom_input":
            copied_videos, prompt_map = _materialize_video_inputs(samples=samples, target_dir=videos_dir)
            if not copied_videos:
                raise ValueError(
                    "vbench_official received no video files. "
                    f"Supported suffixes: {sorted(SUPPORTED_VIDEO_SUFFIXES)}"
                )
            videos_path = videos_dir
            prompt_list: Any = prompt_map
            sample_count = len(copied_videos)
            vbench_mode = "custom_input"
        else:
            videos_path_raw = str(params.get("videos_path", "")).strip()
            if not videos_path_raw:
                raise ValueError("vbench_official mode='standard' requires params.videos_path")
            videos_path = Path(videos_path_raw).resolve()
            if not videos_path.exists():
                raise FileNotFoundError(f"vbench_official videos_path not found: {videos_path}")
            prompt_list = []
            sample_count = _count_video_files(videos_path)
            if sample_count == 0:
                raise ValueError(f"no supported video files found under {videos_path}")
            vbench_mode = "vbench_standard"

        full_info_json = str(params.get("full_info_json", "")).strip() or _resolve_full_info_json()
        device = str(params.get("device", "cuda")).strip() or "cuda"
        run_name = str(params.get("run_name", "temporalci")).strip() or "temporalci"
        local = bool(params.get("load_ckpt_from_local", False))
        read_frame = bool(params.get("read_frame", False))

        evaluator = VBench(device, full_info_json, str(temp_dir))
        evaluator.evaluate(
            videos_path=str(videos_path),
            name=run_name,
            prompt_list=prompt_list,
            dimension_list=[str(dim) for dim in dimensions],
            local=local,
            read_frame=read_frame,
            mode=vbench_mode,
        )

        result_path = temp_dir / f"{run_name}_eval_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"vbench result file not found: {result_path}")

        raw_payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("vbench result file must be a JSON object")

        dim_scores: dict[str, float] = {}
        for dim in dimensions:
            key = str(dim)
            score = _extract_dimension_score(raw_payload.get(key))
            if score is not None:
                dim_scores[key] = round(score, 6)

        total = mean(dim_scores.values()) if dim_scores else 0.0
        result: dict[str, Any] = {
            "score": round(total, 6),
            "dims": dim_scores,
            "sample_count": sample_count,
            "raw_result_path": str(result_path),
            "backend": "vbench_official",
            "mode": mode,
        }

        if keep_workdir:
            result["work_dir"] = str(temp_dir)
        return result
    finally:
        if not keep_workdir:
            shutil.rmtree(temp_dir, ignore_errors=True)
