from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

from temporalci.errors import MetricError
from temporalci.types import GeneratedSample
from temporalci.utils import as_bool, as_int, is_number

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".gif", ".mov", ".mkv", ".avi", ".webm"}
CUSTOM_INPUT_SUPPORTED_DIMS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}


# ---------------------------------------------------------------------------
# Dimension score extraction
# ---------------------------------------------------------------------------


def _extract_dimension_score(payload: Any) -> float | None:
    if is_number(payload):
        return float(payload)
    if isinstance(payload, (list, tuple)) and payload:
        head = payload[0]
        if is_number(head):
            return float(head)
    if isinstance(payload, dict):
        score = payload.get("score")
        if score is not None and is_number(score):
            return float(score)
    return None


# ---------------------------------------------------------------------------
# VBench metadata helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Video materialization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Standard mode video path resolution
# ---------------------------------------------------------------------------


def _count_video_files(path: Path) -> int:
    count = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
            count += 1
    return count


def _latest_video_mtime(path: Path) -> float:
    latest = 0.0
    for child in path.iterdir():
        if not child.is_file() or child.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue
        latest = max(latest, float(child.stat().st_mtime))
    if latest > 0.0:
        return latest
    return float(path.stat().st_mtime)


def _discover_video_dirs(
    *,
    root: Path,
    max_depth: int,
    max_candidates: int,
) -> list[Path]:
    root_depth = len(root.parts)
    candidates: list[Path] = []
    for current_root, child_dirs, _child_files in os.walk(root):
        current_path = Path(current_root)
        depth = len(current_path.parts) - root_depth
        if depth > max_depth:
            child_dirs[:] = []
            continue

        if current_path.name.lower() == "videos":
            if _count_video_files(current_path) > 0:
                candidates.append(current_path.resolve())
                if len(candidates) >= max_candidates:
                    break
            child_dirs[:] = []

    return sorted(candidates, key=_latest_video_mtime, reverse=True)


def _resolve_standard_videos_path(*, params: dict[str, Any]) -> tuple[Path, str]:
    explicit_raw = str(params.get("videos_path", "")).strip()
    if explicit_raw and explicit_raw.lower() != "auto":
        explicit = Path(explicit_raw).resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"vbench_official videos_path not found: {explicit}")
        sample_count = _count_video_files(explicit)
        if sample_count <= 0:
            raise ValueError(f"no supported video files found under {explicit}")
        return explicit, "explicit"

    auto_root_raw = str(params.get("videos_auto_root", "artifacts")).strip() or "artifacts"
    auto_root = Path(auto_root_raw).resolve()
    if not auto_root.exists():
        raise FileNotFoundError(
            f"vbench_official auto videos root not found: {auto_root}. "
            "Set params.videos_path explicitly or create videos under the auto root."
        )
    max_depth = as_int(params.get("videos_auto_max_depth", 8), default=8, minimum=1)
    max_candidates = as_int(params.get("videos_auto_max_candidates", 256), default=256, minimum=1)
    candidates = _discover_video_dirs(
        root=auto_root, max_depth=max_depth, max_candidates=max_candidates
    )
    if not candidates:
        raise FileNotFoundError(
            "vbench_official could not find any non-empty `videos` directory under "
            f"{auto_root}. Set params.videos_path explicitly."
        )
    return candidates[0], "auto"


# ---------------------------------------------------------------------------
# wget shim for Windows
# ---------------------------------------------------------------------------


def _extract_wget_tokens(command: Any) -> list[str] | None:
    if isinstance(command, str):
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            tokens = command.split()
    elif isinstance(command, (list, tuple)):
        tokens = [str(value) for value in command]
    else:
        return None
    if not tokens:
        return None
    executable = Path(tokens[0]).name.lower()
    if executable not in {"wget", "wget.exe"}:
        return None
    return tokens


def _download_wget_command(*, tokens: list[str]) -> None:
    target_dir = Path(".")
    url = ""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-P" and index + 1 < len(tokens):
            target_dir = Path(tokens[index + 1])
            index += 2
            continue
        lowered = token.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            url = token
        index += 1

    if not url:
        raise RuntimeError("vbench wget invocation did not include URL")

    target_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name or "download.bin"
    destination = target_dir / filename
    if destination.exists():
        return
    urllib.request.urlretrieve(url, str(destination))


def _patch_vbench_wget_runner() -> Any:
    import vbench.utils as vbench_utils

    subprocess_proxy_target = getattr(vbench_utils, "subprocess", None)
    if subprocess_proxy_target is None or not hasattr(subprocess_proxy_target, "run"):
        raise RuntimeError("vbench.utils subprocess module is unavailable for wget patching")

    class _SubprocessProxy:
        def __init__(self, delegate: Any) -> None:
            self._delegate = delegate

        def run(self, command: Any, *args: Any, **kwargs: Any) -> Any:
            tokens = _extract_wget_tokens(command)
            if tokens is not None:
                _download_wget_command(tokens=tokens)
                return self._delegate.CompletedProcess(command, 0)
            return self._delegate.run(command, *args, **kwargs)

    vbench_utils.subprocess = _SubprocessProxy(subprocess_proxy_target)

    def _restore() -> None:
        vbench_utils.subprocess = subprocess_proxy_target

    return _restore


def _ensure_wget_command(*, allow_wget_shim: bool) -> bool:
    if shutil.which("wget"):
        return False
    if os.name != "nt":
        raise RuntimeError(
            "vbench official backend requires `wget` command in PATH on this platform."
        )
    if not allow_wget_shim:
        raise RuntimeError(
            "vbench official backend requires `wget` command in PATH. "
            "Install wget or set params.allow_wget_shim=true."
        )
    return True


# ---------------------------------------------------------------------------
# Mode-specific evaluation runners
# ---------------------------------------------------------------------------


def _prepare_custom_input(
    samples: list[GeneratedSample],
    temp_dir: Path,
) -> tuple[Path, Any, int]:
    """Materialize videos for custom_input mode; return (videos_path, prompt_list, count)."""
    videos_dir = temp_dir / "videos"
    copied_videos, prompt_map = _materialize_video_inputs(samples=samples, target_dir=videos_dir)
    if not copied_videos:
        raise MetricError(
            "vbench_official received no video files. "
            f"Supported suffixes: {sorted(SUPPORTED_VIDEO_SUFFIXES)}"
        )
    return videos_dir, prompt_map, len(copied_videos)


def _prepare_standard(params: dict[str, Any]) -> tuple[Path, str, int]:
    """Resolve videos for standard mode; return (videos_path, source, count)."""
    videos_path, source = _resolve_standard_videos_path(params=params)
    sample_count = _count_video_files(videos_path)
    return videos_path, source, sample_count


def _invoke_vbench(
    *,
    VBench: Any,
    temp_dir: Path,
    videos_path: Path,
    prompt_list: Any,
    dimensions: list[str],
    params: dict[str, Any],
    vbench_mode: str,
    use_windows_wget_patch: bool,
) -> Path:
    """Run VBench evaluation and return the result file path."""
    full_info_json = str(params.get("full_info_json", "")).strip() or _resolve_full_info_json()
    device = str(params.get("device", "cuda")).strip() or "cuda"
    run_name = str(params.get("run_name", "temporalci")).strip() or "temporalci"
    local = as_bool(params.get("load_ckpt_from_local", False), default=False)
    read_frame = as_bool(params.get("read_frame", False), default=False)
    allow_unsafe_torch_load = as_bool(params.get("allow_unsafe_torch_load", False), default=False)

    def _noop_restore() -> None:
        return None

    restore_wget_patch = _noop_restore
    if use_windows_wget_patch:
        restore_wget_patch = _patch_vbench_wget_runner()

    torch_env_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    torch_env_was_set = torch_env_key in os.environ
    if allow_unsafe_torch_load and not torch_env_was_set:
        os.environ[torch_env_key] = "1"

    evaluator = VBench(device, full_info_json, str(temp_dir))
    try:
        try:
            evaluator.evaluate(
                videos_path=str(videos_path),
                name=run_name,
                prompt_list=prompt_list,
                dimension_list=[str(dim) for dim in dimensions],
                local=local,
                read_frame=read_frame,
                mode=vbench_mode,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if not allow_unsafe_torch_load and "Weights only load failed" in message:
                raise MetricError(
                    "vbench_official failed to load official checkpoint in safe mode. "
                    "For trusted checkpoints only, rerun with "
                    "`params.allow_unsafe_torch_load=true`."
                ) from exc
            raise
    finally:
        if allow_unsafe_torch_load and not torch_env_was_set:
            os.environ.pop(torch_env_key, None)
        restore_wget_patch()

    result_path = temp_dir / f"{run_name}_eval_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"vbench result file not found: {result_path}")
    return result_path


def _parse_vbench_results(
    result_path: Path,
    dimensions: list[str],
) -> tuple[float, dict[str, float]]:
    """Read VBench result JSON and extract per-dimension scores."""
    raw_payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("vbench result file must be a JSON object")

    dim_scores: dict[str, float] = {}
    for dim in dimensions:
        score = _extract_dimension_score(raw_payload.get(str(dim)))
        if score is not None:
            dim_scores[str(dim)] = round(score, 6)

    total = mean(dim_scores.values()) if dim_scores else 0.0
    return round(total, 6), dim_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    samples: list[GeneratedSample], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run VBench official evaluation on *samples*."""
    params = params or {}

    mode = str(params.get("mode", "custom_input")).strip().lower()
    if mode not in {"custom_input", "standard"}:
        raise ValueError("vbench_official params.mode must be 'custom_input' or 'standard'")

    dimensions = _validate_dimensions(params, mode)

    try:
        from vbench import VBench
    except Exception as exc:  # noqa: BLE001
        raise MetricError(
            "vbench is not installed. Install official dependency with `pip install vbench`."
        ) from exc

    env_videos_path = str(os.getenv("RUN_VBENCH_VIDEOS_PATH", "")).strip()
    if not params.get("videos_path") and env_videos_path:
        params["videos_path"] = env_videos_path

    output_root = Path(str(params.get("output_dir", "artifacts/vbench_official"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    allow_wget_shim = as_bool(params.get("allow_wget_shim", True), default=True)
    use_windows_wget_patch = _ensure_wget_command(allow_wget_shim=allow_wget_shim)

    temp_dir = Path(tempfile.mkdtemp(prefix="temporalci_vbench_", dir=str(output_root)))
    keep_workdir = as_bool(params.get("keep_workdir", False), default=False)

    try:
        if mode == "custom_input":
            videos_path, prompt_list, sample_count = _prepare_custom_input(samples, temp_dir)
            vbench_mode = "custom_input"
            videos_path_source = None
        else:
            videos_path, videos_path_source, sample_count = _prepare_standard(params)
            prompt_list = []
            vbench_mode = "vbench_standard"

        result_path = _invoke_vbench(
            VBench=VBench,
            temp_dir=temp_dir,
            videos_path=videos_path,
            prompt_list=prompt_list,
            dimensions=dimensions,
            params=params,
            vbench_mode=vbench_mode,
            use_windows_wget_patch=use_windows_wget_patch,
        )

        total, dim_scores = _parse_vbench_results(result_path, dimensions)

        result: dict[str, Any] = {
            "score": total,
            "dims": dim_scores,
            "sample_count": sample_count,
            "raw_result_path": str(result_path),
            "backend": "vbench_official",
            "mode": mode,
        }
        if mode == "standard" and videos_path_source is not None:
            result["resolved_videos_path"] = str(videos_path)
            result["videos_path_source"] = videos_path_source

        if keep_workdir:
            result["work_dir"] = str(temp_dir)
        return result
    finally:
        if not keep_workdir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _validate_dimensions(params: dict[str, Any], mode: str) -> list[str]:
    """Validate and return the requested VBench dimension list."""
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
    return dimensions
