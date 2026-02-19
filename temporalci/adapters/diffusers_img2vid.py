from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from temporalci.adapters.base import ModelAdapter
from temporalci.errors import AdapterError
from temporalci.types import GeneratedSample
from temporalci.utils import as_bool


class DiffusersImg2VidAdapter(ModelAdapter):
    """
    Diffusers-based image-to-video adapter.

    This adapter is intentionally focused on Stable Video Diffusion style pipelines.
    Dependencies are imported lazily so TemporalCI can run without GPU packages.
    """

    _pipeline_cache: dict[tuple[str, str, str, str, bool, bool], Any] = {}

    def __init__(self, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, params)
        checkpoint = str(self.params.get("checkpoint", "")).strip()
        if not checkpoint:
            raise ValueError("diffusers_img2vid adapter requires 'checkpoint' in model params")
        self.checkpoint = checkpoint
        self.device = str(self.params.get("device", "cuda")).strip() or "cuda"
        self.torch_dtype = str(self.params.get("torch_dtype", "float16")).strip()
        self.variant = str(self.params.get("variant", "fp16")).strip()
        self.use_safetensors = as_bool(self.params.get("use_safetensors", True), default=True)
        self.disable_progress_bar = as_bool(
            self.params.get("disable_progress_bar", True), default=True
        )
        self.default_fps = int(self.params.get("fps", 6))
        self.default_num_frames = int(self.params.get("num_frames", 25))
        self.default_num_inference_steps = int(self.params.get("num_inference_steps", 25))
        self.default_decode_chunk_size = int(self.params.get("decode_chunk_size", 8))
        self.default_motion_bucket_id = int(self.params.get("motion_bucket_id", 127))
        self.default_noise_aug_strength = float(self.params.get("noise_aug_strength", 0.02))
        self.default_min_guidance_scale = float(self.params.get("min_guidance_scale", 1.0))
        self.default_max_guidance_scale = float(self.params.get("max_guidance_scale", 3.0))

    def generate(
        self,
        *,
        test_id: str,
        prompt: str,
        seed: int,
        video_cfg: dict[str, Any],
        output_dir: Path,
    ) -> GeneratedSample:
        torch, Image, export_to_video = self._load_runtime_dependencies()
        pipeline = self._load_pipeline(torch=torch)

        init_image_path = self._resolve_init_image_path(
            prompt=prompt, seed=seed, video_cfg=video_cfg
        )
        with Image.open(init_image_path) as init_handle:
            init_image = init_handle.convert("RGB")

        num_frames = int(video_cfg.get("num_frames", self.default_num_frames))
        fps = int(video_cfg.get("fps", self.default_fps))
        num_inference_steps = int(
            video_cfg.get("num_inference_steps", self.default_num_inference_steps)
        )
        decode_chunk_size = int(video_cfg.get("decode_chunk_size", self.default_decode_chunk_size))
        motion_bucket_id = int(video_cfg.get("motion_bucket_id", self.default_motion_bucket_id))
        noise_aug_strength = float(
            video_cfg.get("noise_aug_strength", self.default_noise_aug_strength)
        )
        min_guidance_scale = float(
            video_cfg.get("min_guidance_scale", self.default_min_guidance_scale)
        )
        max_guidance_scale = float(
            video_cfg.get("max_guidance_scale", self.default_max_guidance_scale)
        )

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        output = pipeline(
            image=init_image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            generator=generator,
        )
        frames = output.frames
        if frames and isinstance(frames[0], list):
            frames = frames[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha1(f"{test_id}|{prompt}|{seed}".encode("utf-8")).hexdigest()[:12]
        video_path = output_dir / f"{token}.mp4"
        export_to_video(frames, str(video_path), fps=fps)
        encode_requested = (
            str(video_cfg.get("encode", self.params.get("encode", "h264"))).strip().lower()
            or "h264"
        )
        encode_applied, encode_error = self._maybe_reencode(
            video_path=video_path, encode=encode_requested
        )

        stream = self._frame_luma_stream(frames)
        metadata = {
            "adapter": "diffusers_img2vid",
            "checkpoint": self.checkpoint,
            "init_image": str(init_image_path),
            "fps": fps,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "encode_requested": encode_requested,
            "encode_applied": encode_applied,
        }
        if encode_error:
            metadata["encode_error"] = encode_error

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return GeneratedSample(
            test_id=test_id,
            prompt=prompt,
            seed=seed,
            video_path=str(video_path),
            evaluation_stream=stream,
            metadata=metadata,
        )

    def _resolve_init_image_path(
        self, *, prompt: str, seed: int, video_cfg: dict[str, Any]
    ) -> Path:
        single = video_cfg.get("init_image", self.params.get("init_image"))
        if isinstance(single, str) and single.strip():
            path = Path(single)
            if not path.exists():
                raise FileNotFoundError(f"init_image not found: {path}")
            return path

        candidates = video_cfg.get("init_images", self.params.get("init_images"))
        if isinstance(candidates, list) and candidates:
            valid: list[Path] = []
            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                path = Path(candidate)
                if path.exists():
                    valid.append(path)
            if valid:
                key = hashlib.sha1(f"{prompt}|{seed}|{self.model_name}".encode("utf-8")).hexdigest()
                idx = int(key[:8], 16) % len(valid)
                return valid[idx]

        raise ValueError(
            "diffusers_img2vid requires an init image. "
            "Set test.video.init_image or model.params.init_image."
        )

    def _load_pipeline(self, *, torch: Any) -> Any:
        key = (
            self.checkpoint,
            self.device,
            self.torch_dtype,
            self.variant,
            self.use_safetensors,
            self.disable_progress_bar,
        )
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached

        try:
            from diffusers import StableVideoDiffusionPipeline
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(
                "diffusers is not installed. Install optional deps for diffusers adapter."
            ) from exc

        if self.disable_progress_bar:
            try:
                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
            except Exception:  # noqa: BLE001
                pass

        dtype = self._resolve_torch_dtype(torch)
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            variant=self.variant or None,
            use_safetensors=self.use_safetensors,
        )
        pipeline = pipeline.to(self.device)
        if self.disable_progress_bar and hasattr(pipeline, "set_progress_bar_config"):
            try:
                pipeline.set_progress_bar_config(disable=True)
            except Exception:  # noqa: BLE001
                pass
        self._pipeline_cache[key] = pipeline
        return pipeline

    def _maybe_reencode(self, *, video_path: Path, encode: str) -> tuple[bool, str | None]:
        if encode == "h264":
            return False, None
        if encode != "h265":
            return False, f"unsupported encode '{encode}'"

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "ffmpeg not found in PATH"

        temp_path = video_path.with_suffix(".h265.mp4")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p",
            str(temp_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            if temp_path.exists():
                temp_path.unlink()
            message = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg encode failed"
            return False, message
        if not temp_path.exists():
            return False, "ffmpeg did not produce output file"

        temp_path.replace(video_path)
        return True, None

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = mapping.get(self.torch_dtype.lower())
        if dtype is None:
            raise ValueError(
                f"unsupported torch_dtype '{self.torch_dtype}'. "
                "use one of: float16, bfloat16, float32"
            )
        return dtype

    def _load_runtime_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("torch is not installed") from exc
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("Pillow is not installed") from exc
        try:
            from diffusers.utils import export_to_video
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("diffusers export_to_video is unavailable") from exc
        return torch, Image, export_to_video

    def _frame_luma_stream(self, frames: list[Any]) -> list[float]:
        stream: list[float] = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                gray = arr.astype(np.float32)
            else:
                gray = arr[..., :3].astype(np.float32).mean(axis=2)
            value = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
            stream.append(round(value, 6))
        return stream
