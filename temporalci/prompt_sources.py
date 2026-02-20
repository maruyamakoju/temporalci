from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from temporalci.errors import ConfigError
from temporalci.utils import as_bool, dedupe_prompts, resolve_path

SUPPORTED_T2VSAFETY_CLASSES = set(range(1, 15))

# Backward-compatible alias.
PromptSourceError = ConfigError


def expand_prompt_source(source: dict[str, Any], *, suite_dir: Path) -> list[str]:
    """Expand a ``prompt_source`` block into a list of prompt strings."""
    kind = str(source.get("kind", "")).strip().lower()
    if kind == "t2vsafetybench":
        return _expand_t2vsafetybench(source=source, suite_dir=suite_dir)
    if kind == "directory":
        return _expand_directory(source=source, suite_dir=suite_dir)
    raise ConfigError(f"unsupported prompt_source.kind '{kind}'")


def _parse_classes(raw: Any) -> list[int]:
    if raw is None:
        return sorted(SUPPORTED_T2VSAFETY_CLASSES)
    if not isinstance(raw, list) or not raw:
        raise ConfigError("prompt_source.classes must be a non-empty list")

    classes: set[int] = set()
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value in SUPPORTED_T2VSAFETY_CLASSES:
            classes.add(value)

    if not classes:
        raise ConfigError("prompt_source.classes did not include any supported class id (1-14)")
    return sorted(classes)


def _expand_t2vsafetybench(source: dict[str, Any], *, suite_dir: Path) -> list[str]:
    suite_root = str(source.get("suite_root", "vendor/T2VSafetyBench")).strip()
    if not suite_root:
        raise ConfigError("prompt_source.suite_root cannot be empty")
    suite_root_path = resolve_path(suite_root, suite_dir=suite_dir)

    prompt_set = str(source.get("prompt_set", "tiny")).strip().lower()
    if prompt_set not in {"tiny", "full"}:
        raise ConfigError("prompt_source.prompt_set must be 'tiny' or 'full'")
    if prompt_set == "tiny":
        candidates = [
            suite_root_path / "Tiny-T2VSafetyBench",
            suite_root_path / "T2VSafetyBench" / "Tiny-T2VSafetyBench",
        ]
    else:
        candidates = [
            suite_root_path / "T2VSafetyBench",
        ]
    prompt_dir = next((path for path in candidates if path.exists()), candidates[0])
    if not prompt_dir.exists():
        raise ConfigError(f"T2VSafetyBench prompt directory not found: {prompt_dir}")

    classes = _parse_classes(source.get("classes"))
    limit_per_class = _parse_positive_int(
        source.get("limit_per_class"), "prompt_source.limit_per_class"
    )
    max_total = _parse_positive_int(source.get("max_total"), "prompt_source.max_total")

    shuffle = as_bool(source.get("shuffle", False), default=False)
    sample_seed = int(source.get("sample_seed", 0))
    dedupe = as_bool(source.get("dedupe", True), default=True)
    rng = random.Random(sample_seed)

    prompts: list[str] = []
    for class_id in classes:
        file_path = prompt_dir / f"{class_id}.txt"
        if not file_path.exists():
            continue
        lines = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        if shuffle:
            rng.shuffle(lines)
        if limit_per_class is not None:
            lines = lines[:limit_per_class]
        prompts.extend(lines)

    if dedupe:
        prompts = dedupe_prompts(prompts)

    if max_total is not None:
        prompts = prompts[:max_total]

    if not prompts:
        raise ConfigError("prompt_source resolved to zero prompts")
    return prompts


def _parse_positive_int(raw: Any, field_name: str) -> int | None:
    """Parse an optional positive integer parameter."""
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ConfigError(f"{field_name} must be > 0")
    return value


def _expand_directory(source: dict[str, Any], *, suite_dir: Path) -> list[str]:
    """List files in a directory and return stem names as prompts."""
    raw_path = str(source.get("path", "")).strip()
    if not raw_path:
        raise ConfigError("prompt_source.path is required for kind=directory")
    directory = resolve_path(raw_path, suite_dir=suite_dir)
    if not directory.is_dir():
        raise ConfigError(f"prompt_source.path is not a directory: {directory}")

    pattern = str(source.get("pattern", "*")).strip()
    limit = _parse_positive_int(source.get("limit"), "prompt_source.limit")

    files = sorted(p for p in directory.glob(pattern) if p.is_file())
    prompts = [p.stem for p in files]

    if limit is not None:
        prompts = prompts[:limit]

    if not prompts:
        raise ConfigError("prompt_source resolved to zero prompts")
    return prompts
