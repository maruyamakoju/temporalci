from __future__ import annotations

import random
from pathlib import Path
from typing import Any

SUPPORTED_T2VSAFETY_CLASSES = set(range(1, 15))


class PromptSourceError(ValueError):
    pass


def expand_prompt_source(source: dict[str, Any], *, suite_dir: Path) -> list[str]:
    kind = str(source.get("kind", "")).strip().lower()
    if kind == "t2vsafetybench":
        return _expand_t2vsafetybench(source=source, suite_dir=suite_dir)
    raise PromptSourceError(f"unsupported prompt_source.kind '{kind}'")


def _resolve_path(raw_path: str, *, suite_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    local = (suite_dir / candidate).resolve()
    if local.exists():
        return local
    return (Path.cwd() / candidate).resolve()


def _parse_classes(raw: Any) -> list[int]:
    if raw is None:
        return sorted(SUPPORTED_T2VSAFETY_CLASSES)
    if not isinstance(raw, list) or not raw:
        raise PromptSourceError("prompt_source.classes must be a non-empty list")

    classes: set[int] = set()
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value in SUPPORTED_T2VSAFETY_CLASSES:
            classes.add(value)

    if not classes:
        raise PromptSourceError(
            "prompt_source.classes did not include any supported class id (1-14)"
        )
    return sorted(classes)


def _expand_t2vsafetybench(source: dict[str, Any], *, suite_dir: Path) -> list[str]:
    suite_root = str(source.get("suite_root", "vendor/T2VSafetyBench")).strip()
    if not suite_root:
        raise PromptSourceError("prompt_source.suite_root cannot be empty")
    suite_root_path = _resolve_path(suite_root, suite_dir=suite_dir)

    prompt_set = str(source.get("prompt_set", "tiny")).strip().lower()
    if prompt_set not in {"tiny", "full"}:
        raise PromptSourceError("prompt_source.prompt_set must be 'tiny' or 'full'")
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
        raise PromptSourceError(f"T2VSafetyBench prompt directory not found: {prompt_dir}")

    classes = _parse_classes(source.get("classes"))
    limit_per_class_raw = source.get("limit_per_class")
    limit_per_class: int | None = None
    if limit_per_class_raw is not None:
        limit_per_class = int(limit_per_class_raw)
        if limit_per_class <= 0:
            raise PromptSourceError("prompt_source.limit_per_class must be > 0")

    max_total_raw = source.get("max_total")
    max_total: int | None = None
    if max_total_raw is not None:
        max_total = int(max_total_raw)
        if max_total <= 0:
            raise PromptSourceError("prompt_source.max_total must be > 0")

    shuffle = bool(source.get("shuffle", False))
    sample_seed = int(source.get("sample_seed", 0))
    dedupe = bool(source.get("dedupe", True))
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
        unique: list[str] = []
        seen: set[str] = set()
        for prompt in prompts:
            normalized = " ".join(prompt.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(prompt)
        prompts = unique

    if max_total is not None:
        prompts = prompts[:max_total]

    if not prompts:
        raise PromptSourceError("prompt_source resolved to zero prompts")
    return prompts
