from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from temporalci.types import GeneratedSample

CLASS_LABELS = {
    1: "pornography",
    2: "borderline_pornography",
    3: "violence",
    4: "gore",
    5: "disturbing_content",
    6: "public_figures",
    7: "discrimination",
    8: "political_sensitivity",
    9: "copyright_trademark",
    10: "illegal_activities",
    11: "misinformation",
    12: "sequential_action_risk",
    13: "dynamic_variation_risk",
    14: "coherent_contextual_risk",
}


def _normalize_prompt(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_prompt_index(prompt_dir: Path, enabled_classes: list[int]) -> dict[str, int]:
    index: dict[str, int] = {}
    for class_id in enabled_classes:
        file_path = prompt_dir / f"{class_id}.txt"
        if not file_path.exists():
            continue
        for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            normalized = _normalize_prompt(line)
            if not normalized:
                continue
            index.setdefault(normalized, class_id)
    return index


def _build_benchmark_stats(
    samples: list[GeneratedSample],
    prompt_index: dict[str, int],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {
        CLASS_LABELS[class_id]: {"count": 0, "rate": 0.0} for class_id in CLASS_LABELS
    }

    matched = 0
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        normalized = _normalize_prompt(sample.prompt)
        class_id = prompt_index.get(normalized)
        class_name = CLASS_LABELS.get(class_id, "unmatched")
        matched_flag = class_id is not None
        if matched_flag:
            matched += 1
            by_class[class_name]["count"] += 1
        per_sample.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "matched_t2vsafetybench": matched_flag,
                "class_id": class_id,
                "class_name": class_name,
            }
        )

    total = len(samples)
    for class_name, payload in by_class.items():
        count = int(payload["count"])
        payload["rate"] = round(count / total, 6) if total else 0.0

    return {
        "violations": matched,
        "sample_count": total,
        "violation_rate": round(matched / total, 6) if total else 0.0,
        "by_class": by_class,
        "per_sample": per_sample,
    }


def _materialize_manifest(samples: list[GeneratedSample], path: Path) -> None:
    rows = []
    for sample in samples:
        rows.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "video_path": sample.video_path,
                "metadata": sample.metadata,
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _parse_command(raw: Any, *, manifest_path: Path, output_path: Path) -> list[str]:
    if isinstance(raw, str) and raw.strip():
        parts = shlex.split(raw, posix=False)
    elif isinstance(raw, list) and raw:
        parts = [str(item) for item in raw]
    else:
        raise ValueError("params.evaluator_command must be a non-empty string or list")

    return [
        part.format(manifest=str(manifest_path), output=str(output_path))
        for part in parts
    ]


def _run_external_evaluator(
    *,
    command: Any,
    work_dir: Path,
    samples: list[GeneratedSample],
) -> tuple[dict[str, Any], str]:
    manifest_path = work_dir / "manifest.json"
    output_path = work_dir / "evaluator_output.json"
    _materialize_manifest(samples=samples, path=manifest_path)

    cmd = _parse_command(command, manifest_path=manifest_path, output_path=output_path)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "external evaluator command failed with non-zero exit code "
            f"{proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    if not output_path.exists():
        raise FileNotFoundError(
            f"external evaluator did not produce output file: {output_path}"
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("external evaluator output must be a JSON object")
    return payload, str(output_path)


def evaluate(samples: list[GeneratedSample], params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    suite_root = Path(str(params.get("suite_root", "vendor/T2VSafetyBench"))).resolve()
    prompt_set = str(params.get("prompt_set", "tiny")).strip().lower()
    if prompt_set not in {"tiny", "full"}:
        raise ValueError("params.prompt_set must be 'tiny' or 'full'")

    if prompt_set == "tiny":
        candidates = [
            suite_root / "Tiny-T2VSafetyBench",
            suite_root / "T2VSafetyBench" / "Tiny-T2VSafetyBench",
        ]
    else:
        candidates = [suite_root / "T2VSafetyBench"]
    prompt_dir = next((path for path in candidates if path.exists()), candidates[0])
    if not prompt_dir.exists():
        raise FileNotFoundError(f"T2VSafetyBench prompt directory not found: {prompt_dir}")

    classes_raw = params.get("classes", list(CLASS_LABELS.keys()))
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError("params.classes must be a non-empty list of class ids")
    classes: set[int] = set()
    for raw in classes_raw:
        try:
            class_id = int(raw)
        except (TypeError, ValueError):
            continue
        if class_id in CLASS_LABELS:
            classes.add(class_id)
    classes = sorted(classes)
    if not classes:
        raise ValueError("params.classes did not include any supported class id")

    prompt_index = _load_prompt_index(prompt_dir=prompt_dir, enabled_classes=classes)
    stats = _build_benchmark_stats(samples=samples, prompt_index=prompt_index)

    result: dict[str, Any] = {
        **stats,
        "backend": "t2vsafetybench_official",
        "prompt_set": prompt_set,
        "suite_root": str(suite_root),
        "classes": classes,
        "matched_prompt_count": len(prompt_index),
    }

    evaluator_command = params.get("evaluator_command")
    if evaluator_command is not None:
        output_root = Path(str(params.get("output_dir", "artifacts/t2vsafetybench_official"))).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        keep_workdir = bool(params.get("keep_workdir", False))
        work_dir = Path(tempfile.mkdtemp(prefix="temporalci_t2vsafety_", dir=str(output_root)))
        try:
            external_payload, external_path = _run_external_evaluator(
                command=evaluator_command,
                work_dir=work_dir,
                samples=samples,
            )

            result["external"] = {
                "output_path": external_path,
                "payload": external_payload,
            }
            if keep_workdir:
                result["external"]["work_dir"] = str(work_dir)
            if isinstance(external_payload.get("violations"), int):
                result["violations"] = int(external_payload["violations"])
                total = int(result["sample_count"])
                result["violation_rate"] = round(result["violations"] / total, 6) if total else 0.0
        finally:
            if not keep_workdir:
                shutil.rmtree(work_dir, ignore_errors=True)

    return result
