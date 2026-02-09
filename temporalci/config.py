from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from temporalci.constants import GATE_OPERATORS
from temporalci.prompt_sources import PromptSourceError
from temporalci.prompt_sources import expand_prompt_source
from temporalci.types import GateSpec
from temporalci.types import MetricSpec
from temporalci.types import ModelSpec
from temporalci.types import SuiteSpec
from temporalci.types import TestSpec


class SuiteValidationError(ValueError):
    pass


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteValidationError(f"'{field_name}' must be a mapping")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SuiteValidationError(f"'{field_name}' must be a list")
    return value


def _require_non_empty_list(value: Any, field_name: str) -> list[Any]:
    items = _require_list(value, field_name)
    if not items:
        raise SuiteValidationError(f"'{field_name}' cannot be empty")
    return items


def _coerce_int_list(values: list[Any], field_name: str) -> list[int]:
    coerced: list[int] = []
    for idx, item in enumerate(values):
        try:
            coerced.append(int(item))
        except (TypeError, ValueError) as exc:
            raise SuiteValidationError(
                f"'{field_name}[{idx}]' must be an integer"
            ) from exc
    if not coerced:
        raise SuiteValidationError(f"'{field_name}' cannot be empty")
    return coerced


def _coerce_str_list(values: list[Any], field_name: str) -> list[str]:
    coerced: list[str] = []
    for idx, item in enumerate(values):
        if not isinstance(item, str):
            raise SuiteValidationError(f"'{field_name}[{idx}]' must be a string")
        value = item.strip()
        if not value:
            raise SuiteValidationError(f"'{field_name}[{idx}]' cannot be empty")
        coerced.append(value)
    if not coerced:
        raise SuiteValidationError(f"'{field_name}' cannot be empty")
    return coerced


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized = " ".join(prompt.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(prompt)
    return unique


def _resolve_path_like(raw_path: str, *, suite_dir: Path) -> str:
    path = Path(raw_path.strip())
    if not path.is_absolute():
        suite_candidate = (suite_dir / path).resolve()
        cwd_candidate = (Path.cwd() / path).resolve()
        if suite_candidate.exists():
            path = suite_candidate
        elif cwd_candidate.exists():
            path = cwd_candidate
        else:
            path = suite_candidate
    return str(path)


def _normalize_init_image_fields(
    *,
    mapping: dict[str, Any],
    suite_dir: Path,
    field_prefix: str,
) -> dict[str, Any]:
    normalized = dict(mapping)

    init_image = normalized.get("init_image")
    if isinstance(init_image, str) and init_image.strip():
        normalized["init_image"] = _resolve_path_like(init_image, suite_dir=suite_dir)

    init_images = normalized.get("init_images")
    if isinstance(init_images, list):
        resolved: list[str] = []
        for idx, item in enumerate(init_images):
            if not isinstance(item, str):
                raise SuiteValidationError(
                    f"'{field_prefix}.init_images[{idx}]' must be a string path"
                )
            value = item.strip()
            if not value:
                continue
            resolved.append(_resolve_path_like(value, suite_dir=suite_dir))
        normalized["init_images"] = resolved

    return normalized


def _parse_artifacts(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    artifacts = _require_dict(raw, "artifacts")

    parsed: dict[str, Any] = {}
    video_policy = str(artifacts.get("video", "all")).strip().lower()
    allowed_video_policies = {"all", "failures_only", "none"}
    if video_policy not in allowed_video_policies:
        allowed = ", ".join(sorted(allowed_video_policies))
        raise SuiteValidationError(f"'artifacts.video' must be one of: {allowed}")
    parsed["video"] = video_policy

    if "max_samples" in artifacts:
        max_samples = int(artifacts["max_samples"])
        if max_samples <= 0:
            raise SuiteValidationError("'artifacts.max_samples' must be > 0")
        parsed["max_samples"] = max_samples

    if "encode" in artifacts:
        encode = str(artifacts["encode"]).strip().lower()
        if encode not in {"h264", "h265"}:
            raise SuiteValidationError("'artifacts.encode' must be 'h264' or 'h265'")
        parsed["encode"] = encode

    if "keep_workdir" in artifacts:
        parsed["keep_workdir"] = bool(artifacts["keep_workdir"])

    return parsed


def load_suite(path: str | Path) -> SuiteSpec:
    suite_path = Path(path)
    if not suite_path.exists():
        raise SuiteValidationError(f"suite file does not exist: {suite_path}")
    suite_dir = suite_path.parent.resolve()

    with suite_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _require_dict(raw, "suite")

    version = int(root.get("version", 0))
    if version != 1:
        raise SuiteValidationError(f"unsupported suite version: {version}")

    project = str(root.get("project", "")).strip()
    if not project:
        raise SuiteValidationError("'project' is required")

    suite_name = str(root.get("suite_name", "")).strip()
    if not suite_name:
        raise SuiteValidationError("'suite_name' is required")

    raw_models = _require_non_empty_list(root.get("models"), "models")
    models: list[ModelSpec] = []
    seen_models: set[str] = set()
    for i, raw_model in enumerate(raw_models):
        model = _require_dict(raw_model, f"models[{i}]")
        name = str(model.get("name", "")).strip()
        adapter = str(model.get("adapter", "")).strip()
        if not name:
            raise SuiteValidationError(f"'models[{i}].name' is required")
        if not adapter:
            raise SuiteValidationError(f"'models[{i}].adapter' is required")
        if name in seen_models:
            raise SuiteValidationError(f"duplicate model name: {name}")
        seen_models.add(name)
        params_raw = model.get("params", {})
        params = _require_dict(params_raw, f"models[{i}].params") if params_raw else {}
        params = _normalize_init_image_fields(
            mapping=params,
            suite_dir=suite_dir,
            field_prefix=f"models[{i}].params",
        )
        models.append(ModelSpec(name=name, adapter=adapter, params=params))

    raw_tests = _require_non_empty_list(root.get("tests"), "tests")
    tests: list[TestSpec] = []
    seen_tests: set[str] = set()
    supported_test_types = {"generation"}
    for i, raw_test in enumerate(raw_tests):
        test = _require_dict(raw_test, f"tests[{i}]")
        test_id = str(test.get("id", "")).strip()
        test_type = str(test.get("type", "")).strip() or "generation"
        if not test_id:
            raise SuiteValidationError(f"'tests[{i}].id' is required")
        if test_type not in supported_test_types:
            allowed = ", ".join(sorted(supported_test_types))
            raise SuiteValidationError(
                f"'tests[{i}].type' must be one of: {allowed}"
            )
        if test_id in seen_tests:
            raise SuiteValidationError(f"duplicate test id: {test_id}")
        seen_tests.add(test_id)

        prompts: list[str] = []
        if "prompts" in test and test.get("prompts") is not None:
            prompts.extend(
                _coerce_str_list(
                    _require_list(test.get("prompts"), f"tests[{i}].prompts"),
                    f"tests[{i}].prompts",
                )
            )
        if "prompt_source" in test and test.get("prompt_source") is not None:
            source = _require_dict(test.get("prompt_source"), f"tests[{i}].prompt_source")
            try:
                expanded = expand_prompt_source(source=source, suite_dir=suite_dir)
            except PromptSourceError as exc:
                raise SuiteValidationError(f"'tests[{i}].prompt_source' invalid: {exc}") from exc
            prompts.extend(expanded)

        prompts = _dedupe_prompts(prompts)
        if not prompts:
            raise SuiteValidationError(
                f"'tests[{i}]' requires non-empty 'prompts' or valid 'prompt_source'"
            )

        seeds_raw = test.get("seeds", [0])
        seeds = _coerce_int_list(
            _require_list(seeds_raw, f"tests[{i}].seeds"),
            f"tests[{i}].seeds",
        )
        video_raw = test.get("video", {})
        video = _require_dict(video_raw, f"tests[{i}].video") if video_raw else {}
        video = _normalize_init_image_fields(
            mapping=video,
            suite_dir=suite_dir,
            field_prefix=f"tests[{i}].video",
        )
        tests.append(
            TestSpec(
                id=test_id,
                type=test_type,
                prompts=prompts,
                seeds=seeds,
                video=video,
            )
        )

    raw_metrics = _require_non_empty_list(root.get("metrics"), "metrics")
    metrics: list[MetricSpec] = []
    seen_metric_names: set[str] = set()
    for i, raw_metric in enumerate(raw_metrics):
        metric = _require_dict(raw_metric, f"metrics[{i}]")
        name = str(metric.get("name", "")).strip()
        if not name:
            raise SuiteValidationError(f"'metrics[{i}].name' is required")
        if name in seen_metric_names:
            raise SuiteValidationError(f"duplicate metric name: {name}")
        seen_metric_names.add(name)
        params_raw = metric.get("params", {})
        params = _require_dict(params_raw, f"metrics[{i}].params") if params_raw else {}
        metrics.append(MetricSpec(name=name, params=params))

    raw_gates = _require_non_empty_list(root.get("gates"), "gates")
    gates: list[GateSpec] = []
    for i, raw_gate in enumerate(raw_gates):
        gate = _require_dict(raw_gate, f"gates[{i}]")
        metric_path = str(gate.get("metric", "")).strip()
        op = str(gate.get("op", "")).strip()
        if not metric_path:
            raise SuiteValidationError(f"'gates[{i}].metric' is required")
        if not op:
            raise SuiteValidationError(f"'gates[{i}].op' is required")
        if op not in GATE_OPERATORS:
            available = ", ".join(sorted(GATE_OPERATORS))
            raise SuiteValidationError(
                f"'gates[{i}].op' must be one of: {available}"
            )
        if "value" not in gate:
            raise SuiteValidationError(f"'gates[{i}].value' is required")
        gates.append(GateSpec(metric=metric_path, op=op, value=gate["value"]))

    artifacts = _parse_artifacts(root.get("artifacts"))

    return SuiteSpec(
        version=version,
        project=project,
        suite_name=suite_name,
        models=models,
        tests=tests,
        metrics=metrics,
        gates=gates,
        artifacts=artifacts,
    )


def select_model(suite: SuiteSpec, name: str | None) -> ModelSpec:
    if name is None:
        return suite.models[0]
    for model in suite.models:
        if model.name == name:
            return model
    available = ", ".join(m.name for m in suite.models)
    raise SuiteValidationError(f"model '{name}' not found. available: {available}")
