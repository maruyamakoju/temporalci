from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from temporalci.constants import DIRECTION_HIGHER_IS_BETTER
from temporalci.constants import DIRECTION_LOWER_IS_BETTER
from temporalci.constants import GATE_METHODS
from temporalci.constants import GATE_OPERATORS
from temporalci.errors import ConfigError
from temporalci.prompt_sources import expand_prompt_source
from temporalci.types import GateSpec, MetricSpec, ModelSpec, SuiteSpec, TestSpec
from temporalci.utils import as_bool, dedupe_prompts, resolve_path

# Backward-compatible alias so existing callers can still
# ``from temporalci.config import SuiteValidationError``.
SuiteValidationError = ConfigError


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------

def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{field_name}' must be a mapping")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{field_name}' must be a list")
    return value


def _require_non_empty_list(value: Any, field_name: str) -> list[Any]:
    items = _require_list(value, field_name)
    if not items:
        raise ConfigError(f"'{field_name}' cannot be empty")
    return items


def _coerce_int_list(values: list[Any], field_name: str) -> list[int]:
    coerced: list[int] = []
    for idx, item in enumerate(values):
        try:
            coerced.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"'{field_name}[{idx}]' must be an integer"
            ) from exc
    if not coerced:
        raise ConfigError(f"'{field_name}' cannot be empty")
    return coerced


def _coerce_str_list(values: list[Any], field_name: str) -> list[str]:
    coerced: list[str] = []
    for idx, item in enumerate(values):
        if not isinstance(item, str):
            raise ConfigError(f"'{field_name}[{idx}]' must be a string")
        value = item.strip()
        if not value:
            raise ConfigError(f"'{field_name}[{idx}]' cannot be empty")
        coerced.append(value)
    if not coerced:
        raise ConfigError(f"'{field_name}' cannot be empty")
    return coerced


# ---------------------------------------------------------------------------
# Init-image field normalization
# ---------------------------------------------------------------------------

def _normalize_init_image_fields(
    *,
    mapping: dict[str, Any],
    suite_dir: Path,
    field_prefix: str,
) -> dict[str, Any]:
    normalized = dict(mapping)

    init_image = normalized.get("init_image")
    if isinstance(init_image, str) and init_image.strip():
        normalized["init_image"] = str(resolve_path(init_image, suite_dir=suite_dir))

    init_images = normalized.get("init_images")
    if isinstance(init_images, list):
        resolved: list[str] = []
        for idx, item in enumerate(init_images):
            if not isinstance(item, str):
                raise ConfigError(
                    f"'{field_prefix}.init_images[{idx}]' must be a string path"
                )
            value = item.strip()
            if not value:
                continue
            resolved.append(str(resolve_path(value, suite_dir=suite_dir)))
        normalized["init_images"] = resolved

    return normalized


# ---------------------------------------------------------------------------
# Artifacts config parsing
# ---------------------------------------------------------------------------

_ALLOWED_VIDEO_POLICIES = {"all", "failures_only", "none"}


def _parse_artifacts(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    artifacts = _require_dict(raw, "artifacts")

    parsed: dict[str, Any] = {}
    video_policy = str(artifacts.get("video", "all")).strip().lower()
    if video_policy not in _ALLOWED_VIDEO_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_VIDEO_POLICIES))
        raise ConfigError(f"'artifacts.video' must be one of: {allowed}")
    parsed["video"] = video_policy

    if "max_samples" in artifacts:
        max_samples = int(artifacts["max_samples"])
        if max_samples <= 0:
            raise ConfigError("'artifacts.max_samples' must be > 0")
        parsed["max_samples"] = max_samples

    if "encode" in artifacts:
        encode = str(artifacts["encode"]).strip().lower()
        if encode not in {"h264", "h265"}:
            raise ConfigError("'artifacts.encode' must be 'h264' or 'h265'")
        parsed["encode"] = encode

    if "keep_workdir" in artifacts:
        parsed["keep_workdir"] = as_bool(artifacts["keep_workdir"], default=False)

    return parsed


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------

def load_suite(path: str | Path) -> SuiteSpec:
    """Load and validate a suite YAML file, returning a :class:`SuiteSpec`."""
    suite_path = Path(path)
    if not suite_path.exists():
        raise ConfigError(f"suite file does not exist: {suite_path}")
    suite_dir = suite_path.parent.resolve()

    with suite_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _require_dict(raw, "suite")

    version = int(root.get("version", 0))
    if version != 1:
        raise ConfigError(f"unsupported suite version: {version}")

    project = str(root.get("project", "")).strip()
    if not project:
        raise ConfigError("'project' is required")

    suite_name = str(root.get("suite_name", "")).strip()
    if not suite_name:
        raise ConfigError("'suite_name' is required")

    models = _parse_models(root, suite_dir)
    tests = _parse_tests(root, suite_dir)
    metrics = _parse_metrics(root)
    gates = _parse_gates(root)
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


def _parse_models(root: dict[str, Any], suite_dir: Path) -> list[ModelSpec]:
    raw_models = _require_non_empty_list(root.get("models"), "models")
    models: list[ModelSpec] = []
    seen: set[str] = set()
    for i, raw_model in enumerate(raw_models):
        model = _require_dict(raw_model, f"models[{i}]")
        name = str(model.get("name", "")).strip()
        adapter = str(model.get("adapter", "")).strip()
        if not name:
            raise ConfigError(f"'models[{i}].name' is required")
        if not adapter:
            raise ConfigError(f"'models[{i}].adapter' is required")
        if name in seen:
            raise ConfigError(f"duplicate model name: {name}")
        seen.add(name)
        params_raw = model.get("params", {})
        params = _require_dict(params_raw, f"models[{i}].params") if params_raw else {}
        params = _normalize_init_image_fields(
            mapping=params,
            suite_dir=suite_dir,
            field_prefix=f"models[{i}].params",
        )
        models.append(ModelSpec(name=name, adapter=adapter, params=params))
    return models


def _parse_tests(root: dict[str, Any], suite_dir: Path) -> list[TestSpec]:
    raw_tests = _require_non_empty_list(root.get("tests"), "tests")
    tests: list[TestSpec] = []
    seen: set[str] = set()
    supported_types = {"generation"}

    for i, raw_test in enumerate(raw_tests):
        test = _require_dict(raw_test, f"tests[{i}]")
        test_id = str(test.get("id", "")).strip()
        test_type = str(test.get("type", "")).strip() or "generation"
        if not test_id:
            raise ConfigError(f"'tests[{i}].id' is required")
        if test_type not in supported_types:
            allowed = ", ".join(sorted(supported_types))
            raise ConfigError(f"'tests[{i}].type' must be one of: {allowed}")
        if test_id in seen:
            raise ConfigError(f"duplicate test id: {test_id}")
        seen.add(test_id)

        prompts = _collect_prompts(test, i, suite_dir)
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
            TestSpec(id=test_id, type=test_type, prompts=prompts, seeds=seeds, video=video)
        )
    return tests


def _collect_prompts(test: dict[str, Any], index: int, suite_dir: Path) -> list[str]:
    prompts: list[str] = []
    if "prompts" in test and test.get("prompts") is not None:
        prompts.extend(
            _coerce_str_list(
                _require_list(test.get("prompts"), f"tests[{index}].prompts"),
                f"tests[{index}].prompts",
            )
        )
    if "prompt_source" in test and test.get("prompt_source") is not None:
        source = _require_dict(test.get("prompt_source"), f"tests[{index}].prompt_source")
        try:
            expanded = expand_prompt_source(source=source, suite_dir=suite_dir)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"'tests[{index}].prompt_source' invalid: {exc}") from exc
        prompts.extend(expanded)

    prompts = dedupe_prompts(prompts)
    if not prompts:
        raise ConfigError(
            f"'tests[{index}]' requires non-empty 'prompts' or valid 'prompt_source'"
        )
    return prompts


def _parse_metrics(root: dict[str, Any]) -> list[MetricSpec]:
    raw_metrics = _require_non_empty_list(root.get("metrics"), "metrics")
    metrics: list[MetricSpec] = []
    seen: set[str] = set()
    for i, raw_metric in enumerate(raw_metrics):
        metric = _require_dict(raw_metric, f"metrics[{i}]")
        name = str(metric.get("name", "")).strip()
        if not name:
            raise ConfigError(f"'metrics[{i}].name' is required")
        if name in seen:
            raise ConfigError(f"duplicate metric name: {name}")
        seen.add(name)
        params_raw = metric.get("params", {})
        params = _require_dict(params_raw, f"metrics[{i}].params") if params_raw else {}
        metrics.append(MetricSpec(name=name, params=params))
    return metrics


def _parse_gates(root: dict[str, Any]) -> list[GateSpec]:
    raw_gates = _require_non_empty_list(root.get("gates"), "gates")
    gates: list[GateSpec] = []
    for i, raw_gate in enumerate(raw_gates):
        gate = _require_dict(raw_gate, f"gates[{i}]")
        metric_path = str(gate.get("metric", "")).strip()
        op = str(gate.get("op", "")).strip()
        method = str(gate.get("method", "threshold")).strip().lower()
        if not metric_path:
            raise ConfigError(f"'gates[{i}].metric' is required")
        if not op:
            raise ConfigError(f"'gates[{i}].op' is required")
        if op not in GATE_OPERATORS:
            available = ", ".join(sorted(GATE_OPERATORS))
            raise ConfigError(f"'gates[{i}].op' must be one of: {available}")
        if method not in GATE_METHODS:
            available = ", ".join(sorted(GATE_METHODS))
            raise ConfigError(f"'gates[{i}].method' must be one of: {available}")
        if "value" not in gate:
            raise ConfigError(f"'gates[{i}].value' is required")
        params_raw = gate.get("params", {})
        params = _require_dict(params_raw, f"gates[{i}].params") if params_raw else {}
        if method == "sprt_regression":
            if op not in DIRECTION_HIGHER_IS_BETTER and op not in DIRECTION_LOWER_IS_BETTER:
                raise ConfigError(
                    f"'gates[{i}].op' must be one of >=, >, <=, < for method=sprt_regression"
                )
        gates.append(
            GateSpec(
                metric=metric_path,
                op=op,
                value=gate["value"],
                method=method,
                params=params,
            )
        )
    return gates


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_model(suite: SuiteSpec, name: str | None) -> ModelSpec:
    """Select a model from *suite* by *name*, defaulting to the first."""
    if name is None:
        return suite.models[0]
    for model in suite.models:
        if model.name == name:
            return model
    available = ", ".join(m.name for m in suite.models)
    raise ConfigError(f"model '{name}' not found. available: {available}")
