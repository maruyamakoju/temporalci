from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict


@dataclass(slots=True)
class ModelSpec:
    name: str
    adapter: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TestSpec:
    id: str
    type: str
    prompts: list[str]
    seeds: list[int]
    video: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateSpec:
    metric: str
    op: str
    value: Any
    method: str = "threshold"
    params: dict[str, Any] = field(default_factory=dict)
    window: int = 0  # rolling window size (0 = disabled)
    min_failures: int = 0  # gate fails only when this many of last `window` runs failed


@dataclass(slots=True)
class SuiteSpec:
    version: int
    project: str
    suite_name: str
    models: list[ModelSpec]
    tests: list[TestSpec]
    metrics: list[MetricSpec]
    gates: list[GateSpec]
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedSample:
    test_id: str
    prompt: str
    seed: int
    video_path: str
    evaluation_stream: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class RunResult(TypedDict):
    """Schema for the dict returned by :func:`temporalci.engine.run_suite`."""

    run_id: str
    timestamp_utc: str
    project: str
    suite_name: str
    model_name: str
    status: str  # "PASS" | "FAIL"
    sample_count: int
    skipped_count: int
    metrics: dict[str, Any]
    gates: list[dict[str, Any]]
    gate_failed: bool
    regressions: list[dict[str, Any]]
    regression_failed: bool
    baseline_run_id: NotRequired[str | None]
    baseline_mode: str
    artifacts_policy: dict[str, Any]
    samples: list[dict[str, Any]]
    run_dir: str
    git: NotRequired[dict[str, Any]]
    env: NotRequired[str]
