from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
