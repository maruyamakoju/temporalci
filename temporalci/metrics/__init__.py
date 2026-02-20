from __future__ import annotations

from importlib import import_module
from typing import Any
from typing import Callable
from typing import TypeAlias
from typing import cast

from temporalci.types import GeneratedSample

MetricFn: TypeAlias = Callable[..., dict[str, Any]]
MetricTarget: TypeAlias = str | MetricFn

_REGISTRY: dict[str, MetricTarget] = {
    "vbench_temporal": "temporalci.metrics.temporal:evaluate",
    "safety_t2v": "temporalci.metrics.safety:evaluate",
    "vbench_official": "temporalci.metrics.vbench_official:evaluate",
    "t2vsafetybench_official": "temporalci.metrics.t2vsafetybench_official:evaluate",
    "catenary_vegetation": "temporalci.metrics.catenary_vegetation:evaluate",
    "catenary_vegetation_ml": "temporalci.metrics.catenary_vegetation_ml:evaluate",
    "catenary_clearance": "temporalci.metrics.catenary_clearance:evaluate",
}


def register_metric(name: str, target: MetricTarget) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("metric name cannot be empty")
    _REGISTRY[key] = target


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


def _resolve_metric(target: MetricTarget) -> MetricFn:
    if isinstance(target, str):
        if ":" not in target:
            raise ValueError(f"invalid metric target '{target}'")
        module_name, fn_name = target.split(":", 1)
        module = import_module(module_name)
        resolved = getattr(module, fn_name)
    else:
        resolved = target

    if not callable(resolved):
        raise TypeError(f"metric target '{target}' is not callable")
    return cast(MetricFn, resolved)


def run_metric(
    name: str,
    samples: list[GeneratedSample],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_name = name.strip().lower()
    params = params or {}
    target = _REGISTRY.get(metric_name)
    if target is None:
        available = ", ".join(available_metrics())
        raise ValueError(f"unknown metric '{name}'. available metrics: {available}")

    fn = _resolve_metric(target)
    return fn(samples=samples, params=params)
