from __future__ import annotations

from importlib import import_module
from typing import TypeAlias

from temporalci.adapters.base import ModelAdapter
from temporalci.types import ModelSpec

AdapterTarget: TypeAlias = str | type[ModelAdapter]

_REGISTRY: dict[str, AdapterTarget] = {
    "mock": "temporalci.adapters.mock:MockAdapter",
    "http": "temporalci.adapters.http:HttpAdapter",
    "diffusers_img2vid": "temporalci.adapters.diffusers_img2vid:DiffusersImg2VidAdapter",
    "frame_archive": "temporalci.adapters.frame_archive:FrameArchiveAdapter",
}


def register_adapter(name: str, target: AdapterTarget) -> None:
    key = name.strip()
    if not key:
        raise ValueError("adapter name cannot be empty")
    _REGISTRY[key] = target


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _resolve_adapter(target: AdapterTarget) -> type[ModelAdapter]:
    if isinstance(target, str):
        if ":" not in target:
            raise ValueError(f"invalid adapter target '{target}'")
        module_name, class_name = target.split(":", 1)
        module = import_module(module_name)
        resolved = getattr(module, class_name)
    else:
        resolved = target

    if not isinstance(resolved, type) or not issubclass(resolved, ModelAdapter):
        raise TypeError(f"adapter target '{target}' is not a ModelAdapter class")
    return resolved


def build_adapter(model: ModelSpec) -> ModelAdapter:
    target = _REGISTRY.get(model.adapter)
    if target is None:
        available = ", ".join(available_adapters())
        raise ValueError(
            f"unknown adapter '{model.adapter}' for model '{model.name}'. "
            f"available adapters: {available}"
        )

    adapter_cls = _resolve_adapter(target)
    return adapter_cls(model_name=model.name, params=model.params)
