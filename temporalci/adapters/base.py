from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from temporalci.types import GeneratedSample


class ModelAdapter(ABC):
    def __init__(self, model_name: str, params: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.params = params or {}

    @abstractmethod
    def generate(
        self,
        *,
        test_id: str,
        prompt: str,
        seed: int,
        video_cfg: dict[str, Any],
        output_dir: Path,
    ) -> GeneratedSample:
        raise NotImplementedError
