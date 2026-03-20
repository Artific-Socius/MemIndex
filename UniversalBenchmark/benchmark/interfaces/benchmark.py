from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .scene import Scene


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.

    A benchmark is responsible for producing scenes. It does NOT need to implement
    LLM calling or scoring execution; that can be handled by the runtime layer.
    """

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        raise NotImplementedError

    @property
    def eval_prompt(self) -> str:
        """
        Default LLM judge / scoring prompt text for this benchmark (per-benchmark).
        """
        return ""

    def list_scenes(self) -> Iterable[str]:
        """
        Optionally override to enumerate available scene ids.
        """

        raise NotImplementedError

    @abstractmethod
    def get_scene(self, scene_id: str) -> Scene:
        raise NotImplementedError

    def get_scene_by_id(self, scene_id: str) -> Scene:
        """Preferred entrypoint: resolve scene by id within this benchmark."""
        return self.get_scene(scene_id)

    def __call__(self, scene_id: str) -> Scene:
        return self.get_scene_by_id(scene_id)

    def sample_scene(self, scene_id: Optional[str] = None) -> Scene:
        """
        Default sampling strategy: if scene_id is provided, return that scene.
        Otherwise require list_scenes() to be implemented.
        """

        if scene_id is not None:
            return self.get_scene_by_id(scene_id)

        ids = list(self.list_scenes())
        if not ids:
            raise ValueError(f"No scenes available for benchmark '{self.benchmark_name}'.")
        # Simple deterministic selection; callers can implement randomness externally.
        return self.get_scene_by_id(ids[0])

