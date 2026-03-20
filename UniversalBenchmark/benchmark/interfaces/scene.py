from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .question import Question


class Scene(ABC):
    """
    Scene interface.

    A scene may contain multiple questions; each question can carry different evidence.
    """

    @property
    @abstractmethod
    def scene_id(self) -> str:
        raise NotImplementedError

    @property
    def task_type(self) -> Optional[str]:
        return None

    @property
    def scene_name(self) -> str | None:
        """Human-readable scenario name (e.g. corpus scale); None if N/A."""
        return None

    def background_text(self, *, max_chars: int | None = None, join_sep: str = "\n\n") -> str:
        """
        Full retrieval / haystack context for reasoning. Default: empty (no corpus).
        """
        return ""

    def question_count(self) -> int:
        return sum(1 for _ in self.questions())

    def get_question_by_id(self, question_id: str) -> Question:
        for q in self.questions():
            if q.question_id == question_id:
                return q
        raise KeyError(f"Unknown question_id {question_id!r}")

    @abstractmethod
    def questions(self) -> Iterable[Question]:
        raise NotImplementedError

