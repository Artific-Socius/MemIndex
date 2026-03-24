from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

from .question import Question


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A single turn in a structured conversation history.

    Used by Scene implementations to provide pre-existing dialog context
    for memory benchmarks that test conversational recall.  The adapter
    layer converts these into ``preload_history`` entries so the Agent's
    memory is populated before evaluation questions are asked.
    """

    user_message: str
    assistant_response: str


class Scene(ABC):
    """
    Scene interface.

    A scene may contain multiple questions; each question can carry different evidence.

    Scenes can provide context in two ways (or both):

    - ``background_text()`` — a single text blob (corpus / haystack).
    - ``conversation_history()`` — structured multi-turn dialog history.

    The adapter layer inspects both to build the execution scenario.
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

    def background_documents(self) -> list[str]:
        """Individual corpus documents for memory backends that support
        file-level import (e.g. Memecho ``import_file_fast``).

        Default: wraps ``background_text()`` in a single-element list,
        or returns ``[]`` if there is no background text.
        """
        bg = self.background_text()
        return [bg] if bg else []

    def conversation_history(self) -> list[ConversationTurn]:
        """Structured conversation history for memory-based benchmarks.

        Returns a sequence of user-assistant exchanges that represent
        prior conversation context.  The adapter layer converts these
        into ``preload_history`` entries so the Agent's memory is
        populated before evaluation questions are asked.

        Default: empty list (scene uses ``background_text`` or
        questions only).
        """
        return []

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

