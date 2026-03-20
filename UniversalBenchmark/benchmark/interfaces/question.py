from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .evidence import EvidenceBundle
from .scoring import ScoringConfig


@dataclass(frozen=True, slots=True)
class Question:
    """
    A question inside a scene.

    Each question can carry its own evidence and scoring configuration.
    """

    question_id: str
    question_text: str
    ground_truth: Any
    evidence: EvidenceBundle
    scoring: ScoringConfig
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def dependency_ids(self) -> Iterable[str]:
        return self.depends_on

