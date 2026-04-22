from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """
    Evidence payload for a question.

    Different benchmark types may use different evidence_type and payload keys.

    ``references`` is the canonical list of dataset-native ref strings (e.g. LoCoMo
    ``D1:3``). By default it must be non-empty; set ``allow_missing_references=True``
    only when the upstream dataset truly has no ref channel.
    """

    evidence_type: str
    payload: dict[str, Any]
    references: list[str] | None = None
    allow_missing_references: bool = False

    def __post_init__(self) -> None:
        if self.allow_missing_references:
            return
        if not self.references:
            raise ValueError(
                "EvidenceBundle must contain references unless "
                "allow_missing_references is True."
            )
