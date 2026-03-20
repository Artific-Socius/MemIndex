from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """
    Evidence payload for a question.

    Different benchmark types may use different evidence_type and payload keys.
    """

    evidence_type: str
    payload: dict[str, Any]

