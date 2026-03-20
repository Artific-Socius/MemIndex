from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

EvalMode = Literal["binary", "score", "weighted_binary", "multi_score"]


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """
    Scoring configuration for a question.

    The runtime (runner/evaluator) can map eval_prompt_key to a prompt template.
    """

    eval_mode: EvalMode
    eval_prompt_key: str
    max_score: float = 1.0
    post_process: Optional[str] = None

