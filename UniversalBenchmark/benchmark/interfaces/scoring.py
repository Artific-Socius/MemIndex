from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

BuiltinEvalMode = Literal[
    "binary", "score", "weighted_binary", "multi_score", "keyword",
]


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """
    Scoring configuration for a question.

    ``eval_mode`` is a plain string so that custom evaluators registered
    via ``@register_evaluator`` can use arbitrary mode names.  The
    built-in modes are listed in :data:`BuiltinEvalMode`.

    The runtime (runner/evaluator) can map eval_prompt_key to a prompt template.
    """

    eval_mode: str
    eval_prompt_key: str
    max_score: float = 1.0
    post_process: Optional[str] = None

