"""Base evaluator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from benchmark_lite.types import TurnScore


class BaseEvaluator(ABC):
    """Abstract base for all evaluators.

    Subclass and implement ``evaluate`` to create custom scoring logic.
    Register with ``@register_evaluator("my_mode")`` to make it
    available via ``ScoringConfig.eval_mode``.

    All evaluator constructors accept ``**kwargs`` so that the registry
    can forward common parameters (like ``model``) without breaking
    evaluators that do not need them.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        """Evaluate an agent response against a ground truth.

        Parameters
        ----------
        question_text:
            The original question asked.
        ground_truth:
            The reference/expected answer.
        response:
            The agent's actual response.
        max_score:
            Maximum possible score for this question.
        evidence:
            Optional evidence bundle from the data layer.

        Returns
        -------
        TurnScore
            Evaluation result with score, pass/fail, and details.
        """
        ...
