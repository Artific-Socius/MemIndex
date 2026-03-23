"""Simple keyword/substring evaluator (no LLM required).

Registered mode: ``keyword``.
"""

from __future__ import annotations

from typing import Any

from benchmark_lite.types import TurnScore

from .base import BaseEvaluator
from .registry import register_evaluator


@register_evaluator("keyword")
class KeywordEvaluator(BaseEvaluator):
    """Evaluate by checking if keywords from ground_truth appear in the response.

    ``ground_truth`` can be:

    - A single string: checks substring match (case-insensitive).
    - A list of strings: checks if ALL appear in the response.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        response_lower = response.lower()

        if isinstance(ground_truth, str):
            keywords = [ground_truth]
        elif isinstance(ground_truth, (list, tuple)):
            keywords = [str(k) for k in ground_truth]
        else:
            keywords = [str(ground_truth)]

        matched: list[str] = []
        missed: list[str] = []
        for kw in keywords:
            if kw.lower() in response_lower:
                matched.append(kw)
            else:
                missed.append(kw)

        total = len(keywords) or 1
        score = len(matched) / total
        passed = len(missed) == 0

        detail_parts: list[str] = []
        if matched:
            detail_parts.append(f"matched: {matched}")
        if missed:
            detail_parts.append(f"missed: {missed}")

        return TurnScore(
            score=score,
            passed=passed,
            detail=f"[keyword] {'; '.join(detail_parts)}",
        )
