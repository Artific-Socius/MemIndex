"""Evaluator registry for Benchmark Lite.

Built-in evaluators:

- ``binary``: Binary (correct/incorrect) LLM evaluation
- ``score``: Continuous 0-1 LLM scoring
- ``multi_score``: Multi-point LLM scoring
- ``weighted_binary``: Weighted binary LLM evaluation
- ``keyword``: Simple keyword/substring matching (no LLM)
- ``benchmark_prompt``: Uses the benchmark's own eval_prompt template

Custom evaluators can be registered via::

    from benchmark_lite.evaluators import BaseEvaluator, register_evaluator

    @register_evaluator("my_custom_eval")
    class MyEvaluator(BaseEvaluator):
        def evaluate(self, question_text, ground_truth, response, ...):
            ...
"""

from .base import BaseEvaluator
from .registry import get_evaluator, list_evaluators, register_evaluator

from . import keyword as _keyword  # noqa: F401
from . import llm as _llm  # noqa: F401

__all__ = [
    "BaseEvaluator",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
]
