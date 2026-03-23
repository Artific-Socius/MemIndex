"""Evaluator registry: register and look up evaluators by eval_mode string."""

from __future__ import annotations

from typing import Any, Type

from .base import BaseEvaluator

_REGISTRY: dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(eval_mode: str):
    """Decorator to register an evaluator class for a given eval_mode.

    Usage::

        @register_evaluator("my_eval_mode")
        class MyEvaluator(BaseEvaluator):
            def evaluate(self, ...):
                ...
    """

    def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
        _REGISTRY[eval_mode] = cls
        return cls

    return decorator


def get_evaluator(eval_mode: str, **kwargs: Any) -> BaseEvaluator:
    """Create an evaluator instance for the given eval_mode.

    Keyword arguments are forwarded to the evaluator constructor.
    Unknown kwargs are silently absorbed by the base ``__init__``.

    Raises
    ------
    KeyError
        If no evaluator is registered for ``eval_mode``.
    """
    cls = _REGISTRY.get(eval_mode)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"No evaluator registered for eval_mode={eval_mode!r}. "
            f"Available: [{available}]"
        )
    return cls(**kwargs)


def list_evaluators() -> list[str]:
    """Return all registered eval_mode strings."""
    return sorted(_REGISTRY.keys())
