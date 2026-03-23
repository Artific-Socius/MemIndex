"""LLM-based evaluators using litellm.

Registered modes: ``binary``, ``score``, ``multi_score``,
``weighted_binary``, ``benchmark_prompt``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm

from benchmark_lite.types import TurnScore

from .base import BaseEvaluator
from .registry import register_evaluator

# ── Prompt templates ────────────────────────────────────────────

_BINARY_PROMPT = """\
You are an evaluation system. Given the standard answer and the target \
response, determine if the target response correctly addresses the \
standard answer.

Standard Answer: {ground}
Target Response: {target}

Output your result in JSON format:
{{"answer": true/false, "reason": "brief explanation"}}

Only output the JSON, nothing else."""

_SCORE_PROMPT = """\
You are an evaluation system. Given the standard answer and the target \
response, rate how well the target response matches on a scale of \
0.0 to 1.0.

Standard Answer: {ground}
Target Response: {target}

Output your result in JSON format:
{{"score": 0.0-1.0, "reason": "brief explanation"}}

Only output the JSON, nothing else."""

_MULTI_SCORE_PROMPT = """\
You are an evaluation system. The standard answer contains multiple \
points. For each point, evaluate if the target response covers it \
and give a score between 0.0 and 1.0.

Standard Answer: {ground}
Target Response: {target}

Output your result as a JSON array:
[{{"score": 0.0-1.0, "reason": "..."}}, ...]

Only output the JSON array, nothing else."""

# ── Shared LLM call utility ───────────────────────────────────

_MD_JSON = re.compile(r"```(?:json)?\s*([\[{][\s\S]+?[\]}])\s*```")


def _llm_raw_call(prompt: str, model: str, max_tries: int = 3) -> str | None:
    """Call LLM and return the raw text response."""
    for _ in range(max_tries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            continue
    return None


def _llm_json_call(
    prompt: str,
    model: str,
    max_tries: int = 3,
    expect_list: bool = False,
) -> dict[str, Any] | list[Any] | None:
    for _ in range(max_tries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            md = _MD_JSON.search(text)
            raw_text = md.group(1) if md else text
            parsed: dict[str, Any] | list[Any] = json.loads(raw_text)
            if expect_list and not isinstance(parsed, list):
                continue
            return parsed
        except (json.JSONDecodeError, Exception):
            continue
    return None


# ── Evaluator implementations ─────────────────────────────────

_DEFAULT_MODEL = "openrouter/google/gemini-2.5-flash"


@register_evaluator("binary")
class BinaryEvaluator(BaseEvaluator):
    """Binary (correct/incorrect) evaluation via LLM judge."""

    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        prompt = _BINARY_PROMPT.format(
            ground=str(ground_truth), target=response,
        )
        raw = _llm_json_call(prompt, self._model)
        if raw is None or isinstance(raw, list):
            return TurnScore(0.0, False, "Evaluation failed")
        passed = bool(raw.get("answer", False))
        reason = str(raw.get("reason", ""))
        return TurnScore(
            score=1.0 if passed else 0.0,
            passed=passed,
            detail=f"[binary] {reason}",
        )


@register_evaluator("score")
class ScoreEvaluator(BaseEvaluator):
    """Continuous 0-1 scoring via LLM judge.

    If a ``prompt_template`` is provided (from the benchmark's own
    ``eval_prompt``), it is used instead of the built-in prompt.
    The template should contain ``{question}``, ``{reference}``, and
    ``{model_answer}`` placeholders.  When a custom template is used
    the response is parsed as binary (True/False) since most benchmark
    prompts use that format.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._custom_template = prompt_template

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        if self._custom_template:
            return _eval_with_benchmark_prompt(
                self._custom_template, question_text,
                str(ground_truth), response, self._model,
            )

        prompt = _SCORE_PROMPT.format(
            ground=str(ground_truth), target=response,
        )
        raw = _llm_json_call(prompt, self._model)
        if raw is None or isinstance(raw, list):
            return TurnScore(0.0, False, "Evaluation failed")
        norm = max(0.0, min(1.0, float(raw.get("score", 0))))
        reason = str(raw.get("reason", ""))
        return TurnScore(
            score=norm,
            passed=norm >= 0.5,
            detail=f"[score: {norm:.2f}] {reason}",
        )


@register_evaluator("multi_score")
class MultiScoreEvaluator(BaseEvaluator):
    """Multi-point scoring via LLM judge."""

    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        prompt = _MULTI_SCORE_PROMPT.format(
            ground=str(ground_truth), target=response,
        )
        raw = _llm_json_call(prompt, self._model, expect_list=True)
        if raw is None or not isinstance(raw, list):
            return TurnScore(0.0, False, "Evaluation failed")
        items: list[dict[str, Any]] = raw
        total = sum(
            max(0.0, min(1.0, float(r.get("score", 0)))) for r in items
        )
        count = len(items) or 1
        norm = min(total / count, 1.0)
        reasons = [
            f"[{r.get('score', 0):.2f}] {r.get('reason', '')}"
            for r in items
        ]
        return TurnScore(
            score=norm,
            passed=norm >= 0.5,
            detail=f"[multi_score] {'; '.join(reasons)}",
        )


@register_evaluator("weighted_binary")
class WeightedBinaryEvaluator(BaseEvaluator):
    """Weighted binary evaluation via LLM judge.

    Expects ``evidence`` to contain binary items with ``key``,
    ``weight``, and ``answer`` fields — either as an
    ``EvidenceBundle`` with ``payload["binary_items"]`` or as a
    plain list of dicts.  Falls back to simple binary evaluation
    when no items are found.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = model

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        binary_items: list[dict[str, Any]] = []
        if evidence and hasattr(evidence, "payload"):
            binary_items = evidence.payload.get("binary_items", [])
        if not binary_items and isinstance(evidence, list):
            binary_items = evidence

        if not binary_items:
            prompt = _BINARY_PROMPT.format(
                ground=str(ground_truth), target=response,
            )
            raw = _llm_json_call(prompt, self._model)
            if raw is None or isinstance(raw, list):
                return TurnScore(0.0, False, "Evaluation failed")
            passed = bool(raw.get("answer", False))
            return TurnScore(
                score=1.0 if passed else 0.0,
                passed=passed,
                detail=f"[weighted_binary fallback] {raw.get('reason', '')}",
            )

        total_weight = sum(bi.get("weight", 0) for bi in binary_items)
        weighted = 0.0
        reasons: list[str] = []

        for bi in binary_items:
            prompt = _BINARY_PROMPT.format(
                ground=bi["answer"], target=response,
            )
            raw = _llm_json_call(prompt, self._model)
            if raw and not isinstance(raw, list):
                passed = bool(raw.get("answer", False))
                reason = str(raw.get("reason", ""))
            else:
                passed = False
                reason = "eval failed"
            if passed:
                weighted += bi.get("weight", 0)
            mark = "PASS" if passed else "FAIL"
            reasons.append(f"[{mark}] {bi.get('key', '?')}: {reason}")

        final = (weighted / total_weight) if total_weight > 0 else 0.0
        return TurnScore(
            score=final,
            passed=final >= 0.5,
            detail=f"[weighted_binary] {'; '.join(reasons)}",
        )


# ── Benchmark custom-prompt helper ────────────────────────────

_TRUE_FALSE_RE = re.compile(r"^\s*(True|False)\b", re.IGNORECASE)


def _eval_with_benchmark_prompt(
    template: str,
    question_text: str,
    ground_truth: str,
    response: str,
    model: str,
) -> TurnScore:
    """Evaluate using a benchmark-supplied prompt template.

    The template is expected to contain ``{question}``, ``{reference}``,
    and ``{model_answer}`` placeholders.  The LLM response is parsed
    for a leading ``True`` / ``False`` token.
    """
    try:
        prompt = template.format(
            question=question_text,
            reference=ground_truth,
            model_answer=response,
        )
    except KeyError:
        prompt = template.format_map({
            "question": question_text,
            "reference": ground_truth,
            "model_answer": response,
            "ground": ground_truth,
            "target": response,
        })

    raw_text = _llm_raw_call(prompt, model)
    if raw_text is None:
        return TurnScore(0.0, False, "Evaluation failed (no response)")

    m = _TRUE_FALSE_RE.match(raw_text)
    if m:
        passed = m.group(1).lower() == "true"
        rationale = raw_text[m.end():].strip().lstrip(".,;:-– ")
        return TurnScore(
            score=1.0 if passed else 0.0,
            passed=passed,
            detail=f"[benchmark_prompt] {rationale[:200]}",
        )

    lower = raw_text.strip().lower()
    passed = lower.startswith("true") or "correct" in lower[:60]
    return TurnScore(
        score=1.0 if passed else 0.0,
        passed=passed,
        detail=f"[benchmark_prompt] {raw_text[:200]}",
    )


@register_evaluator("benchmark_prompt")
class BenchmarkPromptEvaluator(BaseEvaluator):
    """Evaluator that uses the benchmark's own eval_prompt template.

    The template should contain ``{question}``, ``{reference}``, and
    ``{model_answer}`` placeholders.  The LLM response is parsed for
    a leading ``True`` / ``False`` token.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        if not prompt_template:
            raise ValueError(
                "BenchmarkPromptEvaluator requires a non-empty "
                "prompt_template (from benchmark.eval_prompt)"
            )
        self._template = prompt_template

    def evaluate(
        self,
        question_text: str,
        ground_truth: Any,
        response: str,
        max_score: float = 1.0,
        evidence: Any = None,
    ) -> TurnScore:
        return _eval_with_benchmark_prompt(
            self._template, question_text,
            str(ground_truth), response, self._model,
        )
