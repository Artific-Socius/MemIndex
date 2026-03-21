"""LLM 评估器（同步版）。

使用 litellm 对 Agent 回复进行评分。
支持 binary / score / weighted_binary / multi_score 四种模式，
与 MemIndex 原始 LLMEvaluator 的评估语义保持一致。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import litellm


# ── 评估结果 ───────────────────────────────────────────────────


@dataclass
class EvalResult:
    """单项评估结果。"""

    score: float
    max_score: float
    reason: str


@dataclass
class BinaryItemResult:
    """加权二元评分中单项的评估结果。"""

    key: str
    weight: float
    passed: bool
    reason: str


@dataclass
class WeightedBinaryResult:
    """加权二元评分的整体结果。"""

    score: float
    max_score: float
    reason: str
    item_results: list[BinaryItemResult] = field(default_factory=list)


# ── Prompt 模板 ────────────────────────────────────────────────

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


# ── 评估函数 ───────────────────────────────────────────────────


def evaluate_binary(
    ground_truth: str,
    response: str,
    max_score: float,
    model: str,
) -> EvalResult:
    """二元评估：正确/错误。"""
    prompt = _BINARY_PROMPT.format(ground=ground_truth, target=response)
    raw = _llm_json_call(prompt, model)
    if raw is None or isinstance(raw, list):
        return EvalResult(0.0, max_score, "Evaluation failed")
    passed = raw.get("answer", False)
    return EvalResult(
        score=max_score if passed else 0.0,
        max_score=max_score,
        reason=str(raw.get("reason", "")),
    )


def evaluate_score(
    ground_truth: str,
    response: str,
    max_score: float,
    model: str,
) -> EvalResult:
    """0–1 连续分数评估。"""
    prompt = _SCORE_PROMPT.format(ground=ground_truth, target=response)
    raw_result = _llm_json_call(prompt, model)
    if raw_result is None or isinstance(raw_result, list):
        return EvalResult(0.0, max_score, "Evaluation failed")
    raw = max(0.0, min(1.0, float(raw_result.get("score", 0))))
    return EvalResult(
        score=raw * max_score,
        max_score=max_score,
        reason=f"[Score: {raw:.2f}] {raw_result.get('reason', '')}",
    )


def evaluate_multi_score(
    ground_truth: str,
    response: str,
    max_score: float,
    model: str,
) -> EvalResult:
    """多分数评估：答案包含多个评分点。"""
    prompt = _MULTI_SCORE_PROMPT.format(ground=ground_truth, target=response)
    raw = _llm_json_call(prompt, model, expect_list=True)
    if raw is None or not isinstance(raw, list):
        return EvalResult(0.0, max_score, "Evaluation failed")

    items: list[dict[str, Any]] = raw
    total = sum(
        max(0.0, min(1.0, float(r.get("score", 0)))) for r in items
    )
    score = min(total * max_score, max_score)
    reasons = [
        f"[{r.get('score', 0):.2f}] {r.get('reason', '')}" for r in items
    ]
    return EvalResult(score, max_score, "; ".join(reasons))


def evaluate_weighted_binary(
    binary_items: list[dict[str, Any]],
    response: str,
    max_score: float,
    model: str,
) -> WeightedBinaryResult:
    """加权二元评估：多个独立子项分别判断，按权重加总。"""
    item_results: list[BinaryItemResult] = []
    total_weight = sum(bi.get("weight", 0) for bi in binary_items)
    weighted = 0.0

    for bi in binary_items:
        r = evaluate_binary(bi["answer"], response, 1.0, model)
        passed = r.score > 0
        item_results.append(BinaryItemResult(
            key=bi.get("key", ""),
            weight=bi.get("weight", 0),
            passed=passed,
            reason=r.reason,
        ))
        if passed:
            weighted += bi.get("weight", 0)

    final = (
        (weighted / total_weight * max_score) if total_weight > 0 else 0.0
    )
    reasons = [
        f"[{'PASS' if ir.passed else 'FAIL'}] {ir.key}: {ir.reason}"
        for ir in item_results
    ]
    return WeightedBinaryResult(
        score=final,
        max_score=max_score,
        reason="\n".join(reasons),
        item_results=item_results,
    )


# ── 内部 LLM 调用 ─────────────────────────────────────────────

_MD_JSON = re.compile(r"```(?:json)?\s*([\[{][\s\S]+?[\]}])\s*```")


def _llm_json_call(
    prompt: str,
    model: str,
    max_tries: int = 3,
    expect_list: bool = False,
) -> dict[str, Any] | list[Any] | None:
    """调用 LLM 并解析 JSON 结果，失败自动重试。"""
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
