"""结果格式化与导出工具。"""

from __future__ import annotations

import json
from typing import Any

from .types import (
    AggregateResult,
    BenchmarkResult,
    ScenarioResult,
    ScenarioScore,
    TurnResult,
    TurnType,
)

_SEP = "=" * 60
_SUB_SEP = "-" * 60


def format_report(
    result: BenchmarkResult,
    *,
    verbose: bool = False,
) -> str:
    """将 BenchmarkResult 格式化为人类可读的文本报告。

    Parameters
    ----------
    result:
        Benchmark 运行结果。
    verbose:
        若为 True，输出每个回合的详细信息。
    """
    lines: list[str] = [""]
    lines.append(_SEP)
    lines.append("  Benchmark Lite Report")
    lines.append(_SEP)
    lines.append(f"  Benchmark : {result.benchmark_name}")
    lines.append(f"  Agent     : {result.agent_identifier}")
    lines.append(f"  Timestamp : {result.timestamp}")
    lines.append(_SEP)
    lines.append("")

    if verbose:
        for sr in result.scenario_results:
            _format_scenario(sr, lines)

    lines.append(f"  {_SUB_SEP}")
    lines.append("  Aggregate Results")
    lines.append(f"  {_SUB_SEP}")
    _format_aggregate(result.aggregate, lines)
    lines.append(_SEP)
    lines.append("")

    return "\n".join(lines)


def _format_scenario(
    sr: ScenarioResult, lines: list[str],
) -> None:
    """格式化单个场景的详细信息。"""
    lines.append(f"  Scenario: {sr.scenario_id}")
    if sr.scenario_description:
        lines.append(f"  {sr.scenario_description}")
    lines.append(f"  {_SUB_SEP}")

    annotation_map = _build_annotation_map(sr.scenario_score)

    for tr in sr.turn_results:
        _format_turn(tr, annotation_map, lines)

    if sr.scenario_score:
        ss = sr.scenario_score
        mark = "PASS" if ss.passed else "FAIL"
        lines.append(f"  [SCENARIO] {mark} (score={ss.score:.2f})")
        if ss.detail:
            lines.append(f"  Detail: {ss.detail}")

    lines.append("")


def _build_annotation_map(
    scenario_score: ScenarioScore | None,
) -> dict[int, tuple[str, Any]]:
    """从 ScenarioScore 构建 turn_index -> (label, score) 查找表。"""
    if not scenario_score or not scenario_score.turn_annotations:
        return {}
    return {
        ann.turn_index: (ann.label, ann.score)
        for ann in scenario_score.turn_annotations
    }


def _format_turn(
    tr: TurnResult,
    annotation_map: dict[int, tuple[str, Any]],
    lines: list[str],
) -> None:
    """格式化单个回合。"""
    tag = "EVAL" if tr.turn_type == TurnType.EVALUATION else "CONV"

    score_str = ""
    if tr.score is not None:
        mark = "PASS" if tr.score.passed else "FAIL"
        score_str = f"  [{mark} {tr.score.score:.2f}]"

    ann_str = _format_annotation(tr.turn_index, annotation_map)

    user_preview = _truncate(tr.user_input, 60)
    resp_preview = _truncate(tr.response, 60)

    lines.append(
        f"  [{tag}] Turn {tr.turn_index}: "
        f'"{user_preview}"{score_str}{ann_str}'
    )
    lines.append(f'         -> "{resp_preview}"')

    if tr.score and tr.score.detail:
        lines.append(f"         Detail: {tr.score.detail}")


def _format_annotation(
    turn_index: int,
    annotation_map: dict[int, tuple[str, Any]],
) -> str:
    """格式化回合的事后标注。"""
    if turn_index not in annotation_map:
        return ""
    label, ann_score = annotation_map[turn_index]
    result = f"  <{label}>"
    if ann_score is not None:
        mark = "PASS" if ann_score.passed else "FAIL"
        result += f" [{mark} {ann_score.score:.2f}]"
    return result


def _format_aggregate(agg: AggregateResult, lines: list[str]) -> None:
    """格式化 AggregateResult 的固定字段 + extra。"""
    pct = agg.score * 100
    lines.append(
        f"  Score       : {pct:.2f}%"
        f"  ({agg.total_score:.4f} / {agg.total_max_score:.4f})"
    )
    if agg.total > 0:
        pass_pct = agg.passed / agg.total * 100
        lines.append(
            f"  Evaluations : {agg.passed} / {agg.total} passed"
            f"  ({pass_pct:.2f}%)"
        )
    else:
        lines.append("  Evaluations : (none)")

    if agg.detail:
        lines.append(f"  Detail      : {agg.detail}")

    if agg.extra:
        for key, value in agg.extra.items():
            if isinstance(value, float):
                lines.append(f"  {key:<12s}: {value:.4f}")
            else:
                lines.append(f"  {key:<12s}: {value}")


# ── 序列化 ──────────────────────────────────────────────────────


def _aggregate_to_dict(agg: AggregateResult) -> dict[str, Any]:
    """将 AggregateResult 序列化为 dict。"""
    d: dict[str, Any] = {
        "score": agg.score,
        "total_score": agg.total_score,
        "total_max_score": agg.total_max_score,
        "total": agg.total,
        "passed": agg.passed,
    }
    if agg.detail:
        d["detail"] = agg.detail
    if agg.extra:
        d["extra"] = agg.extra
    return d


def to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """将 BenchmarkResult 转换为可序列化的 dict。"""
    return {
        "benchmark_name": result.benchmark_name,
        "agent_identifier": result.agent_identifier,
        "timestamp": result.timestamp,
        "aggregate": _aggregate_to_dict(result.aggregate),
        "metadata": result.metadata,
        "scenarios": [
            _scenario_result_to_dict(sr)
            for sr in result.scenario_results
        ],
    }


def to_json(result: BenchmarkResult, *, indent: int = 2) -> str:
    """将 BenchmarkResult 转换为 JSON 字符串。"""
    return json.dumps(to_dict(result), indent=indent, ensure_ascii=False)


# ── 工具 ────────────────────────────────────────────────────────


def _scenario_result_to_dict(sr: Any) -> dict[str, Any]:
    """将单个 ScenarioResult 转换为 dict。"""
    d: dict[str, Any] = {
        "id": sr.scenario_id,
        "description": sr.scenario_description,
        "turns": [
            {
                "index": tr.turn_index,
                "type": tr.turn_type.value,
                "user_input": tr.user_input,
                "response": tr.response,
                "score": (
                    _score_to_dict(tr.score)
                    if tr.score
                    else None
                ),
            }
            for tr in sr.turn_results
        ],
    }

    if sr.scenario_score is not None:
        ss = sr.scenario_score
        d["scenario_score"] = {
            "score": ss.score,
            "passed": ss.passed,
            "detail": ss.detail,
            "metadata": ss.metadata,
            "turn_annotations": [
                {
                    "turn_index": ann.turn_index,
                    "label": ann.label,
                    "score": (
                        _score_to_dict(ann.score)
                        if ann.score
                        else None
                    ),
                }
                for ann in ss.turn_annotations
            ],
        }

    return d


def _score_to_dict(score: Any) -> dict[str, Any]:
    return {
        "score": score.score,
        "passed": score.passed,
        "detail": score.detail,
        "metadata": score.metadata,
    }


def _truncate(text: str, max_len: int) -> str:
    """截断文本，超长部分用省略号替代。"""
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
