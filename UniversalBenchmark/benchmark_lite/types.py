"""Benchmark Lite 的所有数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TurnType(Enum):
    """回合类型。

    - ``CONVERSATION``: 普通对话回合，用于向 Agent 注入信息。
    - ``EVALUATION``: 评估回合，Agent 的回复将被评分。
    """

    CONVERSATION = "conversation"
    EVALUATION = "evaluation"


# ── 输入侧数据结构 ─────────────────────────────────────────────


@dataclass
class Turn:
    """对话中的一个回合。

    Attributes
    ----------
    user_input:
        发送给 Agent 的用户消息。
    turn_type:
        回合类型。``CONVERSATION`` 不评分，``EVALUATION`` 会评分。
    reference:
        参考答案或评估依据（仅 ``EVALUATION`` 回合需要）。
        类型为 ``Any`` 以适配不同的评估策略（关键词列表、
        正则表达式、结构化数据等）。
    metadata:
        额外的元数据，供 Benchmark 实现自行使用。
    """

    user_input: str
    turn_type: TurnType = TurnType.CONVERSATION
    reference: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryTurn:
    """一轮预置对话记录（用户消息 + 助手回复）。

    用于 :attr:`Scenario.preload_history`，在 Agent 正式运行
    评估回合之前，先将这些历史对话注入 Agent 的记忆中。
    """

    user_message: str
    assistant_response: str


@dataclass
class Scenario:
    """脚本化评估场景：由预定义的 Turn 列表组成。

    典型用法：先用若干 ``CONVERSATION`` 回合向 Agent 注入信息，
    再用 ``EVALUATION`` 回合检测 Agent 是否正确保留了记忆。

    也支持通过 ``preload_history`` 预置一段对话历史（不经过 LLM
    生成），适配那些 "给定对话历史 → 问答" 形式的 Benchmark 数据集。

    Attributes
    ----------
    id:
        场景唯一标识。
    turns:
        按时间顺序排列的回合列表。
    description:
        场景的人类可读描述。
    preload_history:
        在执行 ``turns`` 之前预先注入的对话历史。
        Runner 会依次调用 ``agent.get_messages()`` 和
        ``agent.add_response()`` 将这些记录注入记忆，但 **不会**
        调用 LLM 生成回复。
    metadata:
        额外的元数据。
    """

    id: str
    turns: list[Turn]
    description: str = ""
    preload_history: list[HistoryTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── 输出侧数据结构 ─────────────────────────────────────────────


@dataclass
class TurnScore:
    """单个 EVALUATION 回合的评估得分。

    Attributes
    ----------
    score:
        0.0 ~ 1.0 之间的连续分数。
    passed:
        是否视为"通过"。
    detail:
        人类可读的评估细节。
    metadata:
        额外的元数据（如 token-level 概率等）。
    """

    score: float
    passed: bool
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnAnnotation:
    """对单个回合的事后标注。

    用于 :class:`ScenarioScore`，在整个场景结束后，回溯标注
    每个回合在对话中的角色（有效信息 / 干扰 / 评估 / 其他）。

    Attributes
    ----------
    turn_index:
        对应的回合索引。
    label:
        角色标签，如 ``"information"``、``"noise"``、``"evaluation"``。
    score:
        如果此回合被视为评估回合，则附带评分。
    """

    turn_index: int
    label: str
    score: Optional[TurnScore] = None


@dataclass
class ScenarioScore:
    """场景级别的整体评估结果（用于事后评估模式）。

    与逐回合评分（:class:`TurnScore`）不同，``ScenarioScore`` 是在
    整个场景对话结束后，由 Benchmark 回溯评估产生的。

    Attributes
    ----------
    score:
        场景整体得分 (0.0 ~ 1.0)。
    passed:
        是否视为"通过"。
    turn_annotations:
        对场景中每个回合的事后标注（哪些是有效信息、干扰、评估等）。
    detail:
        人类可读的评估细节。
    metadata:
        额外的元数据。
    """

    score: float
    passed: bool
    turn_annotations: list[TurnAnnotation] = field(default_factory=list)
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    """单个回合的运行结果（输入 + 输出 + 可选评分）。"""

    turn_index: int
    user_input: str
    response: str
    turn_type: TurnType = TurnType.CONVERSATION
    score: Optional[TurnScore] = None


@dataclass
class ScenarioResult:
    """单个场景的完整运行结果。

    对于脚本化场景，评分体现在各 ``TurnResult.score`` 中。
    对于交互式场景，评分体现在 ``scenario_score`` 中。
    """

    scenario_id: str
    scenario_description: str = ""
    turn_results: list[TurnResult] = field(default_factory=list)
    scenario_score: Optional[ScenarioScore] = None

    @property
    def eval_scores(self) -> list[TurnScore]:
        """所有逐回合评分的列表。"""
        return [r.score for r in self.turn_results if r.score is not None]

    @property
    def eval_count(self) -> int:
        return len(self.eval_scores)

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.eval_scores if s.passed)


@dataclass
class AggregateResult:
    """Benchmark 聚合结果（固定结构）。

    由 Benchmark 实现者在 :meth:`~BenchmarkLite.aggregate` 中填充。
    框架提供统一的展示格式，实现者决定如何计算每个字段。

    Attributes
    ----------
    score:
        最终归一化得分 (0.0 ~ 1.0)。
        具体含义由 Benchmark 自行定义，例如可以是加权准确率、
        累加得分比等。
    total_score:
        原始累加得分（未归一化）。
    total_max_score:
        最大可能得分，与 ``total_score`` 配合展示为
        ``total_score / total_max_score``。
    total:
        评估点总数。
    passed:
        通过的评估点数。
    detail:
        人类可读的评估总结。
    extra:
        Benchmark 特有的额外指标（如 ``memory_distance``），
        会以 key-value 形式附加展示。
    """

    score: float
    total_score: float
    total_max_score: float
    total: int
    passed: int
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """完整的 Benchmark 运行结果。"""

    benchmark_name: str
    agent_identifier: str
    scenario_results: list[ScenarioResult]
    aggregate: AggregateResult
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
