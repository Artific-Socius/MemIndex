"""BenchmarkLite 和 InteractiveScenario 抽象基类。

Benchmark Lite 支持三种场景模式：

1. **脚本化 (Scenario)** — 预定义 Turn 列表，逐回合评估。
2. **预置历史 (Scenario + preload_history)** — 先注入对话历史，
   再运行评估回合。适配 "给定对话历史 → 问答" 形式的数据集。
3. **交互式 (InteractiveScenario)** — 动态生成回合，与 Agent
   实时交互，对话结束后事后评估。

实现者根据 Benchmark 的特点，选择对应的模式即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union

from .types import (
    AggregateResult,
    Scenario,
    ScenarioResult,
    ScenarioScore,
    Turn,
    TurnResult,
    TurnScore,
)


# ── InteractiveScenario ────────────────────────────────────────


class InteractiveScenario(ABC):
    """交互式评估场景：动态生成回合，事后评估。

    与 :class:`Scenario` 不同，``InteractiveScenario`` 的回合不是
    预定义的，而是通过 :meth:`next_turn` 逐个生成，可以根据 Agent
    的实际回复来决定下一步提问或何时结束对话。

    评估在整个对话结束后，由 :meth:`evaluate` 统一进行。
    Benchmark 可以回溯标注每个回合的角色（有效信息 / 干扰 / 评估）。

    实现者需要实现三个方法::

        class MyInteractiveScenario(InteractiveScenario):

            @property
            def id(self) -> str:
                return "my_scenario"

            def next_turn(self, history):
                if len(history) >= 5:
                    return None          # 结束对话
                return Turn("下一个问题...")

            def evaluate(self, history):
                ...
                return ScenarioScore(score=0.8, passed=True, ...)
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """场景唯一标识。"""
        ...

    @property
    def description(self) -> str:
        """场景的人类可读描述（可选）。"""
        return ""

    @property
    def metadata(self) -> dict[str, Any]:
        """额外的元数据（可选）。"""
        return {}

    @abstractmethod
    def next_turn(self, history: list[TurnResult]) -> Optional[Turn]:
        """根据当前对话历史，生成下一个回合。

        Parameters
        ----------
        history:
            当前场景中已完成的所有回合（包含 Agent 的实际回复）。
            首次调用时为空列表。

        Returns
        -------
        Turn | None
            下一个回合。返回 ``None`` 表示结束对话。
            ``Turn.turn_type`` 仅作为标记，实际评估由
            :meth:`evaluate` 统一进行。
        """
        ...

    @abstractmethod
    def evaluate(self, history: list[TurnResult]) -> ScenarioScore:
        """在对话结束后，对整个场景进行事后评估。

        Parameters
        ----------
        history:
            场景中所有回合的完整记录。

        Returns
        -------
        ScenarioScore
            包含场景整体得分，以及每个回合的标注
            （哪些是有效信息、干扰、评估点等）。
        """
        ...


# ── 类型别名 ───────────────────────────────────────────────────

AnyScenario = Union[Scenario, InteractiveScenario]


# ── BenchmarkLite ──────────────────────────────────────────────


class BenchmarkLite(ABC):
    """通用轻量 Benchmark 抽象接口。

    支持三种场景模式（可混合使用）：

    **模式 1 — 脚本化场景** ::

        def get_scenarios(self):
            return [Scenario(id="s1", turns=[
                Turn("我叫小明"),
                Turn("我叫什么？", turn_type=TurnType.EVALUATION, reference=["小明"]),
            ])]

        def evaluate(self, turn, response, history):
            ...  # 逐回合评估

    **模式 2 — 预置历史** ::

        def get_scenarios(self):
            return [Scenario(
                id="s2",
                preload_history=[
                    HistoryTurn("我叫小明", "你好小明！"),
                    HistoryTurn("我住在北京", "北京是个好地方。"),
                ],
                turns=[
                    Turn("我叫什么名字？", turn_type=TurnType.EVALUATION,
                         reference=["小明"]),
                ],
            )]

    **模式 3 — 交互式场景** ::

        def get_scenarios(self):
            return [MyInteractiveScenario()]
            # 评估由 InteractiveScenario.evaluate() 完成
    """

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark 的名称。"""
        ...

    @abstractmethod
    def get_scenarios(self) -> Iterable[AnyScenario]:
        """返回所有评估场景。

        可以返回 :class:`Scenario` 和 :class:`InteractiveScenario`
        的混合列表。可以返回 ``list`` 或生成器。
        """
        ...

    # ------------------------------------------------------------------
    # 逐回合评估（用于脚本化场景中的 EVALUATION 回合）
    # ------------------------------------------------------------------

    def evaluate(
        self,
        turn: Turn,
        response: str,
        history: list[TurnResult],
    ) -> TurnScore:
        """评估单个 ``EVALUATION`` 回合。

        仅在脚本化场景（:class:`Scenario`）中遇到
        ``TurnType.EVALUATION`` 回合时由 Runner 调用。

        如果 Benchmark 只使用 :class:`InteractiveScenario`，
        则无需覆写此方法。

        Parameters
        ----------
        turn:
            当前评估回合，``turn.reference`` 中包含参考答案。
        response:
            Agent 对该回合的实际回复。
        history:
            当前场景中此回合 **之前** 的所有 :class:`TurnResult`。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 的场景包含 EVALUATION 回合，"
            f"但未实现 evaluate() 方法"
        )

    # ------------------------------------------------------------------
    # 结果聚合
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate(
        self,
        scenario_results: list[ScenarioResult],
    ) -> AggregateResult:
        """聚合所有场景的评估结果，返回统一格式的最终指标。

        实现者需要决定如何从各场景结果中提取分数并填充
        :class:`AggregateResult` 的各个字段。

        Parameters
        ----------
        scenario_results:
            所有场景的运行结果列表。

        Returns
        -------
        AggregateResult
            包含 ``score``、``total_score / total_max_score``、
            ``total / passed`` 等标准化字段的聚合结果。
        """
        ...
