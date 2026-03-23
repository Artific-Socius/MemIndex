"""Benchmark Lite — 通用轻量级 Agent 记忆评估框架。

支持三种场景模式：

**模式 1 — 脚本化场景** ::

    Scenario(id="s1", turns=[
        Turn("我叫小明"),
        Turn("我叫什么？", turn_type=TurnType.EVALUATION, reference=["小明"]),
    ])

**模式 2 — 预置历史** ::

    Scenario(id="s2",
        preload_history=[HistoryTurn("我叫小明", "你好小明！")],
        turns=[Turn("我叫什么？", turn_type=TurnType.EVALUATION, reference=["小明"])],
    )

**模式 3 — 交互式场景** ::

    class MyInteractive(InteractiveScenario):
        def next_turn(self, history): ...
        def evaluate(self, history) -> ScenarioScore: ...

快速上手::

    result = Runner().run(agent, MyBenchmark())
    print(format_report(result, verbose=True))
"""

from .adapter import UniversalAdapter
from .base import AnyScenario, BenchmarkLite, InteractiveScenario
from .report import format_report, to_dict, to_json
from .runner import Runner
from .types import (
    AggregateResult,
    BenchmarkResult,
    HistoryTurn,
    Scenario,
    ScenarioResult,
    ScenarioScore,
    Turn,
    TurnAnnotation,
    TurnResult,
    TurnScore,
    TurnType,
)

__all__ = [
    # ABC
    "BenchmarkLite",
    "InteractiveScenario",
    # 适配器
    "UniversalAdapter",
    # 类型别名
    "AnyScenario",
    # 输入侧
    "Turn",
    "TurnType",
    "HistoryTurn",
    "Scenario",
    # 输出侧
    "AggregateResult",
    "TurnScore",
    "TurnAnnotation",
    "ScenarioScore",
    "TurnResult",
    "ScenarioResult",
    "BenchmarkResult",
    # 运行与报告
    "Runner",
    "format_report",
    "to_dict",
    "to_json",
]
