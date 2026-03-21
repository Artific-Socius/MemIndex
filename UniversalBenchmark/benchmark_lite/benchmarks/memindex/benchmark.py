"""MemIndex BenchmarkLite 适配器。

将 MemIndex 的测试数据集和调度逻辑封装为一个 BenchmarkLite 实现，
可以通过 ``run_benchmark.py`` 直接运行。

Usage::

    python run_benchmark.py \\
        --benchmark benchmark_lite.benchmarks.memindex.MemIndexBenchmark \\
        --memory buffer \\
        --model openrouter/google/gemini-2.5-flash

也可以在代码中直接使用::

    from benchmark_lite.benchmarks.memindex import MemIndexBenchmark
    from benchmark_lite import Runner
    from agent import Agent

    agent = Agent(model="...", memory_type="buffer")
    benchmark = MemIndexBenchmark(
        config_path="path/to/1k.json",
        eval_model="openrouter/google/gemini-2.5-flash",
        eval_mode="binary",
    )
    result = Runner().run(agent, benchmark)
"""

from __future__ import annotations

import os
from typing import Iterable

from benchmark_lite.base import AnyScenario, BenchmarkLite
from benchmark_lite.types import AggregateResult, ScenarioResult

from .data import load_dataset
from .scenario import MemIndexScenario

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..")
)
_MEMINDEX_ROOT = os.path.join(_PROJECT_ROOT, "MemIndex")
_DEFAULT_CONFIG = os.path.join(
    _MEMINDEX_ROOT, "data", "config", "1k.json"
)


class MemIndexBenchmark(BenchmarkLite):
    """MemIndex 基准测试的 BenchmarkLite 适配器。

    默认加载 ``MemIndex/data/config/1k.json``（1024 tokens 记忆距离）。
    可通过构造参数指定不同的数据集和评估配置。

    Parameters
    ----------
    config_path:
        MemIndex 配置文件路径（如 ``1k.json``）。
        为 ``None`` 时自动定位项目中的默认配置。
    eval_model:
        用于评估 Agent 回复的 LLM 模型名称。
    eval_mode:
        评估模式 — ``"binary"``（二元）或 ``"score"``（0-1 连续分数）。
    """

    def __init__(
        self,
        config_path: str | None = None,
        eval_model: str = "openrouter/google/gemini-2.5-flash",
        eval_mode: str = "binary",
    ) -> None:
        if config_path is None:
            config_path = _DEFAULT_CONFIG

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"MemIndex config not found: {config_path}\n"
                f"Expected at: {os.path.abspath(config_path)}"
            )

        self._dataset = load_dataset(config_path)
        self._eval_model = eval_model
        self._eval_mode = eval_mode
        self._config_name = os.path.splitext(
            os.path.basename(config_path)
        )[0]

    @property
    def name(self) -> str:
        return f"MemIndex-{self._config_name}"

    def get_scenarios(self) -> Iterable[AnyScenario]:
        return [
            MemIndexScenario(
                dataset=self._dataset,
                eval_model=self._eval_model,
                eval_mode=self._eval_mode,
                scenario_id=f"memindex_{self._config_name}",
                memindex_root=_MEMINDEX_ROOT,
            )
        ]

    def aggregate(
        self, scenario_results: list[ScenarioResult],
    ) -> AggregateResult:
        """按 MemIndex 的方式聚合：sum(scores) / sum(max_scores)。"""
        total_score = 0.0
        total_max = 0.0
        eval_count = 0
        passed_count = 0
        memory_distance = 0

        for sr in scenario_results:
            if sr.scenario_score and sr.scenario_score.metadata:
                meta = sr.scenario_score.metadata
                total_score += float(meta.get("total_score", 0))
                total_max += float(meta.get("total_max_score", 0))
                eval_count += int(meta.get("eval_count", 0))
                passed_count += int(meta.get("passed_count", 0))
                memory_distance = int(
                    meta.get("memory_distance", memory_distance)
                )

        score = total_score / total_max if total_max > 0 else 0.0

        return AggregateResult(
            score=score,
            total_score=total_score,
            total_max_score=total_max,
            total=eval_count,
            passed=passed_count,
            extra={"memory_distance": memory_distance},
        )
