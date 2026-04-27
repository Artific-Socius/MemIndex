#!/usr/bin/env python3
"""手动验证全局 Rich 进度条（请在真实终端中运行，勿依赖 pytest 捕获）。

用法（在 UniversalBenchmark 目录下）::

    python scripts/progress_smoke.py

会启用 progress_context，跑一个极小的 Benchmark + 假 LLM，并故意 sleep，
便于肉眼看到多任务进度叠加。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 保证可导入 agent / benchmark_lite
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import Agent  # noqa: E402
from agent.memory.buffer import BufferMemory  # noqa: E402
from agent.progress import get_progress, progress_context  # noqa: E402
from benchmark_lite import (  # noqa: E402
    AggregateResult,
    BenchmarkLite,
    Runner,
    Scenario,
    ScenarioResult,
    Turn,
    TurnScore,
    TurnType,
)


class _SlowDummyAgent(Agent.compose(BufferMemory)):  # type: ignore[misc]
    def generate(self, messages):  # type: ignore[override]
        time.sleep(0.25)
        return "dummy"


class _SmokeBench(BenchmarkLite):
    @property
    def name(self) -> str:
        return "ProgressSmoke"

    def get_scenarios(self):
        return [
            Scenario(
                id="scene_a",
                turns=[
                    Turn("hello"),
                    Turn("eval?", TurnType.EVALUATION, reference="x"),
                ],
            ),
            Scenario(
                id="scene_b",
                turns=[Turn("only")],
            ),
        ]

    def evaluate(self, turn, response, history) -> TurnScore:
        return TurnScore(score=1.0, passed=True)

    def aggregate(self, scenario_results: list[ScenarioResult]) -> AggregateResult:
        return AggregateResult(
            score=1.0,
            total_score=1.0,
            total_max_score=1.0,
            total=1,
            passed=1,
        )


def main() -> None:
    agent = _SlowDummyAgent(
        model="dummy/model",
        system_prompt="smoke",
    )
    runner = Runner(verbose=False)

    # 与 CLI 一致：非 TTY 时无进度条
    tty = sys.stderr.isatty()
    use_live = tty
    if not tty:
        print("(stderr 非 TTY，已禁用 Rich 进度；加管道时请直接看日志)", file=sys.stderr)

    with progress_context(live=use_live, force_console=tty):
        pg = get_progress()
        extra = pg.add_task("开发者自定义子任务", total=3.0)
        try:
            for i in range(3):
                time.sleep(0.2)
                pg.advance(extra, 1)
            runner.run(agent, _SmokeBench())
        finally:
            pg.remove_task(extra)

    print("完成。")


if __name__ == "__main__":
    main()
