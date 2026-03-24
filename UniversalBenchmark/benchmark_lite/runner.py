"""Runner：编排 Agent 与 BenchmarkLite 的运行器。

支持三种场景模式：
- 脚本化场景 (Scenario)
- 带预置历史的场景 (Scenario + preload_history)
- 交互式场景 (InteractiveScenario)
"""

from __future__ import annotations

import datetime
from typing import Callable, Optional

from loguru import logger

from agent import Agent

from .base import BenchmarkLite, InteractiveScenario
from .types import (
    BenchmarkResult,
    Scenario,
    ScenarioResult,
    ScenarioScore,
    TurnResult,
    TurnType,
)


class Runner:
    """编排 Agent 与 BenchmarkLite 的运行器。

    基本用法::

        runner = Runner()
        result = runner.run(agent, benchmark)

    Parameters
    ----------
    verbose:
        是否在运行过程中输出日志。
    """

    def __init__(self, verbose: bool = True) -> None:
        self._verbose = verbose

    def run(
        self,
        agent: Agent,
        benchmark: BenchmarkLite,
        *,
        on_scenario_start: Optional[
            Callable[[Scenario | InteractiveScenario], None]
        ] = None,
        on_turn_complete: Optional[
            Callable[[Scenario | InteractiveScenario, TurnResult], None]
        ] = None,
    ) -> BenchmarkResult:
        """运行完整的 Benchmark 评估流程。

        自动根据场景类型选择对应的运行策略：

        - :class:`Scenario` → 脚本化执行 + 逐回合评估
        - :class:`InteractiveScenario` → 动态交互 + 事后评估
        """
        scenarios = list(benchmark.get_scenarios())
        scenario_results: list[ScenarioResult] = []

        if self._verbose:
            logger.info(
                f"开始 Benchmark '{benchmark.name}' "
                f"({len(scenarios)} 个场景)"
            )
            logger.info(f"Agent: {agent.identifier}")

        for si, scenario in enumerate(scenarios):
            if on_scenario_start:
                on_scenario_start(scenario)

            sid = scenario.id
            desc = ""
            if isinstance(scenario, InteractiveScenario):
                desc = scenario.description
            elif isinstance(scenario, Scenario):
                desc = scenario.description

            if self._verbose:
                suffix = f" — {desc}" if desc else ""
                logger.info(
                    f"[{si + 1}/{len(scenarios)}] 场景: {sid}{suffix}"
                )

            agent.reset()

            if isinstance(scenario, InteractiveScenario):
                sr = self._run_interactive(
                    agent, scenario, on_turn_complete,
                )
            else:
                sr = self._run_scripted(
                    agent, benchmark, scenario, on_turn_complete,
                )

            scenario_results.append(sr)

        aggregate = benchmark.aggregate(scenario_results)

        if self._verbose:
            logger.info(f"Benchmark 完成: {aggregate}")

        return BenchmarkResult(
            benchmark_name=benchmark.name,
            agent_identifier=agent.identifier,
            scenario_results=scenario_results,
            aggregate=aggregate,
            timestamp=datetime.datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 智能上下文导入
    # ------------------------------------------------------------------

    def _import_context(
        self,
        agent: Agent,
        scenario: Scenario,
    ) -> None:
        """Smart context loading: corpus docs -> import_corpus; history -> bulk_import."""
        meta = scenario.metadata or {}
        corpus_docs: list[str] = meta.get("corpus_documents", [])
        corpus_id: str = meta.get("corpus_id", "")

        if corpus_docs and hasattr(agent, "import_corpus"):
            logger.info(
                f"  语料导入: {len(corpus_docs)} 篇文档 "
                f"(corpus_id={corpus_id})"
            )
            lib_id = agent.import_corpus(corpus_docs, corpus_id)
            if hasattr(agent, "set_persistent_lib"):
                agent.set_persistent_lib(lib_id)
            return

        if scenario.preload_history:
            conversations = [
                (entry.user_message, entry.assistant_response)
                for entry in scenario.preload_history
            ]
            imported = agent.bulk_import(conversations)
            if self._verbose:
                logger.info(f"  批量导入 {imported} 轮对话历史")

    # ------------------------------------------------------------------
    # 脚本化场景
    # ------------------------------------------------------------------

    def _run_scripted(
        self,
        agent: Agent,
        benchmark: BenchmarkLite,
        scenario: Scenario,
        on_turn_complete: Optional[
            Callable[[Scenario, TurnResult], None]
        ],
    ) -> ScenarioResult:
        self._import_context(agent, scenario)

        turn_results: list[TurnResult] = []
        for ti, turn in enumerate(scenario.turns):
            response = agent.chat(turn.user_input)

            score = None
            if turn.turn_type == TurnType.EVALUATION:
                score = benchmark.evaluate(turn, response, turn_results)
                if self._verbose:
                    status = "PASS" if score.passed else "FAIL"
                    logger.info(
                        f"  Turn {ti} [EVAL] {status} "
                        f"(score={score.score:.2f})"
                    )
            elif self._verbose:
                logger.debug(f"  Turn {ti} [CONV] done")

            result = TurnResult(
                turn_index=ti,
                user_input=turn.user_input,
                response=response,
                turn_type=turn.turn_type,
                score=score,
            )
            turn_results.append(result)

            if on_turn_complete:
                on_turn_complete(scenario, result)

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            turn_results=turn_results,
        )

    # ------------------------------------------------------------------
    # 交互式场景
    # ------------------------------------------------------------------

    def _run_interactive(
        self,
        agent: Agent,
        scenario: InteractiveScenario,
        on_turn_complete: Optional[
            Callable[[InteractiveScenario, TurnResult], None]
        ],
    ) -> ScenarioResult:
        turn_results: list[TurnResult] = []
        ti = 0

        while True:
            turn = scenario.next_turn(turn_results)
            if turn is None:
                break

            response = agent.chat(turn.user_input)

            if self._verbose:
                tag = turn.turn_type.value.upper()[:4]
                logger.debug(f"  Turn {ti} [{tag}] done")

            result = TurnResult(
                turn_index=ti,
                user_input=turn.user_input,
                response=response,
                turn_type=turn.turn_type,
            )
            turn_results.append(result)

            if on_turn_complete:
                on_turn_complete(scenario, result)

            ti += 1

        # 事后评估
        scenario_score: ScenarioScore = scenario.evaluate(turn_results)

        if self._verbose:
            status = "PASS" if scenario_score.passed else "FAIL"
            logger.info(
                f"  场景评估 {status} "
                f"(score={scenario_score.score:.2f})"
            )
            if scenario_score.turn_annotations:
                for ann in scenario_score.turn_annotations:
                    ann_info = f"    Turn {ann.turn_index}: [{ann.label}]"
                    if ann.score is not None:
                        mark = "PASS" if ann.score.passed else "FAIL"
                        ann_info += f" {mark} ({ann.score.score:.2f})"
                    logger.info(ann_info)

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            turn_results=turn_results,
            scenario_score=scenario_score,
        )
