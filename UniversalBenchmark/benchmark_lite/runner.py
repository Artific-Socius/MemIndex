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
    MessageTrace,
    PreloadHistoryEntry,
    Scenario,
    ScenarioResult,
    ScenarioScore,
    TurnResult,
    TurnType,
)


def _trace_from_agent(agent: Agent) -> Optional[MessageTrace]:
    """从 Agent 的 last_turn_trace 构建 MessageTrace（Pydantic 模型）。"""
    raw = getattr(agent, "last_turn_trace", None)
    if raw is None:
        return None
    return MessageTrace(
        user_message_id=raw.user_message_id,
        assistant_message_id=raw.assistant_message_id,
        id_source=raw.id_source,
        query_request_id=raw.query_request_id,
        append_request_id=raw.append_request_id,
        extra=dict(raw.extra) if raw.extra else {},
    )


def _memory_library_id_from_agent(agent: Agent) -> str:
    """安全获取当前 memory library id。"""
    try:
        return agent.memory_library_id
    except Exception:
        return ""


def _dependency_from_turn_meta(
    meta: dict[str, object],
) -> tuple[list[int], str]:
    """从 Turn.metadata 提取依赖字段。"""
    raw_indices = (
        meta.get("depends_on_turn_indices")
        or meta.get("dependency_turn_indices")
        or []
    )
    depends_on_turn_indices: list[int] = []
    if isinstance(raw_indices, list):
        for idx in raw_indices:
            if isinstance(idx, int):
                depends_on_turn_indices.append(idx)
    policy = meta.get("dependency_policy", "")
    dependency_policy = policy if isinstance(policy, str) else ""
    return depends_on_turn_indices, dependency_policy


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
        scenario_memory_library_ids: dict[str, str] = {}

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
            if sr.memory_library_id:
                scenario_memory_library_ids[sid] = sr.memory_library_id

        aggregate = benchmark.aggregate(scenario_results)

        if self._verbose:
            logger.info(f"Benchmark 完成: {aggregate}")

        result = BenchmarkResult(
            benchmark_name=benchmark.name,
            agent_identifier=agent.identifier,
            scenario_results=scenario_results,
            aggregate=aggregate,
            timestamp=datetime.datetime.now().isoformat(),
        )
        if scenario_memory_library_ids:
            result.metadata["scenario_memory_library_ids"] = (
                scenario_memory_library_ids
            )
        return result

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

            trace = _trace_from_agent(agent)
            turn_meta = dict(turn.metadata) if turn.metadata else {}
            deps, dep_policy = _dependency_from_turn_meta(turn_meta)

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
                metadata=turn_meta,
                message_trace=trace,
                depends_on_turn_indices=deps,
                dependency_policy=dep_policy,
            )
            turn_results.append(result)

            if on_turn_complete:
                on_turn_complete(scenario, result)

        preload = [
            PreloadHistoryEntry(
                user_message=h.user_message,
                assistant_response=h.assistant_response,
            )
            for h in scenario.preload_history
        ]

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            turn_results=turn_results,
            metadata=dict(scenario.metadata) if scenario.metadata else {},
            preload_history=preload,
            memory_library_id=_memory_library_id_from_agent(agent),
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
            trace = _trace_from_agent(agent)
            turn_meta = dict(turn.metadata) if turn.metadata else {}
            deps, dep_policy = _dependency_from_turn_meta(turn_meta)

            if self._verbose:
                tag = turn.turn_type.value.upper()[:4]
                logger.debug(f"  Turn {ti} [{tag}] done")

            result = TurnResult(
                turn_index=ti,
                user_input=turn.user_input,
                response=response,
                turn_type=turn.turn_type,
                metadata=turn_meta,
                message_trace=trace,
                depends_on_turn_indices=deps,
                dependency_policy=dep_policy,
            )
            turn_results.append(result)

            if on_turn_complete:
                on_turn_complete(scenario, result)

            ti += 1

        # 事后评估
        scenario_score: ScenarioScore = scenario.evaluate(turn_results)
        ann_score_map = {
            ann.turn_index: ann.score
            for ann in scenario_score.turn_annotations
            if ann.label == "evaluation"
        }
        for tr in turn_results:
            if tr.turn_type != TurnType.EVALUATION or tr.score is not None:
                continue
            ann_score = ann_score_map.get(tr.turn_index)
            if ann_score is not None:
                tr.score = ann_score
            else:
                tr.metadata["score_backfill_status"] = "missing_annotation_score"

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
            metadata=dict(scenario.metadata) if scenario.metadata else {},
            memory_library_id=_memory_library_id_from_agent(agent),
        )
