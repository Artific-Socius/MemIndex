"""UniversalAdapter: bridges benchmark.interfaces.Benchmark to BenchmarkLite.

This adapter automatically converts data-layer Benchmark objects into
executable BenchmarkLite instances, so any dataset that implements
``benchmark.interfaces.Benchmark`` can be run through the Benchmark Lite
framework without writing custom scenario/runner code.

Usage::

    from benchmark.interfaces import Benchmark
    from benchmark_lite.adapter import UniversalAdapter

    data_benchmark: Benchmark = ...
    adapted = UniversalAdapter(data_benchmark, eval_model="...")

    from benchmark_lite import Runner
    result = Runner().run(agent, adapted)

Or via CLI::

    python run_benchmark_lite.py \\
        --benchmark benchmark.data.providers.evermind_ai.EverMemBenchStaticBenchmark \\
        --memory buffer --model openrouter/google/gemini-2.5-flash
"""

from __future__ import annotations

from typing import Any, Iterable

from loguru import logger

from benchmark.interfaces import Benchmark, Scene
from benchmark_lite.base import BenchmarkLite
from benchmark_lite.evaluators import BaseEvaluator, get_evaluator
from benchmark_lite.types import (
    AggregateResult,
    HistoryTurn,
    Scenario,
    ScenarioResult,
    Turn,
    TurnResult,
    TurnScore,
    TurnType,
)


class UniversalAdapter(BenchmarkLite):
    """Adapts a data-layer Benchmark into a BenchmarkLite for execution.

    The adapter:

    1. Reads Scenes from the data Benchmark.
    2. Converts each Scene's ``conversation_history`` into ``preload_history``.
    3. If no conversation history exists but ``background_text`` is available,
       injects it as a single preloaded exchange.
    4. Converts each Scene's Questions into ``EVALUATION`` Turns.
    5. Routes evaluation to the appropriate evaluator based on ``ScoringConfig``.

    Parameters
    ----------
    data_benchmark:
        A ``benchmark.interfaces.Benchmark`` instance providing data.
    eval_model:
        LLM model name for LLM-based evaluators.
    scene_ids:
        Optional list of scene IDs to include.  ``None`` means all scenes
        (requires the benchmark to implement ``list_scenes``).
    max_bg_chars:
        Maximum characters to keep from ``background_text()``.
        ``None`` means no limit (use the full corpus).
    max_questions:
        Maximum number of evaluation questions per scene.
        ``None`` means all questions.
    """

    def __init__(
        self,
        data_benchmark: Benchmark,
        eval_model: str = "openrouter/google/gemini-2.5-flash",
        scene_ids: list[str] | None = None,
        max_bg_chars: int | None = None,
        max_questions: int | None = None,
    ) -> None:
        self._benchmark = data_benchmark
        self._eval_model = eval_model
        self._scene_ids = scene_ids
        self._max_bg_chars = max_bg_chars
        self._max_questions = max_questions
        self._evaluator_cache: dict[str, BaseEvaluator] = {}

        self._bench_eval_prompt: str = getattr(
            data_benchmark, "eval_prompt", "",
        ) or ""

    @property
    def name(self) -> str:
        return self._benchmark.benchmark_name

    # ------------------------------------------------------------------
    # get_scenarios — Scene → Scenario conversion
    # ------------------------------------------------------------------

    def get_scenarios(self) -> Iterable[Scenario]:
        if self._scene_ids is not None:
            ids = self._scene_ids
        else:
            try:
                ids = list(self._benchmark.list_scenes())
            except NotImplementedError:
                logger.warning(
                    f"Benchmark '{self.name}' does not implement "
                    f"list_scenes(). Pass scene_ids explicitly."
                )
                return []

        scenarios: list[Scenario] = []
        for sid in ids:
            scene = self._benchmark.get_scene(sid)
            scenario = self._scene_to_scenario(scene)
            scenarios.append(scenario)

        return scenarios

    # ------------------------------------------------------------------
    # evaluate — per-turn scoring via evaluator registry
    # ------------------------------------------------------------------

    def evaluate(
        self,
        turn: Turn,
        response: str,
        history: list[TurnResult],
    ) -> TurnScore:
        meta = turn.metadata
        eval_mode: str = meta.get("eval_mode", "binary")
        ground_truth = meta.get("ground_truth", "")
        max_score: float = meta.get("max_score", 1.0)
        question_text: str = meta.get("question_text", turn.user_input)
        evidence = meta.get("evidence")

        evaluator = self._get_evaluator(eval_mode)
        return evaluator.evaluate(
            question_text=question_text,
            ground_truth=ground_truth,
            response=response,
            max_score=max_score,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # aggregate — default aggregation strategy
    # ------------------------------------------------------------------

    def aggregate(
        self,
        scenario_results: list[ScenarioResult],
    ) -> AggregateResult:
        total_score = 0.0
        total_max = 0.0
        eval_count = 0
        passed_count = 0

        for sr in scenario_results:
            for tr in sr.turn_results:
                if tr.score is not None:
                    total_score += tr.score.score
                    total_max += 1.0
                    eval_count += 1
                    if tr.score.passed:
                        passed_count += 1

        score = total_score / total_max if total_max > 0 else 0.0

        return AggregateResult(
            score=score,
            total_score=total_score,
            total_max_score=total_max,
            total=eval_count,
            passed=passed_count,
            extra={"source_benchmark": self._benchmark.benchmark_name},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scene_to_scenario(self, scene: Scene) -> Scenario:
        preload: list[HistoryTurn] = []

        conv_history = scene.conversation_history()
        for ct in conv_history:
            preload.append(
                HistoryTurn(
                    user_message=ct.user_message,
                    assistant_response=ct.assistant_response,
                )
            )

        if not preload:
            bg = scene.background_text(max_chars=self._max_bg_chars)
            if bg:
                preload.append(
                    HistoryTurn(
                        user_message=(
                            "Please read and remember the following "
                            "information carefully. I will ask you "
                            "questions about it later.\n\n" + bg
                        ),
                        assistant_response=(
                            "I have carefully read and memorized the "
                            "information you provided. Feel free to "
                            "ask me any questions about it."
                        ),
                    )
                )

        turns: list[Turn] = []
        for i, q in enumerate(scene.questions()):
            if self._max_questions is not None and i >= self._max_questions:
                break
            turn = Turn(
                user_input=q.question_text,
                turn_type=TurnType.EVALUATION,
                reference=q.ground_truth,
                metadata={
                    "question_id": q.question_id,
                    "ground_truth": q.ground_truth,
                    "eval_mode": q.scoring.eval_mode,
                    "eval_prompt_key": q.scoring.eval_prompt_key,
                    "max_score": q.scoring.max_score,
                    "question_text": q.question_text,
                    "evidence": q.evidence,
                },
            )
            turns.append(turn)

        if self._max_questions is not None:
            logger.info(
                f"  Scene '{scene.scene_id}': "
                f"{len(turns)} questions (limited from "
                f"{scene.question_count()})"
            )

        return Scenario(
            id=scene.scene_id,
            description=scene.scene_name or "",
            preload_history=preload,
            turns=turns,
            metadata={
                "task_type": scene.task_type,
                "source_benchmark": self._benchmark.benchmark_name,
            },
        )

    def _get_evaluator(self, eval_mode: str) -> BaseEvaluator:
        if eval_mode not in self._evaluator_cache:
            kwargs: dict[str, Any] = {"model": self._eval_model}
            if self._bench_eval_prompt:
                kwargs["prompt_template"] = self._bench_eval_prompt
            self._evaluator_cache[eval_mode] = get_evaluator(
                eval_mode, **kwargs,
            )
        return self._evaluator_cache[eval_mode]
