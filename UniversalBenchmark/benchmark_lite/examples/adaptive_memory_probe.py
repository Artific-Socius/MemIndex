"""示例 Benchmark：自适应记忆探测（交互式场景）。

展示 InteractiveScenario 的用法：

- Benchmark 动态生成回合，根据 Agent 的实际回复决定下一步。
- 对话结束后事后评估，回溯标注每个回合的角色。
- 如果 Agent 未能回忆某个事实，Benchmark 会追加提示再次探测。

用法::

    # CLI
    python run_benchmark.py \\
        --benchmark benchmark_lite.examples.AdaptiveMemoryProbe \\
        --memory buffer --model openrouter/google/gemini-2.5-flash-lite -v

    # 编程
    from benchmark_lite.examples import AdaptiveMemoryProbe
    benchmark = AdaptiveMemoryProbe()
"""

from __future__ import annotations

from typing import Iterable, Optional

from benchmark_lite import (
    AggregateResult,
    BenchmarkLite,
    InteractiveScenario,
    ScenarioResult,
    ScenarioScore,
    Turn,
    TurnAnnotation,
    TurnResult,
    TurnScore,
    TurnType,
)


# ── 交互式场景实现 ──────────────────────────────────────────────


class _AdaptiveProbeScenario(InteractiveScenario):
    """动态探测场景：注入事实 → 干扰 → 探测 → 对失败项追加提示。

    三个阶段：
    1. inject  — 向 Agent 注入若干事实信息
    2. distract — 插入与事实无关的闲聊（干扰）
    3. probe  — 逐个探测 Agent 是否记住了事实
       - 如果 Agent 回忆失败，追加一轮带提示的探测
    """

    FACTS: list[tuple[str, str, str]] = [
        # (注入消息, 探测问题, 关键词)
        ("我叫李华，是一名软件工程师。", "你还记得我叫什么吗？", "李华"),
        ("我有一只叫小黑的猫。", "我的宠物叫什么名字？", "小黑"),
        ("我住在深圳南山区。", "我住在哪个城市？", "深圳"),
    ]

    DISTRACTORS: list[str] = [
        "帮我讲一个笑话吧。",
        "你觉得最近有什么好看的电影吗？",
    ]

    def __init__(self) -> None:
        self._phase = "inject"
        self._inject_idx = 0
        self._distract_idx = 0
        self._probe_idx = 0
        self._retry_queue: list[int] = []

    @property
    def id(self) -> str:
        return "adaptive_memory_probe"

    @property
    def description(self) -> str:
        return "动态探测：注入事实 → 干扰 → 自适应探测"

    def next_turn(self, history: list[TurnResult]) -> Optional[Turn]:
        handlers = {
            "inject": self._next_inject,
            "distract": self._next_distract,
            "probe": self._next_probe,
            "retry": self._next_retry,
        }
        handler = handlers.get(self._phase)
        if handler is None:
            return None
        return handler(history)

    def _next_inject(
        self, history: list[TurnResult],
    ) -> Optional[Turn]:
        if self._inject_idx < len(self.FACTS):
            msg = self.FACTS[self._inject_idx][0]
            self._inject_idx += 1
            return Turn(msg)
        self._phase = "distract"
        return self._next_distract(history)

    def _next_distract(
        self, history: list[TurnResult],
    ) -> Optional[Turn]:
        if self._distract_idx < len(self.DISTRACTORS):
            msg = self.DISTRACTORS[self._distract_idx]
            self._distract_idx += 1
            return Turn(msg)
        self._phase = "probe"
        return self._next_probe(history)

    def _next_probe(
        self, history: list[TurnResult],
    ) -> Optional[Turn]:
        self._check_last_probe_failed(history)
        if self._probe_idx < len(self.FACTS):
            question = self.FACTS[self._probe_idx][1]
            keyword = self.FACTS[self._probe_idx][2]
            self._probe_idx += 1
            return Turn(
                question,
                turn_type=TurnType.EVALUATION,
                reference=keyword,
            )
        self._phase = "retry"
        return self._next_retry(history)

    def _next_retry(
        self, history: list[TurnResult],
    ) -> Optional[Turn]:
        self._check_last_probe_failed(history)
        if not self._retry_queue:
            return None
        fact_idx = self._retry_queue.pop(0)
        _, _, keyword = self.FACTS[fact_idx]
        hint = (
            "我之前告诉过你一些关于我个人的信息，"
            "你能再想想吗？"
        )
        return Turn(
            hint,
            turn_type=TurnType.EVALUATION,
            reference=keyword,
        )

    def _check_last_probe_failed(
        self, history: list[TurnResult],
    ) -> None:
        """检查上一轮探测是否失败，如果失败则加入重试队列。"""
        if not history or self._probe_idx == 0:
            return
        last = history[-1]
        if last.turn_type != TurnType.EVALUATION:
            return
        fact_idx = self._probe_idx - 1
        if fact_idx >= len(self.FACTS):
            return
        keyword = self.FACTS[fact_idx][2]
        if keyword.lower() not in last.response.lower():
            self._retry_queue.append(fact_idx)

    def evaluate(self, history: list[TurnResult]) -> ScenarioScore:
        """事后评估：回溯标注每个回合，计算总分。"""
        annotations: list[TurnAnnotation] = []
        eval_scores: list[TurnScore] = []

        inject_count = len(self.FACTS)
        distract_count = len(self.DISTRACTORS)

        for tr in history:
            idx = tr.turn_index

            if idx < inject_count:
                annotations.append(
                    TurnAnnotation(idx, "information")
                )
            elif idx < inject_count + distract_count:
                annotations.append(
                    TurnAnnotation(idx, "noise")
                )
            else:
                ref = tr.turn_type == TurnType.EVALUATION
                if ref and isinstance(
                    getattr(tr, "reference", None), str
                ):
                    pass

                keyword = _extract_keyword(tr, self.FACTS)
                if keyword:
                    found = keyword.lower() in tr.response.lower()
                    ts = TurnScore(
                        score=1.0 if found else 0.0,
                        passed=found,
                        detail=(
                            f"找到: ['{keyword}']"
                            if found
                            else f"缺失: ['{keyword}']"
                        ),
                    )
                    eval_scores.append(ts)
                    annotations.append(
                        TurnAnnotation(idx, "evaluation", ts)
                    )
                else:
                    annotations.append(
                        TurnAnnotation(idx, "evaluation")
                    )

        if eval_scores:
            avg = sum(s.score for s in eval_scores) / len(eval_scores)
            passed = avg >= 0.6
        else:
            avg = 0.0
            passed = False

        return ScenarioScore(
            score=round(avg, 4),
            passed=passed,
            turn_annotations=annotations,
            detail=(
                f"{sum(1 for s in eval_scores if s.passed)}"
                f"/{len(eval_scores)} 个探测通过"
            ),
        )


def _extract_keyword(
    tr: TurnResult,
    facts: list[tuple[str, str, str]],
) -> Optional[str]:
    """从已知事实表中匹配当前探测回合的关键词。"""
    for _, question, keyword in facts:
        if question in tr.user_input or keyword in tr.user_input:
            return keyword
    for _, _, keyword in facts:
        if keyword.lower() in tr.user_input.lower():
            return keyword
    return facts[0][2] if facts else None


# ── Benchmark 包装 ─────────────────────────────────────────────


class AdaptiveMemoryProbe(BenchmarkLite):
    """自适应记忆探测 Benchmark（交互式场景示例）。

    包含一个交互式场景，动态地向 Agent 注入事实、插入干扰、
    探测记忆，并对失败项追加提示再次探测。

    评估在对话完全结束后进行，同时标注每个回合的角色。
    """

    @property
    def name(self) -> str:
        return "AdaptiveMemoryProbe"

    def get_scenarios(self) -> Iterable[InteractiveScenario]:
        return [_AdaptiveProbeScenario()]

    def aggregate(
        self, scenario_results: list[ScenarioResult],
    ) -> AggregateResult:
        total_score = 0.0
        total_max = 0.0
        eval_count = 0
        passed_count = 0

        for sr in scenario_results:
            if sr.scenario_score is not None:
                ss = sr.scenario_score
                for ann in ss.turn_annotations:
                    if ann.score is not None:
                        total_score += ann.score.score
                        total_max += 1.0
                        eval_count += 1
                        if ann.score.passed:
                            passed_count += 1

        score = total_score / total_max if total_max > 0 else 0.0
        return AggregateResult(
            score=score,
            total_score=total_score,
            total_max_score=total_max,
            total=eval_count,
            passed=passed_count,
        )
