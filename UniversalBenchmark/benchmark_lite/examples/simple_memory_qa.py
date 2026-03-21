"""示例 Benchmark：简单记忆问答（脚本化 + 预置历史）。

展示两种场景模式：

1. **脚本化场景** — Agent 实际参与对话，然后被评估。
2. **预置历史场景** — 对话历史已给定，Agent 直接回答问题。

评估策略：检查 Agent 回复中是否包含参考答案中的所有关键词。

用法::

    # CLI
    python run_benchmark.py \\
        --benchmark benchmark_lite.examples.SimpleMemoryQA \\
        --memory buffer --model openrouter/google/gemini-2.5-flash-lite

    # 编程
    from benchmark_lite.examples import SimpleMemoryQA
    benchmark = SimpleMemoryQA()
"""

from __future__ import annotations

from typing import Iterable

from benchmark_lite import (
    AggregateResult,
    BenchmarkLite,
    HistoryTurn,
    Scenario,
    ScenarioResult,
    Turn,
    TurnResult,
    TurnScore,
    TurnType,
)


class SimpleMemoryQA(BenchmarkLite):
    """简单的记忆问答 Benchmark。

    包含四个场景：
    1. 单事实回忆（脚本化）
    2. 多事实回忆（脚本化）
    3. 干扰后回忆（脚本化）
    4. 预置历史后回忆（预置历史模式）
    """

    @property
    def name(self) -> str:
        return "SimpleMemoryQA"

    def get_scenarios(self) -> Iterable[Scenario]:
        return [
            # ── 场景 1: 单事实回忆 ──────────────────────────
            Scenario(
                id="name_recall",
                description="记住用户提到的名字",
                turns=[
                    Turn("我叫小明，今年25岁，在北京工作。"),
                    Turn("今天天气怎么样？"),
                    Turn(
                        "你还记得我叫什么名字吗？",
                        turn_type=TurnType.EVALUATION,
                        reference=["小明"],
                    ),
                ],
            ),
            # ── 场景 2: 多事实回忆 ──────────────────────────
            Scenario(
                id="multi_fact_recall",
                description="记住多个事实并分别回忆",
                turns=[
                    Turn("我有一只猫叫咪咪，还有一只狗叫旺财。"),
                    Turn("我最喜欢的颜色是蓝色。"),
                    Turn("周末我喜欢去爬山。"),
                    Turn(
                        "我的宠物分别叫什么名字？",
                        turn_type=TurnType.EVALUATION,
                        reference=["咪咪", "旺财"],
                    ),
                    Turn(
                        "我最喜欢什么颜色？",
                        turn_type=TurnType.EVALUATION,
                        reference=["蓝色"],
                    ),
                ],
            ),
            # ── 场景 3: 干扰后回忆 ──────────────────────────
            Scenario(
                id="recall_after_distraction",
                description="在干扰对话后仍能回忆事实",
                turns=[
                    Turn("我下周三有一个重要的会议，需要准备季度报告。"),
                    Turn("帮我写一首关于春天的诗吧。"),
                    Turn("你觉得人工智能未来会怎样发展？"),
                    Turn("请给我推荐一本好书。"),
                    Turn(
                        "我之前提到的会议是什么时候？需要准备什么？",
                        turn_type=TurnType.EVALUATION,
                        reference=["周三", "季度报告"],
                    ),
                ],
            ),
            # ── 场景 4: 预置历史后回忆 ──────────────────────
            Scenario(
                id="preloaded_history_recall",
                description="从预置的对话历史中回忆事实",
                preload_history=[
                    HistoryTurn(
                        "我叫张三，是一名软件工程师。",
                        "你好张三！很高兴认识你，软件工程师是个很棒的职业。",
                    ),
                    HistoryTurn(
                        "我住在上海浦东新区。",
                        "浦东新区是上海很繁华的区域呢！",
                    ),
                    HistoryTurn(
                        "我的手机号是13812345678。",
                        "好的，我记下了。",
                    ),
                ],
                turns=[
                    Turn(
                        "我叫什么名字？做什么工作的？",
                        turn_type=TurnType.EVALUATION,
                        reference=["张三", "软件工程师"],
                    ),
                    Turn(
                        "我住在哪里？",
                        turn_type=TurnType.EVALUATION,
                        reference=["浦东"],
                    ),
                ],
            ),
        ]

    def evaluate(
        self,
        turn: Turn,
        response: str,
        history: list[TurnResult],
    ) -> TurnScore:
        """关键词匹配评估。"""
        reference = turn.reference
        if not isinstance(reference, list):
            reference = [str(reference)]

        found: list[str] = []
        missing: list[str] = []
        for keyword in reference:
            if keyword.lower() in response.lower():
                found.append(keyword)
            else:
                missing.append(keyword)

        total = len(reference)
        score = len(found) / total if total > 0 else 0.0
        passed = len(missing) == 0

        detail_parts: list[str] = []
        if found:
            detail_parts.append(f"找到: {found}")
        if missing:
            detail_parts.append(f"缺失: {missing}")

        return TurnScore(
            score=round(score, 4),
            passed=passed,
            detail="; ".join(detail_parts),
        )

    def aggregate(
        self, scenario_results: list[ScenarioResult],
    ) -> AggregateResult:
        all_scores = [
            ts
            for sr in scenario_results
            for ts in sr.eval_scores
        ]
        total = len(all_scores)
        passed = sum(1 for s in all_scores if s.passed)
        total_score = sum(s.score for s in all_scores)

        return AggregateResult(
            score=total_score / total if total > 0 else 0.0,
            total_score=total_score,
            total_max_score=float(total),
            total=total,
            passed=passed,
        )
