# Benchmark Lite — 开发者指南

## 你的 Benchmark 需要做什么

你有一个评估 Agent 记忆能力的测试方案。你想把它接入这个框架，让它能直接跑。

你只需要做两件事：

1. **描述你的测试流程** — 告诉框架"发什么消息给 Agent、怎么判断对错"
2. **汇总分数** — 告诉框架"最终得分怎么算"

框架会负责：创建 Agent、驱动对话、收集结果、生成报告。

---

## 最小实现

```python
from benchmark_lite import (
    AggregateResult,
    BenchmarkLite,
    Scenario,
    ScenarioResult,
    Turn,
    TurnResult,
    TurnScore,
    TurnType,
)


class MyBenchmark(BenchmarkLite):

    @property
    def name(self) -> str:
        return "MyBenchmark"

    def get_scenarios(self):
        """定义测试场景：先说一句话，再提一个问题。"""
        return [
            Scenario(id="test_1", turns=[
                Turn("我叫小明"),                                          # 对话
                Turn("我叫什么？", TurnType.EVALUATION, reference="小明"),   # 评估
            ]),
        ]

    def evaluate(self, turn, response, history):
        """判分：回复里有没有正确答案。"""
        found = turn.reference in response
        return TurnScore(score=1.0 if found else 0.0, passed=found)

    def aggregate(self, scenario_results):
        """汇总：统计通过率。"""
        scores = [ts for sr in scenario_results for ts in sr.eval_scores]
        total = len(scores)
        passed = sum(1 for s in scores if s.passed)
        return AggregateResult(
            score=passed / total if total else 0.0,
            total_score=sum(s.score for s in scores),
            total_max_score=float(total),
            total=total,
            passed=passed,
        )
```

运行：

```bash
cd UniversalBenchmark

python run_benchmark.py \
    --benchmark my_module.MyBenchmark \
    --memory buffer \
    --model openrouter/google/gemini-2.5-flash-lite
```

就这些。下面是详细说明。

---

## 三种场景模式

根据你的测试数据长什么样，选一种模式。

### 模式 A — 脚本化场景

**适用于：** 你有固定的对话脚本。先说几句话注入信息，再问问题检验。

```python
Scenario(
    id="recall_name",
    turns=[
        Turn("我叫小明，今年25岁"),               # CONVERSATION（默认）
        Turn("今天天气真好"),                       # CONVERSATION
        Turn("我叫什么？",                         # EVALUATION
             turn_type=TurnType.EVALUATION,
             reference=["小明"]),
    ],
)
```

你需要实现 `evaluate()` 方法来判分。

### 模式 B — 预置历史

**适用于：** 你的数据集本身就是一段对话记录 + 问题。Agent 不需要生成历史对话，直接记住然后回答。

```python
Scenario(
    id="history_qa",
    preload_history=[
        HistoryTurn("我叫张三", "你好张三！"),
        HistoryTurn("我住在北京", "北京是个好地方"),
    ],
    turns=[
        Turn("我叫什么？住在哪？",
             turn_type=TurnType.EVALUATION,
             reference=["张三", "北京"]),
    ],
)
```

`preload_history` 中的对话会通过 Agent 的 `bulk_import()` 接口批量写入记忆，不经过 LLM。之后的 `turns` 正常运行。数据量大时效率远高于逐条导入。

判分同样靠 `evaluate()` 方法。

### 模式 C — 交互式场景

**适用于：** 对话流程不是固定的。你需要根据 Agent 的回复动态决定下一步，或者整个对话结束后才能评估。

```python
from benchmark_lite import InteractiveScenario, ScenarioScore

class MyInteractive(InteractiveScenario):

    @property
    def id(self):
        return "dynamic_test"

    def next_turn(self, history):
        """每次被调用时，决定下一轮说什么。返回 None 表示结束。"""
        if len(history) >= 10:
            return None
        return Turn("下一个问题...")

    def evaluate(self, history):
        """对话全部结束后，回顾所有记录，打分。"""
        return ScenarioScore(score=0.8, passed=True)
```

然后在你的 `BenchmarkLite` 子类的 `get_scenarios()` 里返回它：

```python
def get_scenarios(self):
    return [MyInteractive()]
```

交互式场景不需要实现 `BenchmarkLite.evaluate()`，因为评估逻辑在场景自身的 `evaluate()` 里。

---

## 你必须实现的方法

| 方法 | 何时需要 | 作用 |
|------|---------|------|
| `name` (property) | 总是 | Benchmark 名称，显示在报告里 |
| `get_scenarios()` | 总是 | 返回场景列表 |
| `evaluate(turn, response, history)` | 模式 A/B | 对单个评估回合判分 |
| `aggregate(scenario_results)` | 总是 | 把所有场景的分数汇总成最终结果 |

---

## AggregateResult — 分数汇报格式

`aggregate()` 必须返回一个 `AggregateResult`，它有固定的字段：

```python
AggregateResult(
    score=0.75,           # 最终得分 (0~1)，你定义它的含义
    total_score=6.0,      # 原始累加得分
    total_max_score=8.0,  # 最大可能得分
    total=8,              # 评估点总数
    passed=6,             # 通过数
    detail="...",         # 可选，人类可读的总结
    extra={...},          # 可选，你的 benchmark 特有的指标
)
```

框架会这样展示：

```
  Score       : 75.00%  (6.0000 / 8.0000)
  Evaluations : 6 / 8 passed  (75.00%)
```

`extra` 里的内容会作为附加行显示。

---

## 数据流

```
你的代码                           框架代码
────────                         ──────────

get_scenarios()  ──→  Runner 遍历每个场景
                          │
                          ├─ Scenario?
                          │    ├─ 预载 preload_history (如果有)
                          │    ├─ 逐个执行 Turn
                          │    │    ├─ CONVERSATION → agent.chat()
                          │    │    └─ EVALUATION  → agent.chat() → evaluate()
                          │    └─ 收集 ScenarioResult
                          │
                          └─ InteractiveScenario?
                               ├─ 循环调用 next_turn() → agent.chat()
                               ├─ next_turn() 返回 None → 结束
                               ├─ 调用 evaluate() → ScenarioScore
                               └─ 收集 ScenarioResult

aggregate(所有场景结果)  ──→  AggregateResult  ──→  报告 / JSON
```

---

## 数据结构速查

**输入侧（你构造的）：**

| 结构 | 用途 |
|------|------|
| `Turn(user_input, turn_type, reference)` | 一轮对话。`reference` 是评估依据，类型任意 |
| `HistoryTurn(user_message, assistant_response)` | 预置历史中的一轮 |
| `Scenario(id, turns, preload_history)` | 脚本化场景 |

**输出侧（你填充的）：**

| 结构 | 用途 |
|------|------|
| `TurnScore(score, passed, detail)` | 单轮评分。`evaluate()` 返回这个 |
| `ScenarioScore(score, passed, turn_annotations)` | 场景整体评分。`InteractiveScenario.evaluate()` 返回这个 |
| `AggregateResult(score, total_score, ...)` | 最终汇总。`aggregate()` 返回这个 |

**框架生成的（你在 aggregate 里读取的）：**

| 结构 | 用途 |
|------|------|
| `TurnResult(turn_index, user_input, response, score)` | 一轮的完整记录 |
| `ScenarioResult(scenario_id, turn_results, scenario_score)` | 一个场景的完整记录 |
| `ScenarioResult.eval_scores` | 便捷属性：所有 TurnScore 的列表 |
| `ScenarioResult.eval_count` / `.passed_count` | 便捷属性：评估数 / 通过数 |

---

## 运行命令

```bash
cd UniversalBenchmark

python run_benchmark.py \
    --benchmark <你的类的点分路径> \
    --memory buffer \
    --model openrouter/google/gemini-2.5-flash-lite \
    -v                   # 显示每轮详情
    -o results.json      # 保存 JSON
```

`--memory` 可选值：`buffer`、`mem0`、`memecho`

`--benchmark` 的值是 Python 的 import 路径，例如：
- `benchmark_lite.examples.SimpleMemoryQA`
- `benchmark_lite.benchmarks.memindex.MemIndexBenchmark`
- `my_package.my_module.MyBenchmark`

---

## 文件放在哪

推荐放在 `benchmark_lite/benchmarks/` 下：

```
benchmark_lite/
  benchmarks/
    my_bench/
      __init__.py        ← 导出你的 BenchmarkLite 子类
      benchmark.py       ← 主类
      ...                ← 你的数据加载、评估逻辑等
```

`__init__.py` 示例：

```python
from .benchmark import MyBenchmark
__all__ = ["MyBenchmark"]
```

然后用 `--benchmark benchmark_lite.benchmarks.my_bench.MyBenchmark` 运行。

也可以放在项目外的任何 Python 可导入位置。
