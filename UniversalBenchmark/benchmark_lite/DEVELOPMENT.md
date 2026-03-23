# Benchmark Lite — 开发者指南

> **本文档已迁移。** 完整的开发者文档体系位于 [`docs/`](../docs/README.md)：
>
> - [总览](../docs/README.md)
> - [记忆后端实现指南](../docs/guide-memory.md)
> - [数据提供者实现指南](../docs/guide-benchmark-data.md)
> - [自定义 Benchmark 实现指南](../docs/guide-benchmark-lite.md)
>
> 以下为旧版内容，仅作存档参考。

---

## 概述

本框架将 Benchmark 评测分为两层：

- **数据层** (`benchmark/`) — 加载和标准化各种数据集
- **逻辑层** (`benchmark_lite/`) — 执行评测、打分、生成报告

开发者可以选择两种接入方式：

| 方式 | 适用场景 | 你要做的 |
|------|---------|---------|
| **简单模式** — 只提供数据 | 你有一个数据集，想用框架已有的打分器跑 | 实现 `benchmark.interfaces` 中的 `Benchmark` + `Scene` + `Question` |
| **高级模式** — 自定义执行逻辑 | 你需要自定义对话调度或打分方式 | 实现 `benchmark_lite.BenchmarkLite` 子类 |

---

## 简单模式：只提供数据（推荐）

只需要在 `benchmark/` 中实现数据接口，框架通过 `UniversalAdapter` 自动将其转换为可执行的 Benchmark。

### 最小实现

```python
# benchmark/data/providers/my_dataset/my_benchmark.py

from benchmark.interfaces import Benchmark, Scene, Question, EvidenceBundle, ScoringConfig, ConversationTurn

class MyScene(Scene):
    @property
    def scene_id(self) -> str:
        return "scene_0"

    @property
    def scene_name(self) -> str:
        return "基本记忆测试"

    def conversation_history(self) -> list[ConversationTurn]:
        """提供对话历史（会自动注入 Agent 的记忆）。"""
        return [
            ConversationTurn("我叫小明，今年25岁", "你好小明！"),
            ConversationTurn("我住在北京", "北京是个好地方。"),
        ]

    def questions(self):
        """提供评估问题。"""
        yield Question(
            question_id="q0",
            question_text="我叫什么名字？",
            ground_truth="小明",
            evidence=EvidenceBundle(evidence_type="none", payload={}),
            scoring=ScoringConfig(eval_mode="keyword", eval_prompt_key="default"),
        )

class MyBenchmark(Benchmark):
    @property
    def benchmark_name(self) -> str:
        return "MyDataset"

    def list_scenes(self):
        return ["scene_0"]

    def get_scene(self, scene_id: str) -> Scene:
        return MyScene()
```

### 运行

```bash
cd UniversalBenchmark

python run_benchmark_lite.py \
    --benchmark benchmark.data.providers.my_dataset.my_benchmark.MyBenchmark \
    --memory buffer \
    --model openrouter/google/gemini-2.5-flash-lite
```

框架会自动：
1. 检测到这是 `benchmark.interfaces.Benchmark` 的子类
2. 用 `UniversalAdapter` 包装它
3. 将 `conversation_history()` 注入 Agent 记忆
4. 用 `ScoringConfig.eval_mode` 指定的打分器评估每个问题
5. 生成报告

### Scene 提供数据的两种方式

| 方式 | 方法 | 适用场景 |
|------|------|---------|
| 对话历史 | `conversation_history()` | 记忆测试：先聊天再提问 |
| 背景语料 | `background_text()` | RAG/检索测试：给定文档再提问 |

两种方式可以共存。适配器的处理优先级：
1. 有 `conversation_history()` → 转为 `preload_history`（逐轮注入 Agent 记忆）
2. 无对话历史但有 `background_text()` → 作为一整段文本注入

### 内置打分器 (`eval_mode`)

在 `ScoringConfig.eval_mode` 中指定：

| eval_mode | 说明 | 需要 LLM |
|-----------|------|---------|
| `keyword` | 关键词匹配（`ground_truth` 中的关键词是否出现在回复中） | 否 |
| `binary` | LLM 判断正确/错误 | 是 |
| `score` | LLM 给出 0-1 连续分数 | 是 |
| `multi_score` | LLM 对多个评分点分别打分 | 是 |
| `weighted_binary` | 多项加权二元评判 | 是 |

LLM 评估使用 `--eval-model` 参数指定的模型。

### 文件放在哪

```
benchmark/
  data/
    providers/
      my_dataset/            ← 你的数据集
        __init__.py           ← 导出 MyBenchmark
        my_benchmark.py       ← 实现 Benchmark / Scene / Question
        ...                   ← 数据文件、加载逻辑等
```

---

## 高级模式：自定义执行逻辑

如果你需要完全控制对话调度和评分方式，直接实现 `BenchmarkLite` 子类。

### 最小实现

```python
from benchmark_lite import (
    AggregateResult, BenchmarkLite, Scenario, ScenarioResult,
    Turn, TurnResult, TurnScore, TurnType,
)

class MyBenchmark(BenchmarkLite):

    @property
    def name(self) -> str:
        return "MyBenchmark"

    def get_scenarios(self):
        return [
            Scenario(id="test_1", turns=[
                Turn("我叫小明"),
                Turn("我叫什么？", TurnType.EVALUATION, reference="小明"),
            ]),
        ]

    def evaluate(self, turn, response, history):
        found = turn.reference in response
        return TurnScore(score=1.0 if found else 0.0, passed=found)

    def aggregate(self, scenario_results):
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

### 三种场景模式

#### 模式 A — 脚本化场景

预定义 Turn 列表，逐回合评估。

```python
Scenario(
    id="recall_name",
    turns=[
        Turn("我叫小明，今年25岁"),
        Turn("今天天气真好"),
        Turn("我叫什么？", turn_type=TurnType.EVALUATION, reference=["小明"]),
    ],
)
```

#### 模式 B — 预置历史

对话历史已给定，Agent 直接回答问题。

```python
Scenario(
    id="history_qa",
    preload_history=[
        HistoryTurn("我叫张三", "你好张三！"),
        HistoryTurn("我住在北京", "北京是个好地方"),
    ],
    turns=[
        Turn("我叫什么？住在哪？", turn_type=TurnType.EVALUATION, reference=["张三", "北京"]),
    ],
)
```

#### 模式 C — 交互式场景

动态生成回合，事后评估。

```python
from benchmark_lite import InteractiveScenario, ScenarioScore

class MyInteractive(InteractiveScenario):
    @property
    def id(self):
        return "dynamic_test"

    def next_turn(self, history):
        if len(history) >= 10:
            return None
        return Turn("下一个问题...")

    def evaluate(self, history):
        return ScenarioScore(score=0.8, passed=True)
```

### 你必须实现的方法

| 方法 | 何时需要 | 作用 |
|------|---------|------|
| `name` (property) | 总是 | Benchmark 名称 |
| `get_scenarios()` | 总是 | 返回场景列表 |
| `evaluate(turn, response, history)` | 模式 A/B | 对单个评估回合判分 |
| `aggregate(scenario_results)` | 总是 | 汇总所有场景的分数 |

### 文件放在哪

```
benchmark_lite/
  benchmarks/
    my_bench/
      __init__.py
      benchmark.py
      ...
```

---

## 注册自定义打分器

如果内置的 `eval_mode` 不够用，可以注册自定义打分器：

```python
from benchmark_lite.evaluators import BaseEvaluator, register_evaluator
from benchmark_lite.types import TurnScore

@register_evaluator("my_custom_eval")
class MyEvaluator(BaseEvaluator):
    def __init__(self, model: str = "", **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, question_text, ground_truth, response, max_score=1.0, evidence=None):
        # 你的打分逻辑
        score = ...
        return TurnScore(score=score, passed=score > 0.5, detail="...")
```

然后在 `ScoringConfig` 中使用 `eval_mode="my_custom_eval"`。

确保你的 evaluator 模块在运行前被 import（例如放在 `benchmark_lite/evaluators/` 下，
或在你的 benchmark 模块的顶层 import 它）。

---

## 完整数据流

```
简单模式 (数据层 Benchmark)           高级模式 (BenchmarkLite)
──────────────────────              ──────────────────────

benchmark.interfaces.Benchmark       BenchmarkLite 子类
        │                                    │
        ▼                                    │
  UniversalAdapter                           │
  (自动转换 Scene → Scenario)                │
        │                                    │
        └──────────┬─────────────────────────┘
                   │
                   ▼
              Runner.run()
                   │
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
    Scenario   Scenario    InteractiveScenario
    (脚本化)   (预置历史)      (动态交互)
        │          │              │
        └──────────┼──────────────┘
                   │
                   ▼
            AggregateResult → 报告 / JSON
```

---

## 运行命令

```bash
cd UniversalBenchmark

python run_benchmark_lite.py \
    --benchmark <类的点分路径> \
    --memory buffer \
    --model openrouter/google/gemini-2.5-flash-lite \
    --eval-model openrouter/google/gemini-2.5-flash \
    --scene-ids 0 1 2 \
    -v \
    -o results.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--benchmark` | Benchmark 类的 import 路径 | (必填) |
| `--memory` | Agent 的 Memory 类型 | `buffer` |
| `--model` | Agent 使用的 LLM 模型 | `openrouter/google/gemini-2.5-flash-lite` |
| `--eval-model` | 用于评估的 LLM 模型（简单模式） | `openrouter/google/gemini-2.5-flash` |
| `--scene-ids` | 指定运行的 Scene ID（简单模式） | 全部 |
| `-v` | 显示每轮详情 | 关闭 |
| `-o` | 保存 JSON 结果 | 不保存 |

`--memory` 可选值：`buffer`、`mem0`、`memecho`

`--benchmark` 示例：
- `benchmark_lite.examples.SimpleMemoryQA` (高级模式)
- `benchmark_lite.benchmarks.memindex.MemIndexBenchmark` (高级模式)
- `benchmark.data.providers.evermind_ai.evermembench_static.EverMemBenchStaticBenchmark` (简单模式)

---

## 数据结构速查

**输入侧：**

| 结构 | 用途 |
|------|------|
| `Turn(user_input, turn_type, reference)` | 一轮对话。`reference` 是评估依据 |
| `HistoryTurn(user_message, assistant_response)` | 预置历史中的一轮 |
| `Scenario(id, turns, preload_history)` | 脚本化场景 |
| `ConversationTurn(user_message, assistant_response)` | 数据层 Scene 的对话历史 |
| `Question(question_id, question_text, ground_truth, evidence, scoring)` | 数据层的评估问题 |
| `ScoringConfig(eval_mode, eval_prompt_key, max_score)` | 打分配置 |

**输出侧：**

| 结构 | 用途 |
|------|------|
| `TurnScore(score, passed, detail)` | 单轮评分 |
| `ScenarioScore(score, passed, turn_annotations)` | 场景整体评分 |
| `AggregateResult(score, total_score, ...)` | 最终汇总 |
| `TurnResult` / `ScenarioResult` / `BenchmarkResult` | 框架生成的运行记录 |
