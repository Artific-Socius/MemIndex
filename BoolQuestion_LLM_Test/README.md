# BoolQ LLM 评估实验

基于 [BoolQ](https://github.com/google-research-datasets/boolean-questions) 数据集的大语言模型二值分类能力评估框架。

## 目录

- [项目概述](#项目概述)
- [核心概念](#核心概念)
- [实验设计理念](#实验设计理念)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [Provider配置](#provider配置)
- [输出文件](#输出文件)
- [绘图工具](#绘图工具)
- [项目结构](#项目结构)
- [开发指南](#开发指南)

---

## 项目概述

本项目旨在评估大语言模型在**阅读理解+二值判断**任务上的表现。通过BoolQ数据集，我们可以：

1. **测量模型准确率** - 模型能否正确理解段落并回答是非问题
2. **分析模型置信度** - 通过LogProbs了解模型对自己答案的确信程度
3. **研究置信度与准确率的关系** - 高置信度是否意味着高准确率？

### 为什么选择BoolQ？

- **任务简单明确**: 给定段落和问题，回答True/False
- **消除格式噪音**: 输出仅为单词，最大化LogProbs信息价值
- **数据质量高**: Google Research发布的标准数据集

---

## 核心概念

### 评估模式 (Eval Mode)

项目支持两种评估模式：

#### `validate` 模式（验证模式）
```
输入: Passage + Question + Answer (人工标注)
任务: 判断人工标注的Answer是否正确
输出: True (标注正确) / False (标注错误)
```

这种模式用于评估模型**验证答案正确性**的能力。我们可以通过`--reversal`参数故意反转部分答案，测试模型能否识别错误标注。

#### `answer` 模式（回答模式）
```
输入: Passage + Question
任务: 直接回答问题
输出: True / False
```

这种模式用于评估模型**直接回答问题**的能力，更接近传统的阅读理解任务。

### Prompt风格 (Style)

#### `direct` - 直接输出
最简洁的格式，模型仅输出 `true` 或 `false`。

**优点**: LogProbs最纯净，无格式干扰  
**缺点**: 无推理过程，难以debug

#### `sse` - 结构化输出
```
answer: True
reason: 1. Cause: 段落提到..., Outcome: 因此...; 2. ...
```

**优点**: 保留推理过程，便于分析  
**缺点**: 格式解析可能失败

#### `json` - JSON格式
```json
{
  "answer": true,
  "reason": "1. Cause: ..., Outcome: ..."
}
```

**优点**: 结构化程度最高，解析稳定  
**缺点**: 格式token较多

### LogProbs（对数概率）

LogProbs是模型输出每个token时的对数概率，反映模型的**置信度**。

```
LogProbs = log(P(token))

值域: (-∞, 0]
- 0: 100%确定
- -0.01: 约99%确定
- -0.1: 约90%确定
- -1.0: 约37%确定
- -2.3: 约10%确定
```

**avg_logprobs**: 输出所有token的平均对数概率，用于衡量整体置信度。

### 答案反转 (Reversal)

在`validate`模式下，`--reversal`参数控制随机反转部分答案的比例。

```bash
--reversal 0.3  # 30%的样本会被反转答案
```

**为什么需要反转？**
- 防止模型学会"总是输出True"的捷径
- 测试模型能否真正理解段落而非猜测
- 平衡正负样本，获得更准确的评估

### 脏数据过滤 (Dirty Data)

某些数据项可能存在问题（标注错误、段落歧义等），我们通过维护一个脏数据列表来跳过这些项：

```
datasets/google_boolq/dirty_data/
├── dirty_data.jsonl      # 主脏数据文件
└── manual_review.jsonl   # 人工标注的问题数据
```

每个脏数据文件包含问题数据的hash值，评估时自动跳过。

---

## 实验设计理念

### 1. 最小化格式干扰

在研究LogProbs与准确率关系时，我们希望LogProbs尽可能反映模型对**答案本身**的置信度，而非对**输出格式**的置信度。

因此：
- `direct`模式是首选，输出仅为单个单词
- 推理内容(`reason`)放在答案之后，不影响答案token的logprobs

### 2. 置信度筛选假设

**核心假设**: 如果模型高置信度的预测更可能正确，那么通过筛选高置信度样本，可以获得更高的准确率（但样本量减少）。

绘图工具生成的**准确率-置信度曲线**用于验证这一假设：
- X轴: 置信度阈值（负对数概率）
- Y轴: 准确率 & 保留比例

理想情况下，随着阈值提高，准确率应该上升。

### 3. 异步高并发设计

评估大量样本时，网络延迟是主要瓶颈。本项目采用：
- **asyncio并发**: 同时处理多个请求
- **信号量控制**: 限制最大并发数，避免触发限流
- **异步写入**: 结果写入不阻塞评估流程

### 4. 多Provider支持

不同LLM提供商的LogProbs支持程度不同：

| Provider | LogProbs支持 | 备注 |
|----------|-------------|------|
| OpenRouter | 部分模型 | 通过LiteLLM调用 |
| Vertex AI | 完整支持 | Gemini模型，使用Google GenAI SDK |
| 火山引擎 | 完整支持 | DeepSeek等模型 |
| OpenAI | 完整支持 | 直接调用 |
| Anthropic | 不支持 | Claude不提供LogProbs |

---

## 快速开始

### 安装

```bash
# 使用uv（推荐）
uv sync
```

### 配置环境变量

创建 `.env` 文件：

```env
# OpenRouter（通用聚合平台）
OPENROUTER_API_KEY=your_key

# OpenAI直连
OPENAI_API_KEY=your_key

# Google Vertex AI / AI Studio
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=location
GOOGLE_API_KEY=your_key

# 字节火山引擎
VOLCENGINE_API_KEY=your_key
```

### 单任务执行

```bash
# 最简单的测试（3条数据）
python boolq_evaluator.py --model openrouter/openai/gpt-4o-mini --style direct --limit 3

# 完整评估（validate模式）
python boolq_evaluator.py --model gemini-2.5-flash --style direct --eval-mode validate

# 直接回答模式
python boolq_evaluator.py --model openrouter/openai/gpt-4o --style direct --eval-mode answer --limit 100
```

### 批量执行

通过YAML配置文件批量顺序执行多个评估任务：

```bash
# 使用默认配置文件 (batch_config.yaml)
python main.py

# 指定配置文件
python main.py --config my_config.yaml

# 列出所有任务
python main.py --list

# 只执行指定任务
python main.py --task 0        # 执行第0个任务
python main.py --task 0,2,3    # 执行多个任务
```

配置文件示例 (`batch_config.yaml`)：

```yaml
global:
  lang: auto
  output_dir: outputs

tasks:
  - name: "Gemini Flash Direct"
    model: gemini-2.5-flash
    style: direct
    eval_mode: validate
    limit: 100
    concurrency: 10
    enabled: true

  - name: "GPT-4o SSE"
    model: openrouter/openai/gpt-4o
    style: sse
    eval_mode: answer
    limit: 50
    enabled: true
```

---

## 命令行参数

### 单任务模式 (boolq_evaluator.py)

```bash
python boolq_evaluator.py --help
```

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | `-m` | - | 模型名称（必需） |
| `--style` | `-s` | - | Prompt风格: `direct`, `sse`, `json`（必需） |
| `--eval-mode` | `-e` | `validate` | 评估模式: `validate`, `answer` |
| `--limit` | `-l` | `0` | 限制数据条数，0表示全部 |
| `--concurrency` | `-c` | `10` | 最大并发数 |
| `--reversal` | `-r` | `0.3` | 答案反转比例（仅validate模式） |
| `--reasoning` | - | `False` | 启用详细推理 |
| `--reason-order` | - | `reason-after` | 推理顺序 |
| `--split` | - | `validation` | 数据集分割 |
| `--lang` | - | 自动检测 | 界面语言: `zh`, `en` |

### 批量模式 (main.py)

```bash
python main.py --help
```

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | `-c` | `batch_config.yaml` | YAML配置文件路径 |
| `--list` | `-l` | - | 列出所有任务 |
| `--task` | `-t` | - | 执行指定任务索引（逗号分隔） |
| `--lang` | - | 自动检测 | 覆盖配置中的语言设置 |

### 模型名称格式

遵循LiteLLM规范：

```bash
# OpenAI直连
--model openai/gpt-4o
--model gpt-4o  # 无前缀默认OpenAI

# 通过OpenRouter
--model openrouter/openai/gpt-4o
--model openrouter/anthropic/claude-3.5-sonnet

# Vertex AI（Gemini模型自动识别）
--model gemini-2.5-flash
--model gemini-2.0-flash-001

# Azure
--model azure/my-deployment

# 本地Ollama
--model ollama/llama3
```

---

## Provider配置

### OpenRouter（推荐入门）

最简单的方式，支持多种模型：

```bash
# 设置环境变量
OPEN_ROUTER_API_KEY=your_key
# 或 OPENROUTER_API_KEY=your_key（LiteLLM标准格式）

# 使用
python main.py --model openrouter/openai/gpt-4o-mini --style direct
```

### Vertex AI（Gemini完整LogProbs）

如果需要完整的LogProbs支持，使用Google Vertex AI：

```bash
# 设置环境变量
GOOGLE_API_KEY=your_key

# 使用（gemini模型自动识别为Vertex AI）
python main.py --model gemini-2.5-flash --style direct
```

### 火山引擎（DeepSeek）

```bash
# 设置环境变量
VOLCANO_API_KEY=your_key

# 使用
python main.py --model deepseek-v3-250324 --style direct
```

---

## 输出文件

评估结果保存在 `outputs/` 目录：

### 文件命名

```
{model}_{style}_{reason_order}_{timestamp}_{uuid}.jsonl
{model}_{style}_{reason_order}_{timestamp}_{uuid}.log
```

示例：
```
gpt-4o-mini_direct_direct_20251201_020112_345c3e5f.jsonl
gpt-4o-mini_direct_direct_20251201_020112_345c3e5f.log
```

### JSONL格式

每行一个JSON对象：

```json
{
  "status": "success",
  "question": "Is the sky blue?",
  "passage": "The sky appears blue due to...",
  "expected": true,
  "predicted": true,
  "is_correct": true,
  "is_reversal": false,
  "raw_response": "true",
  "parsed_reason": "",
  "latency": 0.85,
  "avg_logprobs": -0.0012,
  "confidence": 0.9988,
  "index": 42,
  "item_hash": "abc123...",
  "token_usage": {"prompt_tokens": 350, "completion_tokens": 1, "total_tokens": 351},
  "cost_info": {"total_cost": 0.000053}
}
```

### 关键字段说明

| 字段 | 说明 |
|------|------|
| `status` | `success` / `parse_error` / `api_error` |
| `expected` | 原始数据集的正确答案 |
| `predicted` | 模型预测的答案 |
| `is_correct` | 预测是否正确 |
| `is_reversal` | 是否为反转样本（validate模式） |
| `avg_logprobs` | 平均对数概率（置信度指标） |
| `confidence` | 置信度 = exp(avg_logprobs) |

---

## 绘图工具

生成准确率-置信度曲线图，默认输出到 `outputs/images/`：

```bash
# 基本用法（输出到 outputs/images/accuracy_vs_logprob.png）
python -m tools.plot_results

# 指定输入目录
python -m tools.plot_results --input outputs/

# 指定输出文件路径
python -m tools.plot_results --output my_plot.png

# 过滤异常值（IQR方法）
python -m tools.plot_results --filter-outliers

# 自定义IQR倍数（更宽松）
python -m tools.plot_results --filter-outliers --iqr-k 2.0

# 显示所有实验（不去重）
python -m tools.plot_results --all

# 输出统计到JSON
python -m tools.plot_results --json stats.json

# 绘制后打开图片
python -m tools.plot_results --show
```

### 曲线解读

生成的图表包含两条曲线：

1. **准确率曲线（实线）**: 在给定置信度阈值下，保留样本的准确率
2. **保留比例曲线（虚线）**: 在给定置信度阈值下，保留的样本比例

**理想结果**: 准确率随阈值提高而上升，且保留比例下降平缓。

---

## 项目结构

```
BoolQuestion_LLM_Test/
├── main.py                     # 批量执行入口（简洁）
├── boolq_evaluator.py          # 单任务评估器
├── batch_runner.py             # 批量任务执行逻辑
├── batch_config.yaml           # 批量任务配置示例
├── config.py                   # 配置管理（LLMConfig, ExperimentConfig, BatchConfig）
│
├── llm_client/                 # LLM客户端抽象层
│   ├── base.py                 # 基类（重试逻辑、成本计算）
│   ├── litellm_client.py       # LiteLLM客户端（100+Provider）
│   └── vertex_client.py        # Vertex AI客户端（Gemini LogProbs）
│
├── prompt_manager/             # Prompt管理
│   ├── prompt_manager.py       # Prompt组装
│   └── prompts.yaml            # Prompt模板配置
│
├── utils/                      # 工具模块
│   ├── utils.py                # 统计工具、进度管理
│   ├── dataset_manager.py      # BoolQ数据集加载、脏数据过滤
│   ├── response_parser.py      # 响应解析（direct/sse/json格式）
│   └── logger.py               # Rich日志、异步写入器
│
├── i18n/                       # 国际化
│   ├── __init__.py             # 翻译函数、语言检测
│   └── translations.py         # 中英翻译字典
│
├── tools/                      # 命令行工具
│   └── plot_results.py         # 绘图工具
│
├── datasets/                   # 数据集相关
│   └── google_boolq/
│       └── dirty_data/         # 脏数据列表
│
├── outputs/                    # 输出目录
│   ├── *.jsonl                 # 评估结果数据
│   ├── *.log                   # 运行日志
│   └── images/                 # 绘图输出
│
└── tests/                      # 单元测试
```

---

## 开发指南

### 添加新的LLM Provider

1. 在 `llm_client/` 创建新的客户端类，继承 `BaseLLMClient`
2. 实现 `generate` 异步方法
3. 在 `models.py` 添加 `LLMProvider` 枚举值
4. 在 `config.py` 更新 `_detect_provider` 方法
5. 在 `llm_client/base.py` 更新 `create_llm_client` 工厂函数

### 添加新的Prompt风格

1. 在 `prompt_manager/prompts.yaml` 添加新的风格配置
2. 在 `models.py` 的 `PromptStyle` 枚举添加新选项
3. 在 `response_parser.py` 添加对应的解析逻辑

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_response_parser.py -v
```

### 重试机制

项目内置了两层重试：

1. **LLM层重试**: 针对速率限制、网络错误、临时服务错误
   - 配置: `max_retries=5`, `base_retry_delay=5s`
   - 策略: 指数退避

2. **解析层重试**: 针对解析失败（模型输出格式不符）
   - 配置: `PARSE_RETRY_COUNT=2`
   - 策略: 重新调用LLM

---

## License

MIT
