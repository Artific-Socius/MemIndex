# MemIndex

MemIndex 是一个长期记忆基准测试框架，用于评估不同记忆系统（如 Mem0、Memecho 等）在长期记忆任务上的表现。

## 快速开始

### 安装

```bash
git clone <repository_url>
cd MemIndex
uv sync
```

### 配置环境变量

在 `MemIndex/` 目录下创建 `.env` 文件：

```bash
# LiteLLM 自动识别的 API Keys
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
OPENROUTER_API_KEY=sk-xxx

# 记忆服务
MEM0_API_KEY=xxx
MEMECHO_API_KEY=xxx
```

### 运行单个测试

```bash
cd MemIndex

# 使用默认配置
python main.py

# 指定配置
python main.py --agent llm --model openai/gpt-4o --context_window 4096
```

### 批量运行多个测试

```bash
cd MemIndex

# 使用 batch_config.yaml 中的配置
python batch_main.py

# 指定并行数
python batch_main.py --max_parallel 4
```

### 查看报告

```bash
cd MemIndex
python -m utils.report_viewer --port 7860
```

---

## 配置文件

### 单任务配置 (`running_config.yaml`)

```yaml
chat_model: openrouter/google/gemini-2.5-flash  # 被测试的模型
eval_model: openrouter/google/gemini-2.5-flash-lite  # 评估用模型
memory_provider: llm  # Agent 类型: llm, memecho, mem0, mem0_graph
context_window: 4096  # 上下文窗口大小
benchmark_config: ./data/config/1k.json  # 数据集配置
report_dir: ./data/reports  # 报告输出目录
```

### 批量任务配置 (`batch_config.yaml`)

```yaml
# 全局设置
max_parallel: 2  # 同时运行的任务数
continue_on_error: true  # 失败时继续其他任务
task_delay: 0.5  # 任务启动间隔（秒）

# 任务列表
tasks:
  - name: "GPT-4o Mini"
    enabled: true  # 是否启用该任务
    chat_model: openai/gpt-4o-mini
    eval_model: openai/gpt-4o-mini
    memory_provider: llm
    context_window: 4096
    benchmark_config: ./data/config/1k.json
    report_dir: ./data/reports

  - name: "Gemini Flash + Mem0"
    enabled: false  # 禁用该任务
    chat_model: openrouter/google/gemini-2.5-flash
    eval_model: openrouter/google/gemini-2.5-flash-lite
    memory_provider: mem0
    context_window: 4096
    benchmark_config: ./data/config/1k.json
    report_dir: ./data/reports
```

### 系统配置 (`config.yaml`)

```yaml
base_path: ./
env_file: .env
llm_config:
  llm_retry_times: 3
```

---

## 命令行参数

### 单任务 (`main.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 系统配置文件 | `config.yaml` |
| `--running_config` | 运行配置文件 | `running_config.yaml` |
| `--agent` | Agent 类型 | 配置文件值 |
| `--model` | 对话模型 | 配置文件值 |
| `--eval_model` | 评估模型 | 配置文件值 |
| `--context_window` | 上下文窗口 | 配置文件值 |
| `--benchmark_config` | 数据集配置 | 配置文件值 |
| `--report_dir` | 报告目录 | 配置文件值 |

### 批量任务 (`batch_main.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 系统配置文件 | `config.yaml` |
| `--batch_config` | 批量配置文件 | `batch_config.yaml` |
| `--max_parallel` | 最大并行数 | 配置文件值 |
| `--log_level` | 日志级别 | `INFO` |
| `--list` | 列出所有任务（不执行） | - |

**检查配置：**

```bash
python batch_main.py --list
```

输出示例：
```
┏━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Status ┃ Name              ┃ Agent   ┃ Model               ┃
┡━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │  ✓ ON  │ Gemini 2.5 Flash  │ llm     │ gemini-2.5-flash    │
│ 2 │ ✗ OFF  │ Memecho           │ memecho │ gemini-2.5-flash    │
└───┴────────┴───────────────────┴─────────┴─────────────────────┘
Total: 2 tasks  ✓ 1 enabled  ✗ 1 disabled
```

---

## Agent 类型

| 类型 | 说明 |
|------|------|
| `llm` | 纯 LLM Agent，无外部记忆，仅依赖上下文窗口 |
| `memecho` | 使用 Memecho 服务的记忆增强 Agent |
| `mem0` | 使用 Mem0 服务的记忆增强 Agent |
| `mem0_graph` | 使用 Mem0 Graph 的记忆增强 Agent |

---

## 项目结构

```
MemIndex/
├── main.py              # 单任务入口
├── batch_main.py        # 批量任务入口
├── config.yaml          # 系统配置
├── running_config.yaml  # 单任务运行配置
├── batch_config.yaml    # 批量任务配置
│
├── core/                # 核心模块
│   ├── actuator.py      # 测试执行器
│   ├── runner.py        # 测试运行器
│   └── report.py        # 报告生成
│
├── components/          # 组件模块
│   ├── agents/          # Agent 实现
│   ├── chat/            # 对话模块
│   ├── evaluator/       # 评估器
│   └── memory/          # 记忆模块
│
├── utils/               # 工具模块
│   ├── litellm_controller.py  # LLM 控制器
│   ├── data_loader.py   # 数据加载
│   ├── report_viewer.py # 报告查看器
│   └── task_display.py  # 批量任务显示
│
├── config/              # 配置模块
│   ├── config.py        # 系统配置
│   ├── running_config.py # 运行配置
│   └── batch_config.py  # 批量配置
│
└── data/                # 数据目录
    ├── config/          # 数据集配置
    ├── json/            # 测试数据
    └── reports/         # 测试报告
```

---

## 执行流程

```
main.py / batch_main.py
    │
    ├── 加载配置 → 初始化 LLMController → 加载 Agent
    │
    ▼
Runner.run()
    │
    ├── 发送开头提示
    ├── 循环执行 Actuator.step()
    │   ├── 发送消息 (Agent.send_message)
    │   ├── 评分 (LLMEvaluator)
    │   └── 管理记忆距离
    │
    ▼
Report.save()
    │
    └── 生成 JSON 报告
```

---

## 评分系统

支持三种评估方式：

### 1. 二元评估 (默认)
答案正确得满分，错误得 0 分。

```json
{
  "score": {
    "score": 1.0,
    "answer": "正确答案"
  }
}
```

### 2. 加权二元评估 (推荐)
多个独立评分项，各自进行二元判断，按权重计算总分。

```json
{
  "score": {
    "score": 1.0,
    "is_multiple": true,
    "binary_items": [
      {"key": "名字正确", "weight": 0.4, "answer": "Alice"},
      {"key": "年龄正确", "weight": 0.3, "answer": "25"},
      {"key": "城市正确", "weight": 0.3, "answer": "Beijing"}
    ]
  }
}
```

### 3. 多分数评估 (已弃用)
由 LLM 生成各项分数，不够稳定，推荐使用加权二元评估替代。

---

## 扩展开发

### 添加新的 Memory 实现

1. 在 `components/memory/` 创建新类，继承 `BaseMemory`
2. 实现 `initialize`, `search`, `add`, `delete` 方法
3. 在 `__init__.py` 中导出

```python
from .base_memory import BaseMemory

class MyMemory(BaseMemory):
    async def search(self, query: str, **kwargs) -> list:
        # 搜索记忆
        pass
    
    async def add(self, messages: list, **kwargs) -> bool:
        # 添加记忆
        pass
```

### 添加新的 Agent 实现

1. 在 `components/agents/` 创建新类，继承 `BaseAgent`
2. 实现 `send_message` 方法
3. 在 `main.py` 的 `load_agent()` 中添加加载逻辑

```python
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def send_message(self, message: str) -> str:
        # 处理消息并返回响应
        pass
```

---

## 许可证

MIT License
