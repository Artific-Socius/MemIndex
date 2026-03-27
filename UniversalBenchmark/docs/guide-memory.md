# 记忆后端实现指南

## 为什么需要记忆抽象

框架中的 Agent 由两部分组成：**LLM**（语言模型）和 **Memory**（记忆系统）。

Benchmark 评测的目标是记忆系统本身——给定相同的 LLM，不同记忆后端产生不同的评测结果。因此记忆必须是一个可替换的抽象接口。

## 接口定义

所有记忆后端继承 `MemoryMixin`（`agent/memory/base.py`）。

### 必须实现：3 个抽象方法

```python
from agent.memory.base import MemoryMixin

class MyMemory(MemoryMixin):

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        """接收用户输入，返回完整的 OpenAI 格式消息列表。

        返回值直接传给 LLM。你需要：
        1. 将 user_input 记录到内部状态
        2. 组装 system prompt + 历史消息 + 当前用户消息
        3. 返回 [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """

    def add_response(self, content: str) -> None:
        """将 LLM 生成的助手回复写入记忆。

        在 get_messages() 之后、下一轮对话之前调用。
        """

    def reset(self) -> None:
        """清空对话状态，准备运行下一个 Scene。

        每个 Scene 开始前 Runner 会调用此方法。
        保留配置（API key 等），只清空对话历史。
        """
```

这三个方法定义了一个完整的对话循环：`get_messages` → LLM 生成 → `add_response` → 下一轮。

### 可选覆写：性能优化

```python
def bulk_import(self, conversations: list[tuple[str, str]]) -> int:
    """批量导入对话历史。

    默认实现逐条调用 get_messages + add_response，性能不佳。
    如果你的后端支持批量操作，覆写此方法。

    参数: [(user_msg, assistant_msg), ...]
    返回: 导入条数
    """

def import_corpus(self, documents: list[str], corpus_id: str = "") -> str:
    """导入语料文档（长期记忆系统专用）。

    当 Benchmark 的 Scene 提供 background_text 时，
    默认实现将所有文档合并后走 bulk_import。
    长期记忆系统（如 Memecho）应覆写此方法，
    实现文件级导入以获得更好的检索质量。

    返回: 语料库标识符
    """

def ensure_memory_library(self) -> str:
    """确保存在可用记忆库并返回其 id。

    默认实现返回框架 fallback id；长期记忆后端应覆写为真实 id。
    """

def get_memory_library_id(self) -> str:
    """返回当前记忆库 id（默认调用 ensure_memory_library）。"""
```

## 最小实现示例

一个只保留最近 N 条消息的简易记忆：

```python
from typing import Any, Optional
from agent.memory.base import MemoryMixin

class SimpleWindowMemory(MemoryMixin):

    def __init__(self, window: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history: list[dict[str, Any]] = []
        self._window = window

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        self._history.append({"role": "user", "content": user_input})
        msgs = self._history[-self._window * 2 :]
        if self._system_prompt:
            return self._build_system_messages() + msgs
        return msgs

    def add_response(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})

    def reset(self) -> None:
        self._history.clear()
```

## 注册到 CLI

在 `run_benchmark_lite.py` 中添加一行：

```python
MEMORY_TYPES: dict[str, Type[MemoryMixin]] = {
    "buffer": BufferMemory,
    "memecho": MemechoMemory,
    "mem0": Mem0Memory,
    "my_memory": SimpleWindowMemory,  # ← 加在这里
}
```

然后通过 `--memory my_memory` 使用。

## 环境配置（可选）

如果你的记忆后端需要 API key 或其他配置，在 `env_config.yaml` 中添加：

```yaml
my_memory:
  api_key: "sk-..."
  base_url: "https://..."
```

并在 `agent/env_config.py` 的 `get_memory_config()` 中读取。

## Memecho 特别说明

使用 Memecho 时，必须在 `env_config.yaml` 中设置 `X-Mark` 请求头，用于在服务端标识请求来源。格式为最多三层、`/` 分隔：

```yaml
memory:
  memecho:
    custom_headers:
      X-Mark: your_name/benchmark_name/run_id
```

| 层级 | 含义 | 必填 | 示例 |
|------|------|------|------|
| 第一层 | 你的名字或团队名 | 是 | `zhangsan` |
| 第二层 | 数据集或实验名 | 否 | `zhangsan/EverMemBench` |
| 第三层 | 具体运行标识 | 否 | `zhangsan/EverMemBench/run1` |

不配置此项将无法在服务端区分你的请求日志。首次使用请复制 `env_config.yaml.example` 为 `env_config.yaml` 并修改。

## Message ID 追踪协议

Benchmark 结果中每个回合会记录 `message_trace`（用户消息 id、助手消息 id、id 来源等），方便记忆提供者在后端对账和定性分析。

### 默认行为

框架在每轮 `Agent.chat()` 调用前，通过 `MemoryMixin._init_turn_trace()` 自动生成 UUID 作为兜底 id（`id_source = "framework"`）。如果你的记忆后端没有自己的消息标识体系，**无需做任何额外操作**，框架会自动处理。

### 提供真实 Message ID

如果你的记忆后端在 query / append 时会返回服务端确认的消息 id，你可以让这些真实 id 出现在 benchmark 结果中。做法是在 `get_messages()` 和/或 `add_response()` 中更新 `self._current_turn_trace`：

```python
from agent.memory.base import MemoryMixin, TurnTrace
from typing import Any
import uuid

class MyCloudMemory(MemoryMixin):

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        query_request_id = str(uuid.uuid4())

        # 发送 query 请求给你的后端
        result = self._call_backend_query(user_input, request_id=query_request_id)

        if self._current_turn_trace is not None:
            # 记录查询 request id
            self._current_turn_trace.query_request_id = query_request_id

            # 后端返回了真实的 user message id
            server_user_id = result.get("user_message_id", "")
            if server_user_id:
                self._current_turn_trace.user_message_id = server_user_id
                self._current_turn_trace.id_source = "provider"

        return self._build_openai_messages(result)

    def add_response(self, content: str) -> None:
        append_request_id = str(uuid.uuid4())

        result = self._call_backend_append(content, request_id=append_request_id)

        if self._current_turn_trace is not None:
            # 记录追加 request id
            self._current_turn_trace.append_request_id = append_request_id

            server_assistant_id = result.get("assistant_message_id", "")
            if server_assistant_id:
                self._current_turn_trace.assistant_message_id = server_assistant_id
                self._current_turn_trace.id_source = "provider"

    # ... 其他方法 ...
```

### TurnTrace 数据结构

`TurnTrace` 定义在 `agent/memory/base.py`。一轮对话包含两次请求（查询 + 追加），各自有独立的 request id：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_message_id` | `str` | 用户消息 id（provider 真实 id 或框架 UUID） |
| `assistant_message_id` | `str` | 助手消息 id（同上） |
| `id_source` | `str` | `"provider"` 或 `"framework"`，标注 id 来源 |
| `query_request_id` | `str` | 查询请求（`get_messages`）的 request id |
| `append_request_id` | `str` | 追加请求（`add_response`）的 request id |
| `extra` | `dict` | 其他后端追踪信息（可选） |

在最终导出的 JSON 结果中，每个 turn 下会有对应的 `message_trace` 字段（参见 [guide-benchmark-lite.md](guide-benchmark-lite.md) 中的「结果 Pydantic 模型」章节）。

此外，评估回合会追加 `depends_on_turn_indices` 与 `dependency_policy`：

- `depends_on_turn_indices`：该评估回合依赖的前序 turn 索引列表。
- `dependency_policy`：依赖生成策略（例如 `ref`、`subtest_prefix_fallback`）。

这样可以把 Memory 后端的 `message_trace`（消息级追踪）与
Benchmark 的依赖链（回合级追踪）联合起来分析：既知道“评分依赖了哪些历史回合”，也能对齐这些回合在后端中的真实消息 id / request id。

## 记忆库 ID 追踪协议

框架会在每个场景结果下记录 `memory_library_id`，并在顶层
`metadata.scenario_memory_library_ids` 中给出场景到库 id 的映射。

- 默认实现（`MemoryMixin`）返回稳定的框架 fallback id（`framework-...`）。
- `MemechoMemory` 会返回真实 `memory_lib_id`。
- `Mem0Memory` 会返回真实 `user_id`（作为记忆命名空间 id）。

如果你的后端有独立库/命名空间概念，建议覆写
`ensure_memory_library()` / `get_memory_library_id()`，便于结果对账与离线分析。

## 现有实现参考

| 实现 | 文件 | 特点 |
|------|------|------|
| `BufferMemory` | `agent/memory/buffer.py` | 最简单：本地列表，可选滑动窗口，使用框架 UUID |
| `MemechoMemory` | `agent/memory/memecho.py` | 云 API：远程记忆管理，支持 `import_corpus` 语料导入，支持 provider 真实 message id |
| `Mem0Memory` | `agent/memory/mem0.py` | Mem0 云 API：事实性记忆提取，使用框架 UUID |
