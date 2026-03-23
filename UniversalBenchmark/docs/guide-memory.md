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

## 现有实现参考

| 实现 | 文件 | 特点 |
|------|------|------|
| `BufferMemory` | `agent/memory/buffer.py` | 最简单：本地列表，可选滑动窗口 |
| `MemechoMemory` | `agent/memory/memecho.py` | 云 API：远程记忆管理，支持 `import_corpus` 语料导入 |
