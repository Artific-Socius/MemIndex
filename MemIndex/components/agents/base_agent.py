"""
BaseAgent - Agent 基础抽象类

定义了 Agent 的基本接口，所有 Agent 实现都需要继承此类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from components.chat.base_chat import BaseChat
    from components.memory.base_memory import BaseMemory


class BaseAgent(ABC):
    """
    Agent 基础抽象类

    组合 Chat 和 Memory 模块，提供完整的对话和记忆功能。
    """

    def __init__(self, name: str):
        """
        初始化 Agent

        Args:
            name: Agent 名称
        """
        self.name = name
        self.token_input = 0
        self.token_output = 0
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # 延迟统计（秒）
        self._memory_latencies: list[float] = []  # memory 查询延迟历史
        self._chat_latencies: list[float] = []    # chat 模型延迟历史
        self._last_memory_latency: float = 0.0    # 最后一次 memory 延迟
        self._last_chat_latency: float = 0.0      # 最后一次 chat 延迟

        # 额外元数据，用于报告
        self.extra_metadata: Dict[str, Dict[str, Any]] = {
            "Token Save Table": {
                "type": "table",
                "value": {
                    "data": [],
                    "columns": ["消息计数", "对话总Token数", "输入到LLM的Token数", "节省率 (%)"],
                },
                "description": "Token Save Table",
            },
        }

        # 对话历史
        self.chat_history: list[dict[str, str]] = []

    @property
    def avg_memory_latency(self) -> float:
        """平均 memory 延迟（秒）"""
        return sum(self._memory_latencies) / len(self._memory_latencies) if self._memory_latencies else 0.0

    @property
    def avg_chat_latency(self) -> float:
        """平均 chat 模型延迟（秒）"""
        return sum(self._chat_latencies) / len(self._chat_latencies) if self._chat_latencies else 0.0

    @property
    def last_memory_latency(self) -> float:
        """最后一次 memory 延迟（秒）"""
        return self._last_memory_latency

    @property
    def last_chat_latency(self) -> float:
        """最后一次 chat 模型延迟（秒）"""
        return self._last_chat_latency

    @property
    def has_memory_backend(self) -> bool:
        """是否有真实的 memory 后端（子类可覆盖）"""
        return False

    def record_memory_latency(self, latency: float) -> None:
        """记录 memory 查询延迟"""
        self._memory_latencies.append(latency)
        self._last_memory_latency = latency

    def record_chat_latency(self, latency: float) -> None:
        """记录 chat 模型延迟"""
        self._chat_latencies.append(latency)
        self._last_chat_latency = latency

    @abstractmethod
    async def send_message(self, message: str) -> str:
        """
        发送消息并获取回复

        这是 Agent 的主要接口方法，子类需要实现具体的逻辑。

        Args:
            message: 用户消息

        Returns:
            Agent 回复
        """
        raise NotImplementedError

    @staticmethod
    def format_token_string(tokens: int) -> str:
        """
        格式化 token 数量为可读字符串

        Args:
            tokens: token 数量

        Returns:
            格式化后的字符串
        """
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.2f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.2f}K"
        else:
            return str(tokens)

    def get_message_tokens(self, message: dict) -> int:
        """
        计算单条消息的 token 数量

        Args:
            message: 消息字典

        Returns:
            token 数量
        """
        template = "role:{role}\ncontent:{content}\n\n"
        return len(self.encoder.encode(
            template.format(role=message["role"], content=message["content"])
        ))

    def get_current_tokens(self) -> int:
        """
        计算当前对话历史的总 token 数量

        Returns:
            总 token 数量
        """
        template = "role:{role}\ncontent:{content}\n\n"
        total_tokens = 0
        for msg in self.chat_history:
            total_tokens += len(self.encoder.encode(
                template.format(role=msg["role"], content=msg["content"])
            ))
        return total_tokens

    def update_token_table(self, current_input_tokens: int) -> None:
        """
        更新 token 统计表格

        Args:
            current_input_tokens: 本次输入的 token 数量
        """
        current_tokens = self.get_current_tokens()
        conversation_count = len(self.chat_history) // 2
        token_offer = current_tokens - current_input_tokens
        token_offer_rate = token_offer / current_tokens if current_tokens > 0 else 0.0

        self.extra_metadata["Token Save Table"]["value"]["data"].append([
            str(conversation_count),
            self.format_token_string(current_tokens),
            self.format_token_string(current_input_tokens),
            f"{token_offer_rate:.2%}",
        ])



