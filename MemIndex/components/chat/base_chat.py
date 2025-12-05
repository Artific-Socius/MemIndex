"""
BaseChat - Chat 模块基础抽象类

定义了 Chat 模块的基本接口，所有 Chat 实现都需要继承此类。
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from utils.controller import LLMController


class BaseChat(ABC):
    """
    Chat 模块基础抽象类

    负责与 LLM 进行对话交互，管理对话历史和上下文窗口。
    """

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
    ):
        """
        初始化 Chat 模块

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
        """
        self.llm_controller = llm_controller
        self.model = model
        self.context_window = context_window
        self.temperature = temperature

        # Token 编码器
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # Token 统计
        self.token_input = 0
        self.token_output = 0

        # 对话历史
        self.chat_history: list[dict[str, str]] = []

    @abstractmethod
    async def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        生成 LLM 响应

        Args:
            messages: 消息列表

        Returns:
            LLM 响应文本
        """
        raise NotImplementedError

    def add_to_history(self, role: str, content: str) -> None:
        """
        添加消息到对话历史

        Args:
            role: 消息角色 (user/assistant)
            content: 消息内容
        """
        self.chat_history.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        """
        获取对话历史

        Returns:
            对话历史列表
        """
        return self.chat_history.copy()

    def clear_history(self) -> None:
        """清空对话历史"""
        self.chat_history.clear()

    def auto_truncate_messages(self, messages: list[dict]) -> list[dict]:
        """
        根据上下文窗口自动截断消息

        Args:
            messages: 消息列表

        Returns:
            截断后的消息列表
        """
        results = []
        tokens = 0
        for message in messages[::-1]:
            content = ""
            if isinstance(message["content"], str):
                content = message["content"]
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "text":
                        content += item.get("text", "")

            token = len(self.encoder.encode(content))
            if tokens + token > self.context_window:
                break
            tokens += token
            results.append(message)
        return results[::-1]

    def get_message_tokens(self, message: dict) -> int:
        """
        计算单条消息的 token 数量

        Args:
            message: 消息字典

        Returns:
            token 数量
        """
        template = "role:{role}\ncontent:{content}\n\n"
        return len(self.encoder.encode(template.format(
            role=message["role"],
            content=message["content"]
        )))

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

    @staticmethod
    def format_time(message: str) -> str:
        """
        格式化消息，添加当前时间

        Args:
            message: 原始消息

        Returns:
            添加时间后的消息
        """
        current = datetime.datetime.now()
        current_time = current.strftime("%Y-%m-%d %H:%M:%S")
        return f"[当前时间为: {current_time}]\n{message}"

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



