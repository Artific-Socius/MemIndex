"""
LLMAgent - 基于纯 LLM 的 Agent 实现

直接使用 LLM 进行对话，不使用外部记忆系统。
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import tiktoken

from .base_agent import BaseAgent
from components.chat.llm_chat import LLMChat

if TYPE_CHECKING:
    from utils.controller import LLMController


class LLMAgent(BaseAgent):
    """
    基于纯 LLM 的 Agent 实现

    直接使用 LLM 进行对话，通过上下文窗口管理对话历史。
    不使用外部记忆系统。
    """

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
    ):
        """
        初始化 LLM Agent

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
        """
        super().__init__(f"LLM Agent - {model.replace('/', '-')}")

        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.llm_controller = llm_controller

        # 初始化 Chat 模块
        self.chat = LLMChat(
            llm_controller=llm_controller,
            model=model,
            context_window=context_window,
            temperature=temperature,
        )

        # Token 编码器
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # 本地历史（与 chat 模块共享引用）
        self.history = []

    async def auto_messages(self, messages: list[dict]) -> list[dict]:
        """
        根据上下文窗口自动截断消息

        Args:
            messages: 消息列表

        Returns:
            截断后的消息列表
        """
        return self.chat.auto_truncate_messages(messages)

    @staticmethod
    def format_time(message: str) -> str:
        """格式化消息，添加当前时间"""
        current = datetime.datetime.now()
        current_time = current.strftime("%Y-%m-%d %H:%M:%S")
        return f"[当前时间为: {current_time}]\n{message}"

    async def send_message(self, message: str) -> str:
        """
        发送消息并获取回复

        Args:
            message: 用户消息

        Returns:
            LLM 回复
        """
        # 准备消息
        messages = self.history.copy()
        messages.append({"role": "user", "content": self.format_time(message)})

        # 截断消息
        messages = await self.auto_messages(messages)

        # 调用 LLM
        response = await self.llm_controller.completion_with_retry_async_t(
            self.model,
            messages,
            temperature=self.temperature
        )

        # 检查 LLM 调用是否成功
        if response is None:
            raise RuntimeError("LLM call failed: no response received after all retries")

        # 记录 chat 延迟
        self.record_chat_latency(response.llm_time_elapsed)

        # 更新统计
        self.token_input += response.token_information.input_tokens
        self.token_output += response.token_information.output_tokens

        # 更新历史
        messages.append({"role": "assistant", "content": response.completion})
        self.history = messages

        # 同步到 chat_history
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response.completion})

        return response.completion



