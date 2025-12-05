"""
LLMChat - 基于 LLM 的 Chat 实现

直接使用 LLM 进行对话，不使用外部记忆系统。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_chat import BaseChat

if TYPE_CHECKING:
    from utils.controller import LLMController


class LLMChat(BaseChat):
    """
    基于 LLM 的 Chat 实现

    直接使用 LLM 进行对话，通过上下文窗口管理对话历史。
    """

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
    ):
        """
        初始化 LLM Chat

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
        """
        super().__init__(llm_controller, model, context_window, temperature)

    async def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        生成 LLM 响应

        Args:
            messages: 消息列表

        Returns:
            LLM 响应文本
        """
        # 根据上下文窗口截断消息
        truncated_messages = self.auto_truncate_messages(messages)

        # 调用 LLM 生成响应
        response = await self.llm_controller.completion_with_retry_async_t(
            self.model,
            truncated_messages,
            temperature=self.temperature
        )

        if response and response.completion:
            # 更新 token 统计
            self.token_input += response.token_information.input_tokens
            self.token_output += response.token_information.output_tokens
            return response.completion

        raise Exception("LLM completion failed")

    async def chat(self, message: str, with_time: bool = True) -> str:
        """
        发送消息并获取回复

        Args:
            message: 用户消息
            with_time: 是否在消息中添加时间戳

        Returns:
            LLM 响应文本
        """
        # 准备消息
        formatted_message = self.format_time(message) if with_time else message

        # 构建消息列表
        messages = self.chat_history.copy()
        messages.append({"role": "user", "content": formatted_message})

        # 生成响应
        response = await self.generate_response(messages)

        # 更新对话历史
        self.add_to_history("user", formatted_message)
        self.add_to_history("assistant", response)

        return response



