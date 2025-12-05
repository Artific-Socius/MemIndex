"""
MemechoAgent - 基于 Memecho 的 Agent 实现

使用 Memecho 作为记忆系统的 Agent 实现。
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING

import tiktoken
from loguru import logger

from .base_agent import BaseAgent
from components.chat.llm_chat import LLMChat
from components.memory.memecho_memory import MemechoMemory

if TYPE_CHECKING:
    from utils.controller import LLMController


class MemechoAgent(BaseAgent):
    """
    基于 Memecho 的 Agent 实现

    使用 Memecho API 作为记忆系统。
    """

    @property
    def has_memory_backend(self) -> bool:
        """Memecho Agent 有真实的 memory 后端"""
        return True

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
        memory_api_base_url: str = "https://api.memecho.cloud",
    ):
        """
        初始化 Memecho Agent

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
            memory_api_base_url: Memecho API 基础 URL
        """
        super().__init__(f"Memecho Agent - {model.replace('/', '-')}")

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

        # 初始化 Memory 模块
        self.memory = MemechoMemory(
            llm_controller=llm_controller,
            api_base_url=memory_api_base_url,
        )

        # Token 编码器
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # 更新额外元数据
        self.extra_metadata["Api Base"] = {
            "type": "text",
            "value": memory_api_base_url,
            "description": "Memecho API Base URL"
        }
        self.extra_metadata["user_id"] = {
            "type": "text",
            "value": "N/A",
            "description": "Memecho User ID"
        }

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
            Agent 回复
        """
        # 初始化记忆库
        if not self.memory.user_id:
            await self.memory.initialize()
            self.extra_metadata["user_id"]["value"] = self.memory.user_id

        # 调用记忆查询 API 获取长期记忆消息栈
        memory_time_start = time.time()
        memory_messages = await self.memory.search(message)
        memory_time_end = time.time()
        memory_latency = memory_time_end - memory_time_start
        self.record_memory_latency(memory_latency)
        logger.debug(f"Memory query took {memory_latency:.2f} seconds and returned {len(memory_messages)} messages")

        # 截断消息
        messages = self.chat.auto_truncate_messages(memory_messages)

        # 生成回复
        response = None
        for _ in range(3):
            try:
                response = await self.llm_controller.completion_with_retry_async_t(
                    self.model, messages, temperature=self.temperature
                )
                if response and response.completion:
                    # 记录 chat 延迟
                    self.record_chat_latency(response.llm_time_elapsed)
                    break
            except Exception as e:
                logger.error(f"LLM completion failed: {e}")

        if not response or not response.completion:
            raise Exception("LLM completion failed after retries")

        # 更新统计
        self.token_input += response.token_information.input_tokens
        self.token_output += response.token_information.output_tokens

        # 追加助手回复到记忆
        await self.memory.add([
            {"role": "assistant", "content": response.completion}
        ])

        # 更新对话历史
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response.completion})

        # 更新 token 表格
        current_input_tokens = sum(self.get_message_tokens(msg) for msg in messages)
        self.update_token_table(current_input_tokens)

        return response.completion



