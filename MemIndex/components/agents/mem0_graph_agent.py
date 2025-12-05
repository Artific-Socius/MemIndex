"""
Mem0GraphAgent - 基于 Mem0 Graph 的 Agent 实现

使用 Mem0 Graph 作为记忆系统的 Agent 实现。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .mem0_agent import Mem0Agent
from .base_agent import BaseAgent
from components.memory.mem0_graph_memory import Mem0GraphMemory

if TYPE_CHECKING:
    from utils.controller import LLMController
    from prompts import PromptManager


class Mem0GraphAgent(Mem0Agent):
    """
    基于 Mem0 Graph 的 Agent 实现

    继承自 Mem0Agent，使用启用图谱功能的 Mem0 作为记忆系统。
    """

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
        api_key: str = None,
        prompt_manager: Optional["PromptManager"] = None,
        chat_prompt_key: str = None,
    ):
        """
        初始化 Mem0 Graph Agent

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
            api_key: Mem0 API Key
            prompt_manager: 提示词管理器（可选，不传则使用全局单例）
            chat_prompt_key: Chat 提示词的 key（可选，不传则使用默认）
        """
        # 调用父类初始化，但不使用父类的 memory
        BaseAgent.__init__(self, f"Mem0 Graph Agent - {model.replace('/', '-')}")

        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.llm_controller = llm_controller

        # 提示词管理
        if prompt_manager is None:
            from prompts import get_prompt_manager
            prompt_manager = get_prompt_manager()
        self.prompt_manager = prompt_manager
        self.chat_prompt_key = chat_prompt_key

        # 初始化 Chat 模块
        from components.chat.llm_chat import LLMChat
        self.chat = LLMChat(
            llm_controller=llm_controller,
            model=model,
            context_window=context_window,
            temperature=temperature,
        )

        # 初始化 Memory 模块（使用 Graph 版本）
        self.memory = Mem0GraphMemory(
            llm_controller=llm_controller,
            api_key=api_key,
        )

        # Token 编码器
        import tiktoken
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # 更新额外元数据
        self.extra_metadata["Memory Provider"] = {
            "type": "text",
            "value": "Mem0 Graph",
            "description": "Memory Provider"
        }
        self.extra_metadata["user_id"] = {
            "type": "text",
            "value": "N/A",
            "description": "Mem0 User ID"
        }
