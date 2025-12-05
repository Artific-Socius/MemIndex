"""
BaseMemory - Memory 模块基础抽象类

定义了 Memory 模块的基本接口，所有 Memory 实现都需要继承此类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.controller import LLMController


class BaseMemory(ABC):
    """
    Memory 模块基础抽象类

    负责记忆的存储、检索和管理。
    """

    def __init__(self, llm_controller: "LLMController" = None):
        """
        初始化 Memory 模块

        Args:
            llm_controller: LLM 控制器实例（某些记忆系统可能需要）
        """
        self.llm_controller = llm_controller
        self.user_id: str | None = None

    @abstractmethod
    async def initialize(self, user_id: str = None) -> str:
        """
        初始化记忆库

        Args:
            user_id: 用户ID，如果为None则自动生成

        Returns:
            用户ID
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        搜索相关记忆

        Args:
            query: 搜索查询
            **kwargs: 其他搜索参数

        Returns:
            记忆列表
        """
        raise NotImplementedError

    @abstractmethod
    async def add(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """
        添加记忆

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            是否添加成功
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        删除指定记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        raise NotImplementedError

    def build_context_from_memories(
        self,
        memories: List[Dict[str, Any]],
        recent_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        根据召回的记忆构建上下文消息

        Args:
            memories: 召回的记忆列表
            recent_history: 最近的对话历史

        Returns:
            构建的上下文消息列表
        """
        messages = []

        # 如果有召回的记忆，将其作为系统消息
        if memories:
            memory_texts = []
            for mem in memories:
                memory_content = mem.get("memory", "")
                if memory_content:
                    memory_texts.append(f"- {memory_content}")

            if memory_texts:
                memory_context = "以下是与当前对话相关的历史记忆：\n" + "\n".join(memory_texts)
                messages.append({"role": "system", "content": memory_context})

        # 添加最近的对话历史
        if recent_history:
            messages.extend(recent_history)

        return messages

    def get_recent_history(
        self,
        chat_history: List[Dict[str, str]],
        max_turns: int = 3
    ) -> List[Dict[str, str]]:
        """
        获取最近的对话历史

        Args:
            chat_history: 完整对话历史
            max_turns: 最大轮数

        Returns:
            最近的对话历史
        """
        recent_count = max_turns * 2  # 每轮包含 user 和 assistant
        if len(chat_history) > recent_count:
            return chat_history[-recent_count:]
        return chat_history.copy()



