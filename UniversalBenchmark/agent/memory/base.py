from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class MemoryMixin(ABC):
    """对话记忆管理的基础 Mixin。

    定义了以 OpenAI API 消息格式存储和检索对话上下文的接口。
    具体子类决定消息的存储、检索和组织方式（buffer、window、summary、RAG 等）。

    使用协作式 ``__init__`` —— 始终接收并转发 ``**kwargs``，
    以确保与 :class:`LLMMixin` / :class:`Agent` 混合时 MRO 链正确工作。
    """

    _memory_type: str = "MemoryMixin"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "get_messages" in cls.__dict__:
            cls._memory_type = cls.__name__

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        memory_tag: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._system_prompt = system_prompt
        self._memory_tag = memory_tag

    # ------------------------------------------------------------------
    # 抽象接口 - 每个具体 Memory 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        """记录 *user_input* 并返回完整的消息上下文。

        子类实现时应当：

        1. 将用户消息追加到内部历史记录。
        2. 组装完整的消息列表（system prompt + 相关历史 + 当前用户消息）。
        3. 应用 Memory 特有的逻辑（窗口截断、检索增强等）。

        Returns
        -------
        list[dict[str, Any]]
            OpenAI chat-completion 格式的消息列表，例如
            ``[{"role": "system", "content": "…"}, …]``
        """
        ...

    @abstractmethod
    def add_response(self, content: str) -> None:
        """将助手回复存入对话历史。"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """清空所有对话历史，保留配置。"""
        ...

    # ------------------------------------------------------------------
    # 通用工具方法 - 所有子类可直接使用
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]) -> None:
        self._system_prompt = value

    def _build_system_messages(self) -> list[dict[str, Any]]:
        """如果设置了 system prompt 则返回 ``[{"role": "system", …}]``，
        否则返回空列表。"""
        if self._system_prompt:
            return [self._make_message("system", self._system_prompt)]
        return []

    @staticmethod
    def _make_message(role: str, content: str) -> dict[str, Any]:
        """创建单条 OpenAI 格式的消息字典。"""
        return {"role": role, "content": content}

    # ------------------------------------------------------------------
    # 标识符
    # ------------------------------------------------------------------

    @property
    def memory_identifier(self) -> str:
        """当前 Memory 配置的可读标识符。

        即使通过动态组合的 Agent 子类访问，也会自动解析到
        实现了 ``get_messages`` 的具体 Memory 类名。
        """
        name = self._memory_type
        if self._memory_tag:
            return f"{name}:{self._memory_tag}"
        return name
