from __future__ import annotations

from typing import Any, Optional

from .base import MemoryMixin


class BufferMemory(MemoryMixin):
    """全量历史缓冲记忆。

    将所有消息保存在一个列表中，返回全部（或最近 *max_turns* 轮）
    作为 LLM 的上下文。这是最简单的记忆策略，无需外部服务。

    Parameters
    ----------
    max_turns:
        若设置，则只在返回的上下文中包含最近 *max_turns* 轮对话
        （每轮 = 一条用户消息 + 一条助手消息）。``None`` 表示不限。
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._max_turns = max_turns
        self._history: list[dict[str, Any]] = []

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        self._history.append(self._make_message("user", user_input))

        messages = self._build_system_messages()
        history = self._history
        if self._max_turns is not None:
            history = history[-(self._max_turns * 2) :]
        messages.extend(history)
        return messages

    def add_response(self, content: str) -> None:
        self._history.append(self._make_message("assistant", content))

    def reset(self) -> None:
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """已完成的对话轮数。"""
        return sum(1 for m in self._history if m["role"] == "assistant")

    @property
    def history(self) -> list[dict[str, Any]]:
        """原始对话历史的只读副本。"""
        return list(self._history)
