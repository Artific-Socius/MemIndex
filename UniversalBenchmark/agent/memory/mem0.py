from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from loguru import logger

from .base import MemoryMixin

try:
    from mem0 import MemoryClient  # type: ignore[import-untyped]
except ImportError:
    MemoryClient = None

_DEFAULT_MEMORY_CONTEXT_PREFIX = "以下是与当前对话相关的历史记忆：\n"


class Mem0Memory(MemoryMixin):
    """基于 Mem0 云 API 的记忆实现。

    Mem0 从对话中抽取并存储 *事实性记忆*。每轮对话中，
    :meth:`get_messages` 搜索 Mem0 获取相关记忆，
    然后组装完整的消息列表：

        system prompt → 记忆上下文 → 最近本地历史 → 当前用户消息

    LLM 生成回复后，:meth:`add_response` 将完整的一轮对话
    （用户 + 助手）提交给 Mem0 进行记忆抽取，**同时** 追加到
    本地历史缓冲区中用于"最近轮次"上下文。

    Parameters
    ----------
    api_key:
        Mem0 API 密钥。若未提供则从 ``MEM0_API_KEY`` 环境变量读取。
    user_id:
        已有的 Mem0 用户 ID。若为 *None* 则自动生成。
    recent_turns:
        在检索到的记忆之外，额外包含多少轮最近的对话历史作为上下文。
    enable_graph:
        是否启用 Mem0 基于图谱的记忆搜索/添加。
    memory_context_prefix:
        将召回的记忆条目拼接后作为 system 消息注入前的前缀文本。
    max_retries:
        Mem0 客户端调用的最大重试次数。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        recent_turns: int = 3,
        enable_graph: bool = False,
        memory_context_prefix: str = _DEFAULT_MEMORY_CONTEXT_PREFIX,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if MemoryClient is None:
            raise ImportError(
                "需要 mem0 包。请通过以下命令安装: pip install mem0ai"
            )

        self._api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self._api_key:
            raise ValueError(
                "需要 MEM0_API_KEY。请通过环境变量设置或传入 api_key 参数。"
            )

        self._client: Any = MemoryClient(api_key=self._api_key)
        self._user_id = user_id
        self._recent_turns = recent_turns
        self._enable_graph = enable_graph
        self._memory_context_prefix = memory_context_prefix
        self._max_retries = max_retries

        self._history: list[dict[str, Any]] = []
        self._pending_user_input: Optional[str] = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self, user_id: Optional[str] = None) -> str:
        """创建（或指定）一个 Mem0 用户身份，返回用户 ID。"""
        self._user_id = user_id or f"mem0_user_{uuid.uuid4().hex[:12]}"
        return self._user_id

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    # ------------------------------------------------------------------
    # MemoryMixin 接口实现
    # ------------------------------------------------------------------

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._pending_user_input = user_input

        memories = self._search(user_input)

        messages: list[dict[str, Any]] = self._build_system_messages()

        # 将召回的记忆作为 system 消息注入
        if memories:
            memory_lines = [
                f"- {m.get('memory', '')}"
                for m in memories
                if m.get("memory")
            ]
            if memory_lines:
                ctx = self._memory_context_prefix + "\n".join(memory_lines)
                messages.append(self._make_message("system", ctx))

        messages.extend(self._get_recent_history())
        messages.append(self._make_message("user", user_input))
        return messages

    def add_response(self, content: str) -> None:
        # 将完整的一轮对话（用户 + 助手）追加到本地历史
        if self._pending_user_input is not None:
            self._history.append(
                self._make_message("user", self._pending_user_input)
            )

        self._history.append(self._make_message("assistant", content))

        # 提交给 Mem0 进行记忆抽取
        if self._pending_user_input is not None:
            self._add_to_mem0(self._pending_user_input, content)

        self._pending_user_input = None

    def bulk_import(
        self,
        conversations: list[tuple[str, str]],
    ) -> int:
        """批量提交给 Mem0 进行记忆抽取，同时追加到本地历史。

        跳过 get_messages 中的搜索步骤，直接调用 _add_to_mem0。
        """
        self._ensure_initialized()
        imported = 0
        for user_input, assistant_response in conversations:
            self._history.append(self._make_message("user", user_input))
            self._history.append(
                self._make_message("assistant", assistant_response),
            )
            if self._add_to_mem0(user_input, assistant_response):
                imported += 1
        return imported

    def reset(self) -> None:
        self._history.clear()
        self._pending_user_input = None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        """本地对话历史的只读副本。"""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """已完成的对话轮数。"""
        return sum(1 for m in self._history if m["role"] == "assistant")

    def _ensure_initialized(self) -> None:
        if not self._user_id:
            self.initialize()

    def _get_recent_history(self) -> list[dict[str, Any]]:
        """返回本地历史中最近 *recent_turns* 轮的对话记录。"""
        count = self._recent_turns * 2
        if len(self._history) > count:
            return list(self._history[-count:])
        return list(self._history)

    # ------------------------------------------------------------------
    # Mem0 客户端封装（带重试）
    # ------------------------------------------------------------------

    def _search(self, query: str) -> list[dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                results: Any = self._client.search(
                    query,
                    filters={"user_id": self._user_id},
                    enable_graph=self._enable_graph,
                )
                if isinstance(results, dict):
                    return list(results.get("results", []))
                return list(results)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Mem0 搜索失败 "
                    f"(第 {attempt + 1}/{self._max_retries} 次): {exc}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(1 * (attempt + 1))

        logger.error(
            f"Mem0 搜索在 {self._max_retries} 次重试后仍然失败: {last_error}"
        )
        return []

    def _add_to_mem0(self, user_input: str, assistant_content: str) -> bool:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_content},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._client.add(
                    messages,
                    user_id=self._user_id,
                    enable_graph=self._enable_graph,
                )
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Mem0 添加失败 "
                    f"(第 {attempt + 1}/{self._max_retries} 次): {exc}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(1 * (attempt + 1))

        logger.error(
            f"Mem0 添加在 {self._max_retries} 次重试后仍然失败: {last_error}"
        )
        return False


class Mem0GraphMemory(Mem0Memory):
    """默认启用图谱搜索的 Mem0 记忆实现。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_graph", True)
        super().__init__(**kwargs)
