from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from agent.progress import get_progress


@dataclass
class TurnTrace:
    """一轮对话的 message id 与 request id 追踪记录。

    Memory 实现者可以通过覆写 :meth:`MemoryMixin.get_last_turn_trace`
    提供来自后端的真实 message id。未覆写时框架自动生成 UUID 兜底。

    一轮对话通常包含两次请求：查询（``get_messages``）和追加
    （``add_response``），每次请求可以有独立的 request id。

    Attributes
    ----------
    user_message_id:
        用户消息的唯一标识。
    assistant_message_id:
        助手消息的唯一标识（在 ``add_response`` 后填充）。
    id_source:
        ``"provider"`` 表示 Memory 后端提供的真实 id，
        ``"framework"`` 表示框架自动生成的 UUID。
    query_request_id:
        查询请求（``get_messages``）的 request id。
    append_request_id:
        追加请求（``add_response``）的 request id。
    extra:
        其他追踪信息。
    """

    user_message_id: str = ""
    assistant_message_id: str = ""
    id_source: str = "framework"
    query_request_id: str = ""
    append_request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


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
        self._current_turn_trace: Optional[TurnTrace] = None
        self._framework_memory_library_id = f"framework-{uuid.uuid4()}"

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
    # Message ID 追踪 - Memory 实现者可覆写以提供真实 ID
    # ------------------------------------------------------------------

    def get_last_turn_trace(self) -> TurnTrace:
        """返回最近一轮对话的 message id 追踪记录。

        **默认行为**：返回框架自动生成的 UUID 兜底 trace。

        **Memory 实现者**可覆写此方法以返回来自后端的真实 id。
        例如 Memecho 可从服务端的 query/append 响应中提取真实
        message id 并在此返回。

        框架会在每次 ``get_messages`` 调用前自动初始化一个
        包含 UUID 的 trace（:attr:`_current_turn_trace`）。
        如果 Memory 后端提供了真实 id，实现者应在
        ``get_messages`` / ``add_response`` 内部更新该 trace，
        或直接覆写本方法。

        Returns
        -------
        TurnTrace
            当前轮次的追踪记录。
        """
        if self._current_turn_trace is not None:
            return self._current_turn_trace
        return TurnTrace(
            user_message_id=str(uuid.uuid4()),
            assistant_message_id=str(uuid.uuid4()),
            id_source="framework",
        )

    def _init_turn_trace(self) -> TurnTrace:
        """框架内部方法：在每轮 get_messages 前初始化 UUID 兜底 trace。

        Memory 子类在 ``get_messages`` 内部可通过修改
        ``self._current_turn_trace`` 来覆盖 id 来源。
        """
        self._current_turn_trace = TurnTrace(
            user_message_id=str(uuid.uuid4()),
            assistant_message_id=str(uuid.uuid4()),
            id_source="framework",
        )
        return self._current_turn_trace

    def _finalize_turn_trace(self, assistant_message_id: str = "") -> None:
        """框架内部方法：在 add_response 后更新 assistant id。

        如果 Memory 子类已在 add_response 中设置了真实的
        assistant_message_id，则本方法不会覆盖。
        """
        if self._current_turn_trace is None:
            return
        if assistant_message_id and not self._current_turn_trace.assistant_message_id:
            self._current_turn_trace.assistant_message_id = assistant_message_id

    # ------------------------------------------------------------------
    # 记忆库 ID 追踪 - 默认实现 + 可覆写
    # ------------------------------------------------------------------

    def ensure_memory_library(self) -> str:
        """确保存在可用记忆库并返回其 ID。

        默认实现返回框架级的稳定 fallback id。长期记忆后端
        （如 Memecho/Mem0）应覆写此方法并返回后端真实 id。
        """
        return self._framework_memory_library_id

    def get_memory_library_id(self) -> str:
        """获取当前使用的记忆库 ID。默认调用 ensure_memory_library。"""
        return self.ensure_memory_library()

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
    # 批量导入
    # ------------------------------------------------------------------

    def bulk_import(
        self,
        conversations: list[tuple[str, str]],
    ) -> int:
        """批量导入历史对话到记忆系统。

        用于 benchmark 等需要先预填充大量历史数据再测试的场景。
        默认实现逐条调用 ``get_messages`` + ``add_response``，
        子类可覆写以使用更高效的批量接口。

        Parameters
        ----------
        conversations:
            对话轮次列表，每个元素为 ``(user_input, assistant_response)`` 元组。

        Returns
        -------
        int
            成功导入的轮次数。
        """
        total = len(conversations)
        ident = self.memory_identifier
        logger.info(
            f"[{ident}] 开始批量导入 {total} 条对话"
            f"（逐条模式，当前记忆方案不支持原生批量导入）"
        )
        pg = get_progress()
        bh = pg.add_task(
            f"[{ident}] bulk_import",
            total=float(total) if total > 0 else 1.0,
            task_key="memory:bulk_import",
        )
        imported = 0
        report_interval = max(1, total // 10)
        try:
            for user_input, assistant_response in conversations:
                self.get_messages(user_input)
                self.add_response(assistant_response)
                imported += 1
                pg.advance(bh, 1)
                if imported % report_interval == 0 or imported == total:
                    logger.info(
                        f"[{ident}] 批量导入进度: "
                        f"{imported}/{total} ({imported * 100 // max(total, 1)}%)"
                    )
            if total == 0:
                pg.advance(bh, 1)
        finally:
            pg.remove_task(bh)
        logger.info(f"[{ident}] 批量导入完成: 共导入 {imported} 条对话")
        return imported

    # ------------------------------------------------------------------
    # 语料导入
    # ------------------------------------------------------------------

    def import_corpus(
        self,
        documents: list[str],
        corpus_id: str = "",
    ) -> str:
        """将独立的语料文档导入记忆系统，返回库标识符。

        与 ``bulk_import`` 不同，此方法接收的是原始文档列表而非
        对话对。长期记忆后端（如 Memecho）可覆写此方法以实现
        文件级导入（分块、索引等）。

        默认实现：合并全部文档为单条对话后调用 ``bulk_import``。

        Parameters
        ----------
        documents:
            原始文档字符串列表。
        corpus_id:
            可选的语料标识符（用于注册表缓存等）。

        Returns
        -------
        str
            库标识符。长期记忆后端返回实际的 library ID；
            本地内存后端返回 ``corpus_id`` 或 ``"local"``。
        """
        merged = "\n\n".join(documents)
        self.bulk_import([(
            "Please read and remember the following information "
            "carefully. I will ask you questions about it later."
            "\n\n" + merged,
            "I have carefully read and memorized the information "
            "you provided. Feel free to ask me any questions about it.",
        )])
        return corpus_id or "local"

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
