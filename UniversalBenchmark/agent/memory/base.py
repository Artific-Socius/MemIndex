from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger


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
        imported = 0
        report_interval = max(1, total // 10)
        for user_input, assistant_response in conversations:
            self.get_messages(user_input)
            self.add_response(assistant_response)
            imported += 1
            if imported % report_interval == 0 or imported == total:
                logger.info(
                    f"[{ident}] 批量导入进度: "
                    f"{imported}/{total} ({imported * 100 // total}%)"
                )
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
