"""
Mem0GraphMemory - 基于 Mem0 Graph 的 Memory 实现

使用 Mem0 Cloud API 并启用图谱功能作为记忆系统。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .mem0_memory import Mem0Memory

if TYPE_CHECKING:
    from utils.controller import LLMController


class Mem0GraphMemory(Mem0Memory):
    """
    基于 Mem0 Graph 的 Memory 实现

    继承自 Mem0Memory，默认启用图谱功能。
    """

    def __init__(
        self,
        llm_controller: "LLMController" = None,
        api_key: str = None,
    ):
        """
        初始化 Mem0 Graph Memory

        Args:
            llm_controller: LLM 控制器实例
            api_key: Mem0 API Key，如果为None则从环境变量获取
        """
        super().__init__(
            llm_controller=llm_controller,
            api_key=api_key,
            enable_graph=True,  # 默认启用图谱
        )



