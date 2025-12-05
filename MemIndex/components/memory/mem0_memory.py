"""
Mem0Memory - 基于 Mem0 的 Memory 实现

使用 Mem0 Cloud API 作为记忆系统。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, List, Dict, TYPE_CHECKING

from loguru import logger

try:
    from mem0 import MemoryClient
except ImportError:
    # Handle the case where mem0 is not installed, though user said they installed it.
    # This prevents import errors from crashing the whole app if mem0 is optional.
    MemoryClient = None

from .base_memory import BaseMemory

if TYPE_CHECKING:
    from utils.controller import LLMController


class Mem0Memory(BaseMemory):
    """
    基于 Mem0 的 Memory 实现

    使用 Mem0 Cloud API 进行记忆的存储和检索。
    """

    def __init__(
        self,
        llm_controller: "LLMController" = None,
        api_key: str = None,
        enable_graph: bool = False,
    ):
        """
        初始化 Mem0 Memory

        Args:
            llm_controller: LLM 控制器实例
            api_key: Mem0 API Key，如果为None则从环境变量获取
            enable_graph: 是否启用图谱功能
        """
        super().__init__(llm_controller)

        if MemoryClient is None:
            raise ImportError("mem0 package is not installed. Please install it with `pip install mem0ai`.")

        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MEM0_API_KEY is required. "
                "Set it via environment variable or pass api_key parameter."
            )

        self.client = MemoryClient(api_key=self.api_key)
        self.enable_graph = enable_graph

    async def initialize(self, user_id: str = None) -> str:
        """
        初始化记忆库

        Args:
            user_id: 用户ID，如果为None则自动生成

        Returns:
            用户ID
        """
        if user_id:
            self.user_id = user_id
        else:
            self.user_id = f"mem0_user_{uuid.uuid4().hex[:12]}"
        return self.user_id

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        搜索相关记忆

        Args:
            query: 搜索查询
            **kwargs: 其他搜索参数
                - retry: 重试次数，默认 3
                - enable_graph: 是否启用图搜索

        Returns:
            记忆列表
        """
        max_retries = kwargs.get("retry", 3)
        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                results = self.client.search(
                    query,
                    filters={"user_id": self.user_id},
                    enable_graph=kwargs.get("enable_graph", self.enable_graph)
                )
                if had_error:
                    logger.success(f"✅ Memory search recovered after {attempt} retry(ies)")
                return results.get("results", []) if isinstance(results, dict) else results
            except Exception as e:
                had_error = True
                last_error = e
                logger.warning(f"Memory search failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # 递增等待时间

        logger.error(f"Memory search failed after {max_retries} attempts: {last_error}")
        return []

    async def add(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """
        添加记忆

        Args:
            messages: 消息列表
            **kwargs: 其他参数
                - retry: 重试次数，默认 3
                - enable_graph: 是否启用图搜索

        Returns:
            是否添加成功
        """
        max_retries = kwargs.get("retry", 3)
        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                self.client.add(
                    messages,
                    user_id=self.user_id,
                    enable_graph=kwargs.get("enable_graph", self.enable_graph)
                )
                if had_error:
                    logger.success(f"✅ Memory add recovered after {attempt} retry(ies)")
                return True
            except Exception as e:
                had_error = True
                last_error = e
                logger.warning(f"Memory add failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # 递增等待时间

        logger.error(f"Memory add failed after {max_retries} attempts: {last_error}")
        return False

    async def delete(self, memory_id: str, **kwargs) -> bool:
        """
        删除指定记忆

        Args:
            memory_id: 记忆ID
            **kwargs: 其他参数
                - retry: 重试次数，默认 3

        Returns:
            是否删除成功
        """
        max_retries = kwargs.get("retry", 3)
        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                self.client.delete(memory_id=memory_id)
                if had_error:
                    logger.success(f"✅ Memory delete recovered after {attempt} retry(ies)")
                logger.debug(f"Memory deleted: {memory_id}")
                return True
            except Exception as e:
                had_error = True
                last_error = e
                logger.warning(f"Memory delete failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        logger.error(f"Memory delete failed after {max_retries} attempts: {last_error}")
        return False


