"""
MemechoMemory - 基于 Memecho 的 Memory 实现

使用 Memecho API 作为记忆系统。
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from typing import Any, List, Dict, Optional, Literal, TYPE_CHECKING

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from .base_memory import BaseMemory

if TYPE_CHECKING:
    from utils.controller import LLMController


class Content(BaseModel):
    """消息内容模型"""
    type: Literal["text", "file"]
    text: Optional[str] = None
    file_id: Optional[str] = None


class InputMessage(BaseModel):
    """输入消息模型"""
    id: str
    role: Optional[Literal["user", "assistant", "tool", "function"]] = None
    content: list[Content] = Field(default_factory=list)


class QueryResult(BaseModel):
    """查询结果模型"""
    ready_messages: list[InputMessage]
    message_id: str | None
    recall_message_id_list: list[str]


class AppendResult(BaseModel):
    """追加结果模型"""
    result: Literal["ok", "fail"]
    message_id: str


class MemechoMemory(BaseMemory):
    """
    基于 Memecho 的 Memory 实现

    使用 Memecho API 进行记忆的存储和检索。
    """

    def __init__(
        self,
        llm_controller: "LLMController" = None,
        api_base_url: str = "https://api.memecho.cloud",
        api_key: str = None,
    ):
        """
        初始化 Memecho Memory

        Args:
            llm_controller: LLM 控制器实例
            api_base_url: Memecho API 基础 URL
            api_key: Memecho API Key，如果为None则从环境变量获取
        """
        super().__init__(llm_controller)

        self.api_base_url = api_base_url
        self.api_key = api_key or os.getenv("MEMECHO_API_KEY")

    async def initialize(self, user_id: str = None, max_retries: int = 3) -> str:
        """
        初始化记忆库，创建新的记忆库

        Args:
            user_id: 用户ID，如果为None则创建新的
            max_retries: 最大重试次数

        Returns:
            用户ID（记忆库ID）
        """
        if user_id:
            self.user_id = user_id
            return self.user_id

        # 创建新的记忆库
        alias = f"memecho_user_{uuid.uuid4().hex[:12]}"
        data = {"alias": alias}
        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
                    }

                    async with session.post(
                        f"{self.api_base_url}/api/v1/memory/create",
                        json=data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status != 200:
                            had_error = True
                            error_text = await response.text()
                            logger.warning(
                                f"Create memory library failed (attempt {attempt + 1}/{max_retries}): "
                                f"status={response.status}, response={error_text}"
                            )
                            last_error = Exception(f"HTTP {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1 * (attempt + 1))
                            continue

                        if had_error:
                            logger.success(f"✅ Create memory library recovered after {attempt} retry(ies)")
                        result = await response.json()
                        self.user_id = result["id"]
                        return self.user_id

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                had_error = True
                last_error = e
                logger.warning(
                    f"Create memory library network error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        raise Exception(f"Create memory library failed after {max_retries} attempts: {last_error}")

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        搜索相关记忆（通过查询API）

        Args:
            query: 搜索查询
            **kwargs: 其他搜索参数
                - retry: 重试次数，默认 3
                - read_only: 是否只读模式
                - include_user_query: 是否包含用户查询

        Returns:
            记忆列表（转换为标准消息格式）
        """
        input_message = InputMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=[Content(type="text", text=query)]
        )

        max_retries = kwargs.get("retry", 3)
        read_only = kwargs.get("read_only", False)
        include_user_query = kwargs.get("include_user_query", True)
        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "X-User-Id": self.user_id,
                        "X-Request-Id": str(uuid.uuid4()),
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
                    }

                    async with session.post(
                        f"{self.api_base_url}/api/v1/memory/query",
                        json={
                            "query": input_message.model_dump(),
                            "memory_lib_id": self.user_id,
                            "read_only": read_only,
                            "include_user_query": include_user_query,
                            "require_raw_recall_message_id_list": False
                        },
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        if response.status != 200:
                            had_error = True
                            error_text = await response.text()
                            logger.warning(
                                f"Memory query failed (attempt {attempt + 1}/{max_retries}): "
                                f"status={response.status}, response={error_text}"
                            )
                            last_error = Exception(f"HTTP {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1 * (attempt + 1))
                            continue

                        if had_error:
                            logger.success(f"✅ Memory query recovered after {attempt} retry(ies)")

                        result = await response.json()
                        result = QueryResult(**result)

                        # 转换为标准消息格式
                        messages = []
                        for item in result.ready_messages:
                            content = ""
                            for c in item.content:
                                if c.type == "text" and c.text:
                                    content += c.text

                            if content.strip():
                                messages.append({
                                    "role": item.role or "user",
                                    "content": content
                                })

                        return messages

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                had_error = True
                last_error = e
                logger.warning(
                    f"Memory query network error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        logger.error(f"Memory query failed after {max_retries} attempts: {last_error}")
        return []

    async def add(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """
        添加记忆（追加助手消息）

        注意：Memecho 的添加是通过 search 时自动添加 user 消息，
        然后通过 append_assistant 添加 assistant 消息。

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            是否添加成功
        """
        # 查找 assistant 消息并追加
        for msg in messages:
            if msg.get("role") == "assistant":
                return await self._append_assistant(msg.get("content", ""))
        return True

    async def _append_assistant(self, response: str, max_retries: int = 3) -> bool:
        """
        追加助手回复

        Args:
            response: 助手回复内容
            max_retries: 最大重试次数

        Returns:
            是否添加成功
        """
        assistant_message = InputMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=[Content(type="text", text=response)]
        )

        last_error = None
        had_error = False

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "X-User-Id": self.user_id,
                        "X-Request-Id": str(uuid.uuid4()),
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
                    }

                    async with session.post(
                        f"{self.api_base_url}/api/v1/memory/append-assistant-message",
                        json={
                            "assistant_message": assistant_message.model_dump(),
                            "memory_lib_id": self.user_id
                        },
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response_obj:
                        if not re.match(r"^2\d\d$", str(response_obj.status)):
                            had_error = True
                            error_text = await response_obj.text()
                            logger.warning(
                                f"Memory append failed (attempt {attempt + 1}/{max_retries}): "
                                f"status={response_obj.status}, response={error_text}"
                            )
                            last_error = Exception(f"HTTP {response_obj.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1 * (attempt + 1))
                            continue
                        if had_error:
                            logger.success(f"✅ Memory append recovered after {attempt} retry(ies)")
                        return True

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                had_error = True
                last_error = e
                logger.warning(
                    f"Memory append network error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        logger.error(f"Memory append failed after {max_retries} attempts: {last_error}")
        return False

    async def delete(self, memory_id: str) -> bool:
        """
        删除指定记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        # Memecho 目前不支持单条记忆删除
        logger.warning(f"Memecho does not support single memory deletion: {memory_id}")
        return False



