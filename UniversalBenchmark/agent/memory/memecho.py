from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

import requests  # type: ignore[import-untyped]
from loguru import logger

from .base import MemoryMixin

_DEFAULT_TIMEOUT = 60


class MemechoMemory(MemoryMixin):
    """基于 Memecho 云 API 的记忆实现。

    Memecho 在服务端管理完整的对话上下文。每次调用 :meth:`get_messages`
    时，用户消息会被发送到 ``/api/v1/memory/query`` 端点，该端点会
    **同时存储用户消息并返回** ``ready_messages`` —— 一个可直接用于
    LLM 生成的完整消息列表。

    :meth:`add_response` 通过 ``/api/v1/memory/append-assistant-message``
    追加助手回复。

    Parameters
    ----------
    api_base_url:
        Memecho API 根地址（末尾不带 ``/``）。
    api_key:
        Bearer 令牌。若未提供则从 ``MEMECHO_API_KEY`` 环境变量读取。
    memory_lib_id:
        已有的记忆库 ID。若为 *None*，则在首次调用 :meth:`get_messages`
        时自动创建（也可通过 :meth:`initialize` 手动创建）。
    include_user_query:
        query 端点返回的 ``ready_messages`` 中是否包含用户消息。
    read_only:
        若为 *True*，query 端点不会持久化该条用户消息。
    max_retries:
        每个 HTTP 请求的最大重试次数。
    """

    def __init__(
        self,
        api_base_url: str = "https://api.memecho.cloud",
        api_key: Optional[str] = None,
        memory_lib_id: Optional[str] = None,
        include_user_query: bool = True,
        read_only: bool = False,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key or os.getenv("MEMECHO_API_KEY")
        self._memory_lib_id = memory_lib_id
        self._include_user_query = include_user_query
        self._read_only = read_only
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self, memory_lib_id: Optional[str] = None) -> str:
        """创建（或指定）一个记忆库，返回记忆库 ID。"""
        if memory_lib_id:
            self._memory_lib_id = memory_lib_id
            return self._memory_lib_id

        alias = f"memecho_user_{uuid.uuid4().hex[:12]}"
        data = self._request_with_retry(
            "POST",
            "/api/v1/memory/create",
            json={"alias": alias},
            timeout=30,
        )
        self._memory_lib_id = data["id"]
        return str(self._memory_lib_id)

    @property
    def memory_lib_id(self) -> Optional[str]:
        return self._memory_lib_id

    # ------------------------------------------------------------------
    # MemoryMixin 接口实现
    # ------------------------------------------------------------------

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        self._ensure_initialized()

        query_msg: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": [{"type": "text", "text": user_input}],
        }
        result: dict[str, Any] = self._request_with_retry(
            "POST",
            "/api/v1/memory/query",
            json={
                "query": query_msg,
                "memory_lib_id": self._memory_lib_id,
                "read_only": self._read_only,
                "include_user_query": self._include_user_query,
                "require_raw_recall_message_id_list": False,
            },
            extra_headers={
                "X-User-Id": self._memory_lib_id or "",
                "X-Request-Id": str(uuid.uuid4()),
            },
        )

        # 将 Memecho 格式的 ready_messages 转换为 OpenAI 标准格式
        messages: list[dict[str, Any]] = []
        for item in result.get("ready_messages", []):
            text_parts: list[str] = []
            for c in item.get("content", []):
                if c.get("type") == "text" and c.get("text"):
                    text_parts.append(c["text"])
            content = "".join(text_parts)
            if content.strip():
                messages.append(
                    {
                        "role": item.get("role") or "user",
                        "content": content,
                    }
                )

        if self._system_prompt:
            messages = self._build_system_messages() + messages
        return messages

    def add_response(self, content: str) -> None:
        self._ensure_initialized()

        assistant_msg: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
        }
        self._request_with_retry(
            "POST",
            "/api/v1/memory/append-assistant-message",
            json={
                "assistant_message": assistant_msg,
                "memory_lib_id": self._memory_lib_id,
            },
            extra_headers={
                "X-User-Id": self._memory_lib_id or "",
                "X-Request-Id": str(uuid.uuid4()),
            },
        )

    def reset(self) -> None:
        self._memory_lib_id = None

    # ------------------------------------------------------------------
    # HTTP 工具方法
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._memory_lib_id:
            self.initialize()

    def _build_headers(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        timeout: int = _DEFAULT_TIMEOUT,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """发起 HTTP 请求，失败时按递增间隔重试。"""
        url = f"{self._api_base_url}{path}"
        headers = self._build_headers(extra_headers)
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                )
                if resp.ok:
                    return resp.json()

                logger.warning(f"Memecho {path} 请求失败 " f"(第 {attempt + 1}/{self._max_retries} 次): " f"status={resp.status_code}, body={resp.text[:200]}")
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as exc:
                logger.warning(f"Memecho {path} 网络错误 " f"(第 {attempt + 1}/{self._max_retries} 次): " f"{type(exc).__name__}: {exc}")
                last_error = exc

            if attempt < self._max_retries - 1:
                time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"Memecho {path} 在 {self._max_retries} 次重试后仍然失败") from last_error
