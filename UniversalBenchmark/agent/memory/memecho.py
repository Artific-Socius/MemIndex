from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from typing import Any, Optional

import requests  # type: ignore[import-untyped]
from loguru import logger
import tiktoken

from agent.progress import get_progress

from .base import MemoryMixin, TurnTrace

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
    use_simulated_import:
        若为 *True*，语料与批量导入不走 Memecho 原生 fast 接口，
        而是将文本按 token 上限聚合后，逐条走 ``query`` + ``append``，
        助手回复固定为 ``simulated_import_ack_text``（不调用 LLM）。
    simulated_import_max_tokens:
        模拟导入时每条用户消息的最大 token 数（默认 5000）。
    simulated_import_ack_text:
        模拟导入时每轮 ``add_response`` 写入的固定文本（默认 ``Copy that.``）。
    """

    def __init__(
        self,
        api_base_url: str = "https://api.memecho.cloud",
        # api_base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        memory_lib_id: Optional[str] = None,
        include_user_query: bool = True,
        read_only: bool = False,
        max_retries: int = 10,
        custom_headers: Optional[dict[str, str]] = None,
        use_simulated_import: bool = False,
        simulated_import_max_tokens: int = 5000,
        simulated_import_ack_text: str = "Copy that.",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key or os.getenv("MEMECHO_API_KEY")
        self._memory_lib_id = memory_lib_id
        self._include_user_query = include_user_query
        self._read_only = read_only
        self._max_retries = max_retries
        self._custom_headers: dict[str, str] = dict(custom_headers) if custom_headers else {}
        self._persistent_lib: bool = memory_lib_id is not None
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self._use_simulated_import = bool(use_simulated_import)
        self._simulated_import_max_tokens = max(1, int(simulated_import_max_tokens))
        self._simulated_import_ack_text = str(simulated_import_ack_text)

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

    def ensure_memory_library(self) -> str:
        """确保存在 Memecho 记忆库并返回真实 memory_lib_id。"""
        self._ensure_initialized()
        return str(self._memory_lib_id)

    def get_memory_library_id(self) -> str:
        """返回当前 Memecho 记忆库 ID。"""
        return self.ensure_memory_library()

    # ------------------------------------------------------------------
    # 模拟导入：按 token 聚合 + query + append（不调用 LLM）
    # ------------------------------------------------------------------

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """将过长文本按 token 切分为多段（每段不超过 max_tokens）。"""
        if max_tokens < 1:
            max_tokens = 1
        ids = self.encoding.encode(text)
        if not ids:
            return []
        parts: list[str] = []
        for i in range(0, len(ids), max_tokens):
            chunk_ids = ids[i : i + max_tokens]
            parts.append(self.encoding.decode(chunk_ids))
        return parts

    def _merge_documents_by_tokens(
        self,
        documents: list[str],
        max_tokens: int,
    ) -> list[str]:
        """将多篇文档合并为若干条文本，每条 token 数不超过 max_tokens。"""
        if max_tokens < 1:
            max_tokens = 1
        sep = "\n\n"
        sep_tok = len(self.encoding.encode(sep))
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tok = 0

        for doc in documents:
            if not doc:
                continue
            doc_tok = len(self.encoding.encode(doc))
            if doc_tok > max_tokens:
                if current_parts:
                    chunks.append(sep.join(current_parts))
                    current_parts = []
                    current_tok = 0
                chunks.extend(self._split_text_by_tokens(doc, max_tokens))
                continue

            extra = sep_tok + doc_tok if current_parts else doc_tok
            if current_tok + extra > max_tokens:
                chunks.append(sep.join(current_parts))
                current_parts = [doc]
                current_tok = doc_tok
            else:
                current_parts.append(doc)
                current_tok += extra

        if current_parts:
            chunks.append(sep.join(current_parts))
        return chunks

    def _simulated_import_text_chunks(self, chunks: list[str]) -> int:
        """对每条聚合文本执行 query + 固定 append，返回写入轮次数。"""
        ident = self.memory_identifier
        ack = self._simulated_import_ack_text
        count = 0
        pg = get_progress()
        nonempty = [c for c in chunks if c.strip()]
        total_n = len(nonempty) if nonempty else 1
        ph = pg.add_task(
            f"[{ident}] 模拟导入",
            total=float(total_n),
            task_key="memecho:simulated_text_chunks",
        )
        try:
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                logger.debug(
                    f"[{ident}] 模拟导入 [{i + 1}/{len(chunks)}] "
                    f"({len(self.encoding.encode(chunk))} tokens)"
                )
                pg.update(
                    ph,
                    description=(
                        f"[{ident}] 模拟导入 "
                        f"[{count + 1}/{total_n}]"
                    ),
                )
                self.get_messages(chunk)
                self.add_response(ack)
                count += 1
                pg.advance(ph, 1)
            if not nonempty:
                pg.advance(ph, 1)
        finally:
            pg.remove_task(ph)
        logger.info(f"[{ident}] 模拟导入完成: {count} 轮 query+append")
        return count

    # ------------------------------------------------------------------
    # MemoryMixin 接口实现
    # ------------------------------------------------------------------

    def get_messages(self, user_input: str) -> list[dict[str, Any]]:
        self._ensure_initialized()

        client_user_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        query_msg: dict[str, Any] = {
            "id": client_user_id,
            "role": "user",
            "content": [{"type": "text", "text": user_input}],
        }
        # API 要求 body 含 request_id（与 X-Request-Id、trace.query_request_id 一致）
        result: dict[str, Any] = self._request_with_retry(
            "POST",
            "/api/v1/memory/query",
            json={
                "query": query_msg,
                "memory_lib_id": self._memory_lib_id,
                "read_only": self._read_only,
                "include_user_query": self._include_user_query,
                "require_raw_recall_message_id_list": False,
                "request_id": request_id,
            },
            extra_headers={
                "X-User-Id": self._memory_lib_id or "",
                "X-Request-Id": request_id,
            },
        )

        server_user_id = result.get("user_message_id") or ""
        if server_user_id:
            resolved_user_id = server_user_id
            id_source = "provider"
        else:
            resolved_user_id = client_user_id
            id_source = "framework"

        if self._current_turn_trace is not None:
            self._current_turn_trace.user_message_id = resolved_user_id
            self._current_turn_trace.id_source = id_source
            self._current_turn_trace.query_request_id = request_id

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

        client_assistant_id = str(uuid.uuid4())
        append_request_id = str(uuid.uuid4())

        assistant_msg: dict[str, Any] = {
            "id": client_assistant_id,
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
        }
        result = self._request_with_retry(
            "POST",
            "/api/v1/memory/append-assistant-message",
            json={
                "assistant_message": assistant_msg,
                "memory_lib_id": self._memory_lib_id,
            },
            extra_headers={
                "X-User-Id": self._memory_lib_id or "",
                "X-Request-Id": append_request_id,
            },
        )

        server_assistant_id = ""
        if isinstance(result, dict):
            server_assistant_id = result.get("assistant_message_id") or ""

        resolved_id = server_assistant_id or client_assistant_id
        if self._current_turn_trace is not None:
            self._current_turn_trace.assistant_message_id = resolved_id
            self._current_turn_trace.append_request_id = append_request_id
            if server_assistant_id:
                self._current_turn_trace.id_source = "provider"

    def bulk_import(
        self,
        conversations: list[tuple[str, str]],
        *,
        timeout: int = 600,
    ) -> int:
        """Memecho 原生批量导入（覆盖基类实现）。

        将对话列表转换为 Memecho 消息格式后，调用
        ``/api/v1/memory/memory_import_fast`` 端点一次性提交，
        通过 SSE 流实时跟踪导入进度。

        Parameters
        ----------
        conversations:
            对话轮次列表，每个元素为 ``(user_input, assistant_response)``。
        timeout:
            SSE 流的超时时间（秒），默认 600。

        Returns
        -------
        int
            成功导入的记忆条数。
        """
        self._ensure_initialized()
        ident = self.memory_identifier

        if self._use_simulated_import:
            logger.info(
                f"[{ident}] 模拟批量导入: {len(conversations)} 条对话 "
                f"(query+append，助手固定 {self._simulated_import_ack_text!r}，"
                f"忽略每条中的 assistant 文本)"
            )
            pg = get_progress()
            n_total = len(conversations)
            turn_total = float(n_total) if n_total > 0 else 1.0
            sh = pg.add_task(
                f"[{ident}] 模拟 bulk_import",
                total=turn_total,
                task_key="memecho:simulated_bulk_import",
            )
            n = 0
            try:
                for user_input, _assistant in conversations:
                    self.get_messages(user_input)
                    self.add_response(self._simulated_import_ack_text)
                    n += 1
                    pg.advance(sh, 1)
                if n_total == 0:
                    pg.advance(sh, 1)
            finally:
                pg.remove_task(sh)
            return n

        memories: list[dict[str, Any]] = []
        for user_input, assistant_response in conversations:
            memories.append({
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": [{"type": "text", "text": user_input}],
            })
            memories.append({
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            })

        logger.info(
            f"[{ident}] 开始原生批量导入: "
            f"{len(conversations)} 条对话 ({len(memories)} 条消息)"
        )

        url = f"{self._api_base_url}/api/v1/memory/memory_import_fast"
        headers = self._build_headers()
        payload: dict[str, Any] = {
            "memories": memories,
            "memory_lib_id": self._memory_lib_id,
        }

        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Memecho 批量导入请求失败: "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            return self._consume_import_sse(resp)
        finally:
            resp.close()

    def _consume_import_sse(
        self,
        resp: Any,
        arrived: Optional[threading.Event] = None,
    ) -> int:
        """解析 import SSE 流，打印进度并返回导入条数。

        Parameters
        ----------
        arrived:
            If provided, will be :meth:`~threading.Event.set` as soon
            as the first SSE ``data:`` line is received, signalling
            that the request has reached the server.
        """
        ident = self.memory_identifier
        imported_count = 0
        last_stage = ""

        pg = get_progress()
        sse_h = pg.add_task(
            f"[{ident}] memory_import_fast",
            total=100.0,
            task_key="memecho:memory_import_fast_sse",
        )
        logger.debug(f"[{ident}] 等待 SSE 首事件...")
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                if arrived is not None and not arrived.is_set():
                    arrived.set()

                try:
                    event: dict[str, Any] = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                stage = event.get("stage", "")
                message = event.get("message", "")
                raw_prog = event.get("progress", 0)
                try:
                    progress_f = float(raw_prog)
                except (TypeError, ValueError):
                    progress_f = 0.0

                if event_type == "connected":
                    req_id = event.get("request_id", "")
                    logger.info(
                        f"[{ident}] SSE 连接已建立 "
                        f"(request_id={req_id})"
                    )
                elif event_type == "progress":
                    if stage != last_stage or stage == "completed":
                        logger.info(
                            f"[{ident}] [{stage}] "
                            f"{message} ({progress_f:.0f}%)"
                        )
                        last_stage = stage

                    short_msg = (message or "")[:48]
                    pg.update(
                        sse_h,
                        completed=min(100.0, max(0.0, progress_f)),
                        description=(
                            f"[{ident}] import [{stage}] {short_msg}"
                        ),
                    )

                    extra = event.get("extra_data")
                    if stage == "completed" and extra:
                        msg_ids = extra.get("message_ids", [])
                        elapsed = extra.get("elapsed_time", 0)
                        imported_count = len(msg_ids)
                        logger.info(
                            f"[{ident}] 导入完成: "
                            f"{imported_count} 条记忆, "
                            f"耗时 {elapsed:.2f}s"
                        )
        finally:
            pg.remove_task(sse_h)

        return imported_count

    def reset(self) -> None:
        if not self._persistent_lib:
            self._memory_lib_id = None

    def set_persistent_lib(self, lib_id: str) -> None:
        """Point to an existing library and mark it as persistent.

        Persistent libraries survive ``reset()`` calls, allowing the
        Runner to reuse a pre-imported corpus across scenarios.
        """
        self._memory_lib_id = lib_id
        self._persistent_lib = True

    # ------------------------------------------------------------------
    # 语料文档导入 (import_file_fast)
    # ------------------------------------------------------------------

    _DEFAULT_CHUNK_CHARS = 5_000_000

    def import_corpus(
        self,
        documents: list[str],
        corpus_id: str = "",
        *,
        max_chunk_chars: int = _DEFAULT_CHUNK_CHARS,
        first_event_timeout: int = 60,
    ) -> str:
        """Import corpus documents via ``import_file_fast`` (base64).

        Small documents are merged into larger chunks to reduce the
        number of sequential API calls.  Each chunk is base64-encoded
        and submitted as a ``data:`` URI.

        Returns the ``memory_lib_id`` of the library that received
        the corpus.
        """
        self._ensure_initialized()
        ident = self.memory_identifier

        if self._use_simulated_import:
            tok_chunks = self._merge_documents_by_tokens(
                documents, self._simulated_import_max_tokens,
            )
            total_tok = sum(len(self.encoding.encode(c)) for c in tok_chunks)
            logger.info(
                f"[{ident}] 模拟语料导入: {len(documents)} 篇文档 "
                f"→ {len(tok_chunks)} 条消息 (每条约 ≤ "
                f"{self._simulated_import_max_tokens} tokens, 共 {total_tok} tokens)"
            )
            self._simulated_import_text_chunks(tok_chunks)
            self._persistent_lib = True
            return str(self._memory_lib_id)

        chunks = self._merge_documents(documents, max_chunk_chars)
        total_chars = sum(len(c) for c in chunks)
        logger.info(
            f"[{ident}] 语料导入: {len(documents)} 篇文档 "
            f"→ 合并为 {len(chunks)} 个块 "
            f"(共 {total_chars:,} 字符)"
        )

        pg = get_progress()
        n_chunks = len(chunks)
        chunk_total = float(n_chunks) if n_chunks > 0 else 1.0
        ch_h = pg.add_task(
            f"[{ident}] import_file_fast 块",
            total=chunk_total,
            task_key="memecho:import_corpus_chunks",
        )
        try:
            for i, chunk_text in enumerate(chunks):
                tokens = len(self.encoding.encode(chunk_text))
                b64 = base64.b64encode(
                    chunk_text.encode("utf-8"),
                ).decode("ascii")
                data_uri = f"data:text/plain;base64,{b64}"

                b64_kb = len(b64) / 1024
                logger.info(
                    f"[{ident}] import_file_fast "
                    f"[{i + 1}/{len(chunks)}] "
                    f"({len(chunk_text):,} chars, "
                    f"base64 ~{b64_kb:.0f} KB) {tokens} tokens"
                )
                pg.update(
                    ch_h,
                    description=(
                        f"[{ident}] 语料块 [{i + 1}/{n_chunks}]"
                    ),
                )
                self._import_file_fast(
                    data_uri,
                    first_event_timeout=first_event_timeout,
                )
                pg.advance(ch_h, 1)
            if n_chunks == 0:
                pg.advance(ch_h, 1)
        finally:
            pg.remove_task(ch_h)

        logger.info(
            f"[{ident}] 语料导入完成 "
            f"(library={self._memory_lib_id})"
        )
        self._persistent_lib = True
        return str(self._memory_lib_id)

    _IMPORT_RETRIES = 5

    def _import_file_fast(
        self,
        file_url: str,
        *,
        first_event_timeout: int = 60,
    ) -> int:
        """Call ``/api/v1/memory/import_file_fast`` and consume SSE.

        The HTTP request and SSE consumption run inside a daemon
        thread.  The main thread only enforces a timeout on the
        **arrival of the first SSE event** — this covers the upload
        phase and initial server response, which is the stage where
        requests can silently get lost in transit.

        Once the first SSE ``data:`` line is received (proving the
        server got the request), **all** timeouts are cancelled: the
        socket read timeout is ``None`` and the thread is allowed to
        run to completion with no wall-clock limit.

        Parameters
        ----------
        first_event_timeout:
            Max seconds to wait for the first SSE data line.  If no
            data arrives within this window the attempt is aborted and
            retried.
        """
        url = f"{self._api_base_url}/api/v1/memory/import_file_fast_v2"
        headers = self._build_headers()
        payload: dict[str, Any] = {
            "file_url": file_url,
            "memory_lib_id": self._memory_lib_id,
        }
        ident = self.memory_identifier
        last_error: Exception | None = None
        payload_kb = len(file_url) / 1024

        for attempt in range(self._IMPORT_RETRIES):
            result_box: dict[str, Any] = {}
            resp_box: list[Any] = []
            arrived = threading.Event()

            def _worker() -> None:
                try:
                    logger.debug(
                        f"[{ident}] 发送 import_file_fast 请求 "
                        f"(attempt {attempt + 1}/{self._IMPORT_RETRIES}, "
                        f"payload ~{payload_kb:.0f} KB)..."
                    )
                    resp = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        stream=True,
                        timeout=(30, None),
                    )
                    resp_box.append(resp)
                    logger.debug(
                        f"[{ident}] 收到响应 HTTP {resp.status_code}"
                    )
                    if resp.ok:
                        try:
                            result_box["count"] = (
                                self._consume_import_sse(resp, arrived)
                            )
                        finally:
                            resp.close()
                    else:
                        result_box["error"] = RuntimeError(
                            f"HTTP {resp.status_code}: "
                            f"{resp.text[:200]}"
                        )
                        resp.close()
                except Exception as exc:
                    result_box["error"] = exc
                finally:
                    arrived.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            server_reached = arrived.wait(timeout=first_event_timeout)

            if not server_reached:
                for r in resp_box:
                    try:
                        r.close()
                    except Exception:
                        pass
                logger.warning(
                    f"[{ident}] import_file_fast 首事件超时 "
                    f"({first_event_timeout}s), 请求可能未到达服务器, "
                    f"attempt {attempt + 1}/{self._IMPORT_RETRIES}"
                )
                last_error = TimeoutError(
                    f"No SSE event within {first_event_timeout}s"
                )
            else:
                t.join()
                if "error" in result_box:
                    last_error = result_box["error"]
                    logger.warning(
                        f"[{ident}] import_file_fast 失败 "
                        f"(attempt {attempt + 1}/"
                        f"{self._IMPORT_RETRIES}): {last_error}"
                    )
                else:
                    return result_box.get("count", 0)

            if attempt < self._IMPORT_RETRIES - 1:
                wait = 5 * (attempt + 1)
                logger.info(
                    f"[{ident}] 等待 {wait}s 后重试..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Memecho import_file_fast 在 "
            f"{self._IMPORT_RETRIES} 次重试后仍然失败"
        ) from last_error

    @staticmethod
    def _merge_documents(
        documents: list[str],
        max_chunk_chars: int,
    ) -> list[str]:
        """Merge small documents into larger chunks.

        Adjacent documents are concatenated (separated by double
        newlines) until adding the next one would exceed
        ``max_chunk_chars``.  A single document larger than the
        limit is emitted as its own chunk.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for doc in documents:
            if not doc:
                continue
            sep_len = 2 if current_parts else 0  # "\n\n"
            if current_parts and current_len + sep_len + len(doc) > max_chunk_chars:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            current_parts.append(doc)
            current_len += (2 if len(current_parts) > 1 else 0) + len(doc)

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

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
        if self._custom_headers:
            headers.update(self._custom_headers)
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
