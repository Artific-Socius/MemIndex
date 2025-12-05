"""
Mem0Agent - 基于 Mem0 的 Agent 实现

使用 Mem0 作为记忆系统的 Agent 实现。
"""

from __future__ import annotations

import datetime
import os
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import tiktoken
from loguru import logger

from .base_agent import BaseAgent
from components.chat.llm_chat import LLMChat
from components.memory.mem0_memory import Mem0Memory

if TYPE_CHECKING:
    from utils.controller import LLMController
    from prompts import PromptManager


def _is_prompt_debug() -> bool:
    """检查是否启用 prompt 调试日志"""
    return os.environ.get("PROMPT_DEBUG_LOG", "").lower() in ("1", "true", "yes", "on")


def _debug_log(message: str) -> None:
    """仅在 PROMPT_DEBUG_LOG 环境变量启用时打印调试日志"""
    if _is_prompt_debug():
        logger.info(f"[PROMPT_DEBUG] {message}")


class Mem0Agent(BaseAgent):
    """
    基于 Mem0 的 Agent 实现

    使用 Mem0 Cloud API 作为记忆系统。
    """

    @property
    def has_memory_backend(self) -> bool:
        """Mem0 Agent 有真实的 memory 后端"""
        return True

    def __init__(
        self,
        llm_controller: "LLMController",
        model: str,
        context_window: int,
        temperature: float = 0.7,
        api_key: str = None,
        prompt_manager: Optional["PromptManager"] = None,
        chat_prompt_key: str = None,
    ):
        """
        初始化 Mem0 Agent

        Args:
            llm_controller: LLM 控制器实例
            model: 使用的模型名称
            context_window: 上下文窗口大小（tokens）
            temperature: 生成温度
            api_key: Mem0 API Key
            prompt_manager: 提示词管理器（可选，不传则使用全局单例）
            chat_prompt_key: Chat 提示词的 key（可选，不传则使用默认）
        """
        super().__init__(f"Mem0 Agent - {model.replace('/', '-')}")

        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.llm_controller = llm_controller

        # 提示词管理
        if prompt_manager is None:
            from prompts import get_prompt_manager
            prompt_manager = get_prompt_manager()
        self.prompt_manager = prompt_manager
        self.chat_prompt_key = chat_prompt_key

        _debug_log(f"Mem0Agent initialized with chat_prompt_key='{chat_prompt_key}'")

        # 初始化 Chat 模块
        self.chat = LLMChat(
            llm_controller=llm_controller,
            model=model,
            context_window=context_window,
            temperature=temperature,
        )

        # 初始化 Memory 模块
        self.memory = Mem0Memory(
            llm_controller=llm_controller,
            api_key=api_key,
        )

        # Token 编码器
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # 更新额外元数据
        self.extra_metadata["Memory Provider"] = {
            "type": "text",
            "value": "Mem0",
            "description": "Memory Provider"
        }
        self.extra_metadata["user_id"] = {
            "type": "text",
            "value": "N/A",
            "description": "Mem0 User ID"
        }

    async def _is_delete_request(self, message: str) -> bool:
        """判断用户消息是否是删除记忆的请求"""
        _debug_log(f"Mem0Agent._is_delete_request() using chat_prompt_key='{self.chat_prompt_key}'")
        delete_check_prompt = self.prompt_manager.get_chat_delete_request_check(self.chat_prompt_key)

        if _is_prompt_debug():
            preview = delete_check_prompt[:100].replace('\n', '\\n') + ('...' if len(delete_check_prompt) > 100 else '')
            _debug_log(f"delete_request_check prompt: '{preview}'")

        prompt = [
            {
                "role": "system",
                "content": delete_check_prompt
            },
            {
                "role": "user",
                "content": f"用户消息: {message}\n\n这是一个删除记忆的请求吗？"
            }
        ]
        try:
            response = await self.llm_controller.completion_with_retry_async_t(
                self.model, prompt, temperature=0.0
            )
            if response and response.completion:
                self.token_input += response.token_information.input_tokens
                self.token_output += response.token_information.output_tokens
                return "YES" in response.completion.upper()
        except Exception as e:
            logger.error(f"Delete request check failed: {e}")
        return False

    async def _select_memories_to_delete(
        self,
        message: str,
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """让 LLM 选择要删除的记忆"""
        if not memories:
            return []

        memory_list = []
        for mem in memories:
            mem_id = mem.get("id", "")
            mem_content = mem.get("memory", "")
            if mem_id and mem_content:
                memory_list.append(f"[{mem_id}] {mem_content}")

        if not memory_list:
            return []

        _debug_log(f"Mem0Agent._select_memories_to_delete() using chat_prompt_key='{self.chat_prompt_key}'")
        select_delete_prompt = self.prompt_manager.get_chat_select_memories_to_delete(self.chat_prompt_key)

        if _is_prompt_debug():
            preview = select_delete_prompt[:100].replace('\n', '\\n') + ('...' if len(select_delete_prompt) > 100 else '')
            _debug_log(f"select_memories_to_delete prompt: '{preview}'")

        prompt = [
            {
                "role": "system",
                "content": select_delete_prompt
            },
            {
                "role": "user",
                "content": f"""用户请求: {message}

可用记忆列表:
{chr(10).join(memory_list)}

请列出需要删除的记忆ID:"""
            }
        ]

        try:
            response = await self.llm_controller.completion_with_retry_async_t(
                self.model, prompt, temperature=0.0
            )
            if response and response.completion:
                self.token_input += response.token_information.input_tokens
                self.token_output += response.token_information.output_tokens

                content = response.completion.strip()
                if "NONE" in content.upper():
                    return []

                ids_to_delete = []
                for line in content.split("\n"):
                    line = line.strip().lstrip("-").lstrip("*").strip().strip("[]")
                    for mem in memories:
                        if mem.get("id") == line:
                            ids_to_delete.append(line)
                            break
                return ids_to_delete
        except Exception as e:
            logger.error(f"Select memories to delete failed: {e}")
        return []

    async def _handle_delete_request(self, message: str) -> str | None:
        """处理删除记忆的请求"""
        is_delete = await self._is_delete_request(message)
        if not is_delete:
            return None

        logger.debug(f"Detected delete request: {message}")

        memories = await self.memory.search(message)
        if not memories:
            return "没有找到需要删除的相关记忆。"

        ids_to_delete = await self._select_memories_to_delete(message, memories)
        if not ids_to_delete:
            return "没有找到需要删除的记忆。"

        deleted_count = 0
        for mem_id in ids_to_delete:
            if await self.memory.delete(mem_id):
                deleted_count += 1

        return f"已成功删除 {deleted_count} 条记忆。"

    def _build_context_from_memories(
        self,
        memories: List[Dict[str, Any]],
        current_message: str
    ) -> List[Dict[str, str]]:
        """根据召回的记忆构建上下文消息"""
        messages = []

        if memories:
            memory_texts = [f"- {mem.get('memory', '')}" for mem in memories if mem.get("memory")]
            if memory_texts:
                _debug_log(f"Mem0Agent._build_context_from_memories() using chat_prompt_key='{self.chat_prompt_key}'")
                memory_context_prefix = self.prompt_manager.get_chat_memory_context_prefix(self.chat_prompt_key)

                if _is_prompt_debug():
                    preview = memory_context_prefix[:80].replace('\n', '\\n') + ('...' if len(memory_context_prefix) > 80 else '')
                    _debug_log(f"memory_context_prefix: '{preview}'")

                memory_context = memory_context_prefix + "\n".join(memory_texts)
                messages.append({"role": "system", "content": memory_context})

        # 添加最近的对话历史
        recent_history = self.memory.get_recent_history(self.chat_history, max_turns=3)
        messages.extend(recent_history)

        # 添加当前用户消息
        messages.append({"role": "user", "content": self._format_time(current_message)})

        return messages

    @staticmethod
    def _format_time(message: str) -> str:
        """格式化消息，添加当前时间"""
        current = datetime.datetime.now()
        current_time = current.strftime("%Y-%m-%d %H:%M:%S")
        return f"[当前时间为: {current_time}]\n{message}"

    async def send_message(self, message: str) -> str:
        """
        发送消息并获取回复

        Args:
            message: 用户消息

        Returns:
            Agent 回复
        """
        # 初始化用户ID
        if not self.memory.user_id:
            await self.memory.initialize()
            self.extra_metadata["user_id"]["value"] = self.memory.user_id

        # 检查是否是删除请求
        delete_result = await self._handle_delete_request(message)
        if delete_result is not None:
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": delete_result})
            return delete_result

        # 搜索相关记忆
        memory_time_start = time.time()
        memories = await self.memory.search(message)
        memory_time_end = time.time()
        memory_latency = memory_time_end - memory_time_start
        self.record_memory_latency(memory_latency)
        logger.debug(f"Memory search took {memory_latency:.2f} seconds and returned {len(memories)} memories")

        # 构建上下文消息
        context_messages = self._build_context_from_memories(memories, message)

        # 截断消息
        messages = self.chat.auto_truncate_messages(context_messages)

        # 生成回复
        response = None
        for _ in range(3):
            try:
                response = await self.llm_controller.completion_with_retry_async_t(
                    self.model, messages, temperature=self.temperature
                )
                if response and response.completion:
                    # 记录 chat 延迟
                    self.record_chat_latency(response.llm_time_elapsed)
                    break
            except Exception as e:
                logger.error(f"LLM completion failed: {e}")

        if not response or not response.completion:
            raise Exception("LLM completion failed after retries")

        # 更新统计
        self.token_input += response.token_information.input_tokens
        self.token_output += response.token_information.output_tokens

        # 保存到记忆
        await self.memory.add([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.completion},
        ])

        # 更新对话历史
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response.completion})

        # 更新 token 表格
        current_input_tokens = sum(self.get_message_tokens(msg) for msg in messages)
        self.update_token_table(current_input_tokens)

        return response.completion
