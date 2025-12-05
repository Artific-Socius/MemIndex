"""
LLMController - LLM 控制器（基于 OpenAI SDK）

提供与各种 LLM 提供商交互的统一接口。
这是基于 OpenAI SDK 的实现，需要在 config.yaml 中配置各个 Provider。

注意: 推荐使用 litellm_controller.py 中的 LiteLLMController，
      它基于 LiteLLM 库，支持更多 Provider 且配置更简单。

核心功能:
    1. 管理多个 LLM Provider（OpenAI 兼容接口）
    2. 自动选择 Provider 和重试
    3. Token 使用统计
    4. 同步/异步调用支持

配置方式:
    在 config.yaml 中配置 providers:
    ```yaml
    providers:
      - name: openai
        base_url: https://api.openai.com/v1
        api_key_env: OPENAI_API_KEY
      - name: volcano
        base_url: https://ark.cn-beijing.volces.com/api/v3
        api_key_env: VOLCANO_API_KEY
    ```
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from copy import deepcopy
from typing import Tuple, TYPE_CHECKING

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

if TYPE_CHECKING:
    from config import Config, LLMProvider


def is_inside_async() -> bool:
    """
    检查是否在异步上下文中

    用于自动选择同步或异步的 Provider。

    Returns:
        是否在异步上下文中
    """
    try:
        return asyncio.get_running_loop() is not None
    except RuntimeError:
        return False


def run_async_adaptively(coro):
    """
    自适应运行异步协程

    如果当前在异步上下文中，返回 Future；
    否则使用 asyncio.run() 运行。

    Args:
        coro: 协程对象

    Returns:
        协程结果或 Future
    """
    if is_inside_async():
        return asyncio.ensure_future(coro)
    else:
        return asyncio.run(coro)


class TokenUseInformation(BaseModel):
    """
    Token 使用信息

    记录单次 LLM 调用的 Token 使用情况。

    Attributes:
        input_tokens: 输入 Token 数量（Prompt）
        output_tokens: 输出 Token 数量（Completion）
        reasoning_tokens: 推理 Token 数量（仅推理模型有）
        content_tokens: 内容 Token 数量（output - reasoning）
    """
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    content_tokens: int


class LLMResult(BaseModel):
    """
    LLM 调用结果

    封装单次 LLM 调用的完整结果。

    Attributes:
        model_name: 模型名称
        provider_name: Provider 名称
        messages: 发送的消息列表
        completion: LLM 的响应内容
        reasoning: LLM 的推理内容（仅推理模型有）
        time_start: 调用开始时间
        time_end: 调用结束时间
        time_elapsed: 总耗时（含重试）
        llm_time_start: 实际 LLM 调用开始时间
        llm_time_end: 实际 LLM 调用结束时间
        llm_time_elapsed: 实际 LLM 调用耗时
        retry_count: 重试次数
        token_information: Token 使用信息
    """
    model_name: str
    provider_name: str
    messages: list[dict] = []
    completion: str
    reasoning: str = ""
    time_start: float = 0
    time_end: float = 0
    time_elapsed: float = 0
    llm_time_start: float = 0
    llm_time_end: float = 0
    llm_time_elapsed: float = 0
    retry_count: int = 0
    token_information: TokenUseInformation = TokenUseInformation(
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        content_tokens=0
    )


class LLMController:
    """
    LLM 控制器

    基于 OpenAI SDK 的 LLM 控制器，支持管理多个 Provider。
    每个 Provider 使用 OpenAI 兼容的 API 接口。

    使用方式:
        controller = LLMController(config)
        await controller._init_provider()  # 初始化
        result = controller.completion("openai/gpt-4o", "Hello!")

    Attributes:
        config: 配置对象
        providers: Provider 名称到同步客户端的映射
        async_providers: Provider 名称到异步客户端的映射
        provider_config: Provider 配置
        retry_times: 重试次数
    """

    def __init__(self, config: "Config"):
        """
        初始化 LLM 控制器

        Args:
            config: 配置对象（包含 Provider 列表）
        """
        self.config = config
        self.providers: dict[str, OpenAI] = {}           # 同步客户端
        self.async_providers: dict[str, AsyncOpenAI] = {} # 异步客户端
        self.provider_config: dict[str, "LLMProvider"] = {}
        self.retry_times = self.config.llm_config.llm_retry_times

    async def _init_provider(self) -> None:
        """
        初始化所有 Provider

        并行加载所有配置的 Provider。
        """
        load_dotenv(self.config.env_file)

        # 并行加载所有 Provider
        tasks = []
        for provider in self.config.providers:
            tasks.append(asyncio.to_thread(self._load_provider, provider))

        for (provider, async_provider), provider_config in zip(
            await asyncio.gather(*tasks),
            self.config.providers
        ):
            self.providers[provider_config.name] = provider
            self.async_providers[provider_config.name] = async_provider

    @staticmethod
    def _load_provider(provider: "LLMProvider") -> Tuple[OpenAI, AsyncOpenAI]:
        """
        加载单个 Provider

        从环境变量或配置中获取 API Key，创建客户端。

        Args:
            provider: Provider 配置

        Returns:
            (同步客户端, 异步客户端)
        """
        api_key = provider.api_key or os.getenv(provider.api_key_env)
        return (
            OpenAI(api_key=api_key, base_url=provider.base_url),
            AsyncOpenAI(api_key=api_key, base_url=provider.base_url),
        )

    def adapter_provider(
        self,
        model_path: str
    ) -> Tuple[OpenAI | AsyncOpenAI, str, str]:
        """
        适配 Provider

        根据模型路径选择合适的 Provider 和客户端。
        模型路径格式: provider/model_name

        Args:
            model_path: 模型路径 (例如 openai/gpt-4o)

        Returns:
            (客户端, 模型名称, Provider名称)
        """
        provider_name = model_path.split("/")[0]
        model_name = '/'.join(model_path.split("/")[1:])

        # 根据上下文选择同步或异步客户端
        if is_inside_async():
            async_provider = self.async_providers.get(provider_name, None)
            if not async_provider:
                # 回退到第一个 Provider
                provider_name, async_provider = list(self.async_providers.items())[0]
            return async_provider, model_name, provider_name
        else:
            provider = self.providers.get(provider_name, None)
            if not provider:
                # 回退到第一个 Provider
                provider_name, provider = list(self.providers.items())[0]
            return provider, model_name, provider_name

    @staticmethod
    def payload_to_messages(
        payload: list[dict] | list[str] | str,
        all_system: bool = True,
        is_gemini: bool = False,
        force_role: str | None = None,
    ) -> list[dict]:
        """
        将 payload 转换为 OpenAI 格式的消息列表

        支持多种输入格式:
            - 字符串: 转为单条消息
            - 字符串列表: 转为多条消息
            - 消息字典列表: 直接使用

        Args:
            payload: 原始 payload
            all_system: 是否全部作为 system 消息
            is_gemini: 是否是 Gemini 模型（需要特殊处理角色）
            force_role: 强制使用的角色

        Returns:
            OpenAI 格式的消息列表
        """
        messages: list[dict] = []

        if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
            # 已经是消息格式
            messages = deepcopy(payload)
        elif isinstance(payload, list):
            # 字符串列表
            for i, item in enumerate(payload):
                if force_role is not None:
                    role = force_role
                else:
                    role = "system" if all_system or i == 0 else "user"
                messages.append({"content": item, "role": role})
        else:
            # 单个字符串
            messages = [{"content": payload, "role": "system" if force_role is None else force_role}]

        # Gemini 特殊处理：不支持纯 system 角色
        if is_gemini:
            if len(messages) == 1:
                messages[0]["role"] = "user"
            else:
                for i, message in enumerate(messages):
                    if i == 0:
                        continue
                    message["role"] = "user"

        return messages

    def completion(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult:
        """
        同步 LLM 调用

        Args:
            model_path: 模型路径 (例如 openai/gpt-4o)
            payload: 消息 payload
            **kwargs: 其他参数 (temperature, max_tokens 等)

        Returns:
            LLM 调用结果
        """
        time_start = time.time()
        provider, model_name, provider_name = self.adapter_provider(model_path)

        all_system: bool = kwargs.pop("all_system", True)
        force_role: str | None = kwargs.pop("force_role", None)

        messages = self.payload_to_messages(
            payload,
            all_system=all_system,
            is_gemini=provider_name == "gemini",
            force_role=force_role
        )

        llm_time_start = time.time()
        response = provider.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )
        llm_time_end = time.time()

        # 提取响应内容
        result_content = response.choices[0].message.content
        result_reasoning = (
            response.choices[0].message.reasoning_content
            if hasattr(response.choices[0].message, "reasoning_content")
            else ""
        )

        # 提取 Token 信息
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        reasoning_token = 0

        if (
            hasattr(response.usage, "completion_tokens_details")
            and hasattr(response.usage.completion_tokens_details, "reasoning_tokens")
        ):
            reasoning_token = response.usage.completion_tokens_details.reasoning_tokens

        content_token = output_tokens - reasoning_token
        time_end = time.time()
        time_elapsed = time_end - time_start
        llm_time_elapsed = llm_time_end - llm_time_start

        return LLMResult(
            model_name=model_name,
            provider_name=provider_name,
            messages=messages,
            completion=result_content,
            reasoning=result_reasoning,
            time_start=time_start,
            time_end=time_end,
            time_elapsed=time_elapsed,
            llm_time_start=llm_time_start,
            llm_time_end=llm_time_end,
            llm_time_elapsed=llm_time_elapsed,
            retry_count=0,
            token_information=TokenUseInformation(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_token,
                content_tokens=content_token
            )
        )

    async def completion_async_t(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult:
        """
        异步 LLM 调用（使用线程包装同步调用）

        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数

        Returns:
            LLM 调用结果
        """
        result = await asyncio.to_thread(self.completion, model_path, payload, **kwargs)
        return result

    def completion_with_retry(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult | None:
        """
        带重试的同步 LLM 调用

        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数

        Returns:
            LLM 调用结果，失败返回 None
        """
        start_time = time.time()
        for i in range(self.retry_times):
            try:
                result = self.completion(model_path, payload, **kwargs)
                result.time_start = start_time
                result.time_end = time.time()
                result.time_elapsed = result.time_end - result.time_start
                return result
            except Exception as e:
                traceback.print_exc()
                continue
        return None

    async def completion_with_retry_async_t(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult | None:
        """
        带重试的异步 LLM 调用（使用线程）

        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数

        Returns:
            LLM 调用结果，失败返回 None
        """
        result = await asyncio.to_thread(
            self.completion_with_retry,
            model_path,
            payload,
            **kwargs
        )
        return result

    async def completion_async(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult:
        """
        异步 LLM 调用（原生异步）

        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数

        Returns:
            LLM 调用结果
        """
        time_start = time.time()
        provider, model_name, provider_name = self.adapter_provider(model_path)

        all_system: bool = kwargs.pop("all_system", True)
        force_role: str | None = kwargs.pop("force_role", None)

        messages = self.payload_to_messages(
            payload,
            all_system=all_system,
            is_gemini=provider_name == "gemini",
            force_role=force_role
        )

        llm_time_start = time.time()
        response = await provider.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )
        llm_time_end = time.time()

        # 提取响应内容
        result_content = response.choices[0].message.content
        result_reasoning = (
            response.choices[0].message.reasoning_content
            if hasattr(response.choices[0].message, "reasoning_content")
            else ""
        )

        # 提取 Token 信息
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        reasoning_token = 0

        if (
            hasattr(response.usage, "completion_tokens_details")
            and hasattr(response.usage.completion_tokens_details, "reasoning_tokens")
        ):
            reasoning_token = response.usage.completion_tokens_details.reasoning_tokens

        content_token = output_tokens - reasoning_token
        time_end = time.time()
        time_elapsed = time_end - time_start
        llm_time_elapsed = llm_time_end - llm_time_start

        return LLMResult(
            model_name=model_name,
            provider_name=provider_name,
            messages=messages,
            completion=result_content,
            reasoning=result_reasoning,
            time_start=time_start,
            time_end=time_end,
            time_elapsed=time_elapsed,
            llm_time_start=llm_time_start,
            llm_time_end=llm_time_end,
            llm_time_elapsed=llm_time_elapsed,
            retry_count=0,
            token_information=TokenUseInformation(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_token,
                content_tokens=content_token
            )
        )

    async def completion_with_retry_async(
        self,
        model_path: str,
        payload: list[dict] | list[str] | str,
        **kwargs
    ) -> LLMResult | None:
        """
        带重试的异步 LLM 调用（原生异步）

        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数

        Returns:
            LLM 调用结果，失败返回 None
        """
        start_time = time.time()
        for i in range(self.retry_times):
            try:
                result = await self.completion_async(model_path, payload, **kwargs)
                result.time_start = start_time
                result.time_end = time.time()
                result.time_elapsed = result.time_end - result.time_start
                return result
            except Exception as e:
                continue
        return None
