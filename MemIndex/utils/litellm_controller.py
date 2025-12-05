"""
LiteLLMController - 基于 LiteLLM 的 LLM 控制器

使用 LiteLLM 提供统一的多 Provider LLM 调用接口。
LiteLLM 自动支持多种 Provider，只需在 .env 中配置对应的 API Key。

核心功能:
    1. 统一的 LLM 调用接口（支持 30+ Provider）
    2. 自动重试机制（含网络错误特殊处理）
    3. Token 用量和费用追踪
    4. 同步/异步调用支持

使用方式:
    模型名称格式: provider/model_name
    例如: openai/gpt-4o, anthropic/claude-3-opus, openrouter/google/gemini-2.5-flash

支持的 Provider:
    - OpenAI (openai/gpt-4o)
    - Anthropic (anthropic/claude-3-opus)
    - Google (gemini/gemini-pro)
    - OpenRouter (openrouter/...)
    - Azure (azure/...)
    - AWS Bedrock (bedrock/...)
    - 以及更多...

配置:
    在 .env 文件中设置对应的 API Key:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - OPEN_ROUTER_API_KEY
    等等
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING

import litellm
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

# 禁用 LiteLLM 的默认日志输出
litellm.suppress_debug_info = True


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


class CostInformation(BaseModel):
    """
    费用信息
    
    记录单次 LLM 调用的费用。
    
    Attributes:
        input_cost: 输入费用
        output_cost: 输出费用
        total_cost: 总费用
        currency: 货币单位（默认 USD）
    """
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"


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
        cost_information: 费用信息
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
    cost_information: CostInformation = CostInformation()


class CostTracker:
    """
    费用追踪器
    
    追踪所有 LLM 调用的累计费用和 Token 使用情况。
    支持按模型分组统计。
    
    使用方式:
        tracker = get_cost_tracker()
        print(tracker.total_cost)
        print(tracker.get_summary())
    """
    
    def __init__(self):
        """初始化费用追踪器"""
        self.reset()
    
    def reset(self):
        """
        重置所有统计数据
        
        在开始新的测试任务前调用。
        """
        self.total_input_tokens = 0     # 累计输入 Token
        self.total_output_tokens = 0    # 累计输出 Token
        self.total_cost = 0.0           # 累计费用
        self.call_count = 0             # 调用次数
        self.model_stats: dict[str, dict] = {}  # 按模型分组的统计
    
    def add_usage(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int, 
        cost: float
    ):
        """
        添加一次调用的使用记录
        
        Args:
            model: 模型名称
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            cost: 费用
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1
        
        # 按模型分组统计
        if model not in self.model_stats:
            self.model_stats[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "calls": 0,
            }
        
        self.model_stats[model]["input_tokens"] += input_tokens
        self.model_stats[model]["output_tokens"] += output_tokens
        self.model_stats[model]["cost"] += cost
        self.model_stats[model]["calls"] += 1
    
    @property
    def total_tokens(self) -> int:
        """累计总 Token 数"""
        return self.total_input_tokens + self.total_output_tokens
    
    def get_summary(self) -> dict:
        """
        获取使用情况摘要
        
        Returns:
            包含所有统计数据的字典
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "model_stats": self.model_stats,
        }


# 全局费用追踪器（单例模式）
_global_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """
    获取全局费用追踪器实例
    
    Returns:
        全局费用追踪器
    """
    return _global_cost_tracker


class LiteLLMController:
    """
    基于 LiteLLM 的 LLM 控制器
    
    使用 LiteLLM 提供统一的多 Provider LLM 调用接口。
    支持的 Provider 包括: OpenAI, Anthropic, Google, Azure, AWS Bedrock 等。
    
    使用方式:
        controller = LiteLLMController()
        result = await controller.completion_with_retry_async(
            "openrouter/google/gemini-2.5-flash",
            "Hello, how are you?"
        )
        print(result.completion)
    
    Attributes:
        env_file: 环境变量文件路径
        retry_times: 默认重试次数
        track_cost: 是否追踪费用
        cost_tracker: 费用追踪器
    """
    
    def __init__(
        self, 
        env_file: str = ".env",
        retry_times: int = 3,
        track_cost: bool = True,
    ):
        """
        初始化 LiteLLM 控制器
        
        Args:
            env_file: 环境变量文件路径（包含 API Keys）
            retry_times: 默认重试次数
            track_cost: 是否追踪费用
        """
        self.env_file = env_file
        self.retry_times = retry_times
        self.track_cost = track_cost
        self._initialized = False
        self.cost_tracker = get_cost_tracker()
    
    async def _init_provider(self) -> None:
        """
        初始化 Provider（加载环境变量）
        
        首次调用时自动执行，后续调用跳过。
        """
        if self._initialized:
            return
        
        # 加载环境变量
        load_dotenv(self.env_file)
        self._initialized = True
        logger.debug("LiteLLM Controller initialized")
    
    @staticmethod
    def _parse_model_path(model_path: str) -> tuple[str, str]:
        """
        解析模型路径
        
        从 "provider/model_name" 格式中提取 provider 和完整模型路径。
        
        Args:
            model_path: 模型路径 (如 openrouter/google/gemini-2.5-flash)
            
        Returns:
            (provider_name, full_model_path)
        """
        parts = model_path.split("/", 1)
        if len(parts) == 2:
            return parts[0], model_path  # LiteLLM 使用完整路径
        return "unknown", model_path
    
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
            is_gemini: 是否是 Gemini 模型（需要特殊处理）
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
        
        # Gemini 特殊处理：不支持 system 角色
        if is_gemini:
            if len(messages) == 1:
                messages[0]["role"] = "user"
            else:
                for i, message in enumerate(messages):
                    if i == 0:
                        continue
                    message["role"] = "user"
        
        return messages
    
    def _extract_token_info(self, response) -> TokenUseInformation:
        """
        从 LLM 响应中提取 Token 信息
        
        Args:
            response: LiteLLM 的响应对象
            
        Returns:
            Token 使用信息
        """
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
            
            # 尝试获取 reasoning tokens（推理模型特有）
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if details and hasattr(details, 'reasoning_tokens'):
                    reasoning_tokens = details.reasoning_tokens or 0
        
        content_tokens = output_tokens - reasoning_tokens
        
        return TokenUseInformation(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            content_tokens=content_tokens,
        )
    
    def _extract_cost_info(self, response, model_path: str) -> CostInformation:
        """
        从 LLM 响应中提取费用信息
        
        使用 LiteLLM 的费用计算功能。
        
        Args:
            response: LiteLLM 的响应对象
            model_path: 模型路径
            
        Returns:
            费用信息
        """
        try:
            # 使用 LiteLLM 的费用计算
            cost = litellm.completion_cost(completion_response=response)
            return CostInformation(
                input_cost=0.0,  # LiteLLM 只返回总费用
                output_cost=0.0,
                total_cost=cost or 0.0,
                currency="USD",
            )
        except Exception:
            # 无法计算费用时返回空
            return CostInformation()
    
    def _extract_content(self, response) -> tuple[str, str]:
        """
        从 LLM 响应中提取内容和推理
        
        Args:
            response: LiteLLM 的响应对象
            
        Returns:
            (content, reasoning)
        """
        content = ""
        reasoning = ""
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content or ""
            
            # 尝试获取推理内容（推理模型特有）
            if hasattr(message, 'reasoning_content'):
                reasoning = message.reasoning_content or ""
        
        return content, reasoning
    
    def _track_usage(self, model_path: str, token_info: TokenUseInformation, cost_info: CostInformation):
        """
        记录使用情况到费用追踪器
        
        Args:
            model_path: 模型路径
            token_info: Token 信息
            cost_info: 费用信息
        """
        if self.track_cost:
            self.cost_tracker.add_usage(
                model=model_path,
                input_tokens=token_info.input_tokens,
                output_tokens=token_info.output_tokens,
                cost=cost_info.total_cost,
            )
    
    async def completion_async(
        self, 
        model_path: str, 
        payload: list[dict] | list[str] | str, 
        **kwargs
    ) -> LLMResult:
        """
        异步 LLM 调用（不带重试）
        
        直接调用 LiteLLM 的异步接口。
        
        Args:
            model_path: 模型路径 (例如: openai/gpt-4o)
            payload: 消息 payload
            **kwargs: 其他参数 (temperature, max_tokens 等)
            
        Returns:
            LLM 调用结果
        """
        time_start = time.time()
        provider_name, model_name = self._parse_model_path(model_path)
        
        all_system: bool = kwargs.pop("all_system", True)
        force_role: str | None = kwargs.pop("force_role", None)
        is_gemini = "gemini" in model_path.lower()
        
        messages = self.payload_to_messages(
            payload, 
            all_system=all_system, 
            is_gemini=is_gemini, 
            force_role=force_role
        )
        
        llm_time_start = time.time()
        response = await litellm.acompletion(
            model=model_path,
            messages=messages,
            **kwargs
        )
        llm_time_end = time.time()
        
        content, reasoning = self._extract_content(response)
        token_info = self._extract_token_info(response)
        cost_info = self._extract_cost_info(response, model_path)
        
        # 追踪使用情况
        self._track_usage(model_path, token_info, cost_info)
        
        time_end = time.time()
        
        return LLMResult(
            model_name=model_name,
            provider_name=provider_name,
            messages=messages,
            completion=content,
            reasoning=reasoning,
            time_start=time_start,
            time_end=time_end,
            time_elapsed=time_end - time_start,
            llm_time_start=llm_time_start,
            llm_time_end=llm_time_end,
            llm_time_elapsed=llm_time_end - llm_time_start,
            retry_count=0,
            token_information=token_info,
            cost_information=cost_info,
        )
    
    def completion(
        self, 
        model_path: str, 
        payload: list[dict] | list[str] | str, 
        **kwargs
    ) -> LLMResult:
        """
        同步 LLM 调用（不带重试）
        
        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数
            
        Returns:
            LLM 调用结果
        """
        time_start = time.time()
        provider_name, model_name = self._parse_model_path(model_path)
        
        all_system: bool = kwargs.pop("all_system", True)
        force_role: str | None = kwargs.pop("force_role", None)
        is_gemini = "gemini" in model_path.lower()
        
        messages = self.payload_to_messages(
            payload, 
            all_system=all_system, 
            is_gemini=is_gemini, 
            force_role=force_role
        )
        
        llm_time_start = time.time()
        response = litellm.completion(
            model=model_path,
            messages=messages,
            **kwargs
        )
        llm_time_end = time.time()
        
        content, reasoning = self._extract_content(response)
        token_info = self._extract_token_info(response)
        cost_info = self._extract_cost_info(response, model_path)
        
        # 追踪使用情况
        self._track_usage(model_path, token_info, cost_info)
        
        time_end = time.time()
        
        return LLMResult(
            model_name=model_name,
            provider_name=provider_name,
            messages=messages,
            completion=content,
            reasoning=reasoning,
            time_start=time_start,
            time_end=time_end,
            time_elapsed=time_end - time_start,
            llm_time_start=llm_time_start,
            llm_time_end=llm_time_end,
            llm_time_elapsed=llm_time_end - llm_time_start,
            retry_count=0,
            token_information=token_info,
            cost_information=cost_info,
        )
    
    async def completion_async_t(
        self, 
        model_path: str, 
        payload: list[dict] | list[str] | str, 
        **kwargs
    ) -> LLMResult:
        """
        异步 LLM 调用（使用线程包装同步调用）
        
        当需要在异步环境中使用同步代码时使用。
        
        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数
            
        Returns:
            LLM 调用结果
        """
        result = await asyncio.to_thread(self.completion, model_path, payload, **kwargs)
        return result
    
    @staticmethod
    def _is_network_error(error: Exception) -> bool:
        """
        判断是否为网络相关错误
        
        网络错误会获得更多的重试机会。
        
        Args:
            error: 异常对象
            
        Returns:
            是否为网络错误
        """
        error_str = str(error).lower()
        network_keywords = [
            'ssl', 'eof', 'connection', 'timeout', 'reset', 
            'refused', 'network', 'socket', 'broken pipe',
            'unexpected_eof', 'protocol', 'certificate',
        ]
        return any(keyword in error_str for keyword in network_keywords)
    
    def completion_with_retry(
        self, 
        model_path: str, 
        payload: list[dict] | list[str] | str, 
        **kwargs
    ) -> LLMResult | None:
        """
        带重试的同步 LLM 调用
        
        特点:
            - 普通错误重试 retry_times 次
            - 网络错误最多重试 10 次
            - 网络错误会递增延迟
        
        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数
            
        Returns:
            LLM 调用结果，失败返回 None
        """
        start_time = time.time()
        last_error = None
        
        # 网络错误时使用更多重试次数
        max_retries = self.retry_times
        network_retry_delay = 1.0  # 网络错误重试延迟（秒）
        
        had_error = False  # 标记是否发生过错误
        
        for i in range(max_retries):
            try:
                result = self.completion(model_path, payload, **kwargs)
                result.time_start = start_time
                result.time_end = time.time()
                result.time_elapsed = result.time_end - result.time_start
                result.retry_count = i
                
                # 如果之前发生过错误，现在成功了，打印恢复日志
                if had_error:
                    logger.success(f"✅ LLM call recovered after {i} retry(ies)")
                
                return result
            except Exception as e:
                had_error = True
                last_error = e
                is_network = self._is_network_error(e)
                
                # 网络错误时增加重试次数
                if is_network and max_retries == self.retry_times:
                    max_retries = 10  # 网络错误最多重试 10 次
                    logger.warning(f"Network error detected, increasing max retries to {max_retries}")
                
                logger.warning(f"LLM call failed (attempt {i + 1}/{max_retries}): {e}")
                
                # 网络错误时添加延迟
                if is_network and i < max_retries - 1:
                    delay = network_retry_delay * (i + 1)  # 递增延迟
                    logger.info(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                
                continue
        
        logger.error(f"LLM call failed after {max_retries} attempts: {last_error}")
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
    
    async def completion_with_retry_async(
        self, 
        model_path: str, 
        payload: list[dict] | list[str] | str, 
        **kwargs
    ) -> LLMResult | None:
        """
        带重试的异步 LLM 调用（原生异步）
        
        这是推荐的调用方式，使用原生异步获得最佳性能。
        
        特点:
            - 普通错误重试 retry_times 次
            - 网络错误最多重试 10 次
            - 网络错误会递增延迟
        
        Args:
            model_path: 模型路径
            payload: 消息 payload
            **kwargs: 其他参数
            
        Returns:
            LLM 调用结果，失败返回 None
        """
        start_time = time.time()
        last_error = None
        had_error = False  # 标记是否发生过错误
        
        # 网络错误时使用更多重试次数
        max_retries = self.retry_times
        network_retry_delay = 1.0  # 网络错误重试延迟（秒）
        
        for i in range(max_retries):
            try:
                result = await self.completion_async(model_path, payload, **kwargs)
                result.time_start = start_time
                result.time_end = time.time()
                result.time_elapsed = result.time_end - result.time_start
                result.retry_count = i
                
                # 如果之前发生过错误，现在成功了，打印恢复日志
                if had_error:
                    logger.success(f"✅ LLM call recovered after {i} retry(ies)")
                
                return result
            except Exception as e:
                had_error = True
                last_error = e
                is_network = self._is_network_error(e)
                
                # 网络错误时增加重试次数
                if is_network and max_retries == self.retry_times:
                    max_retries = 10  # 网络错误最多重试 10 次
                    logger.warning(f"Network error detected, increasing max retries to {max_retries}")
                
                logger.warning(f"LLM call failed (attempt {i + 1}/{max_retries}): {e}")
                
                # 网络错误时添加延迟
                if is_network and i < max_retries - 1:
                    delay = network_retry_delay * (i + 1)  # 递增延迟
                    logger.info(f"Waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)
                
                continue
        
        logger.error(f"LLM call failed after {max_retries} attempts: {last_error}")
        return None


# 为了兼容性，创建一个别名
LLMController = LiteLLMController
