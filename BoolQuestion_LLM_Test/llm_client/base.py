"""
LLM客户端基类
"""
from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from config import LLMConfig
    from utils.models_utils import LLMProvider, LLMResponse, TokenUsage, CostInfo


# Vertex AI / GenAI 定价 (每百万token, USD)
# https://cloud.google.com/vertex-ai/generative-ai/pricing
VERTEX_AI_PRICING = {
    # Gemini 2.5 Flash
    "gemini-2.5-flash": {"prompt": 0.15, "completion": 0.60},
    "gemini-2.5-flash-preview": {"prompt": 0.15, "completion": 0.60},
    # Gemini 2.0 Flash
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.0-flash-001": {"prompt": 0.10, "completion": 0.40},
    # Gemini 1.5 Flash
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash-001": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash-002": {"prompt": 0.075, "completion": 0.30},
    # Gemini 1.5 Pro
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-pro-001": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-pro-002": {"prompt": 1.25, "completion": 5.00},
    # 默认
    "default": {"prompt": 0.15, "completion": 0.60},
}


class BaseLLMClient(ABC):
    """LLM客户端抽象基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.enable_logprobs = config.enable_logprobs
        self.top_logprobs = config.top_logprobs
        self.max_retries = config.max_retries
        self.base_retry_delay = config.base_retry_delay
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        enable_logprobs: Optional[bool] = None,
    ) -> LLMResponse:
        """
        异步生成LLM响应
        
        Args:
            prompt: 用户提示词
            temperature: 温度参数 (覆盖默认值)
            enable_logprobs: 是否启用logprobs (覆盖默认值)
            
        Returns:
            LLMResponse: 包含响应内容和logprobs信息
        """
        pass
    
    def generate_sync(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        enable_logprobs: Optional[bool] = None,
    ) -> LLMResponse:
        """同步生成LLM响应 (用于兼容旧代码)"""
        return asyncio.run(self.generate(prompt, temperature=temperature, enable_logprobs=enable_logprobs))
    
    # 可重试的错误关键词
    RETRYABLE_ERRORS = [
        # 速率限制
        "429", "rate_limit", "RateLimitError", "ResourceExhausted", "Quota",
        # 临时服务错误
        "500", "502", "503", "504", "InternalServerError", "ServiceUnavailable",
        "overloaded", "temporarily unavailable",
        # 网络错误
        "timeout", "TimeoutError", "ConnectionError", "ConnectionReset",
        "SSLError", "RemoteDisconnected",
        # API临时错误
        "APIConnectionError", "APIError", "ServerError",
    ]
    
    async def _retry_with_backoff(
        self,
        coro_func,
        *args,
        **kwargs
    ) -> Any:
        """带指数退避的重试逻辑"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # 检查是否是可重试的错误
                is_retryable = any(
                    keyword.lower() in error_str or keyword.lower() in error_type.lower()
                    for keyword in self.RETRYABLE_ERRORS
                )
                
                if is_retryable and attempt < self.max_retries:
                    delay = self.base_retry_delay * (2 ** attempt)
                    from loguru import logger
                    logger.warning(
                        f"LLM请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): "
                        f"{error_type}: {str(e)[:100]}... 等待 {delay:.1f}s 后重试"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # 不可重试的错误直接抛出
                raise
        
        raise last_error
    
    @staticmethod
    def calculate_confidence(avg_logprobs: float) -> float:
        """从平均logprobs计算置信度"""
        return math.pow(math.e, avg_logprobs)
    
    def calculate_cost(self, token_usage: TokenUsage) -> CostInfo:
        """
        计算成本 - 子类可以覆盖此方法
        
        Args:
            token_usage: Token使用量
            
        Returns:
            CostInfo: 成本信息
        """
        from utils.models_utils import CostInfo
        return CostInfo()  # 默认返回空成本，子类实现具体逻辑


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """
    工厂函数: 根据配置创建对应的LLM客户端
    
    Args:
        config: LLM配置
        
    Returns:
        BaseLLMClient: LLM客户端实例
    """
    from utils.models_utils import LLMProvider
    from llm_client.litellm_client import LiteLLMClient
    from llm_client.vertex_client import VertexAIClient
    
    if config.provider == LLMProvider.VERTEX_AI:
        return VertexAIClient(config)
    else:
        # OpenRouter, Volcano 等都使用 LiteLLM
        return LiteLLMClient(config)
