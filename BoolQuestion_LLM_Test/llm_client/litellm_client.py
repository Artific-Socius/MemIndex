"""
LiteLLM客户端 - 支持OpenRouter, 火山引擎以及LiteLLM支持的所有Provider

LiteLLM支持的Provider列表: https://docs.litellm.ai/docs/providers
包括但不限于: OpenAI, Anthropic, Azure, AWS Bedrock, Google Vertex AI, 
Cohere, Together AI, Groq, Mistral, DeepSeek, Ollama, HuggingFace等

用法示例:
- OpenRouter: --model google/gemini-2.0-flash (自动检测为OpenRouter)
- OpenAI直连: --model openai/gpt-4o
- Anthropic直连: --model anthropic/claude-3-5-sonnet
- Azure: --model azure/gpt-4-deployment
- 本地Ollama: --model ollama/llama3
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional
import time

import litellm
from litellm import acompletion, completion_cost

from llm_client.base import BaseLLMClient

if TYPE_CHECKING:
    from config import LLMConfig

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from utils.models_utils import LLMResponse, LogProbInfo, LLMProvider, TokenUsage, CostInfo


class LiteLLMClient(BaseLLMClient):
    """
    LiteLLM客户端
    
    支持LiteLLM的所有Provider，包括:
    - OpenRouter (聚合多个模型)
    - 火山引擎 (字节跳动)
    - OpenAI, Anthropic, Azure, Bedrock, Vertex AI, Groq, Together AI等
    - 本地模型 (Ollama)
    
    API Key配置:
    - 对于特定Provider (OPENROUTER, VOLCANO)，从配置中读取
    - 对于通用LiteLLM模式，由LiteLLM自动从标准环境变量读取:
      * OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY等
      * 参考: https://docs.litellm.ai/docs/set_keys
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_litellm()
    
    def _setup_litellm(self) -> None:
        """配置LiteLLM"""
        # 环境变量兼容层：映射常见变体到LiteLLM标准格式
        # LiteLLM期望特定的环境变量名，这里处理常见的变体
        self._setup_env_compat()
        
        # 只有特定Provider需要手动设置API key
        # 通用LiteLLM模式下，让LiteLLM自己从标准环境变量读取
        
        if self.config.provider == LLMProvider.OPENROUTER:
            if self.config.api_key:
                os.environ["OPENROUTER_API_KEY"] = self.config.api_key
                litellm.openrouter_key = self.config.api_key
                
        elif self.config.provider == LLMProvider.VOLCANO:
            if self.config.api_key:
                os.environ["VOLCENGINE_API_KEY"] = self.config.api_key
        
        # 通用LiteLLM模式 - 不需要特殊设置，LiteLLM会自动处理
        # 用户只需要设置对应的环境变量即可，如:
        # - OPENAI_API_KEY for openai/xxx
        # - ANTHROPIC_API_KEY for anthropic/xxx
        # - AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION for azure/xxx
        # 参考: https://docs.litellm.ai/docs/set_keys
        
        # 启用详细日志 (调试用)
        # litellm.set_verbose = True
    
    def _setup_env_compat(self) -> None:
        """
        环境变量兼容层
        
        映射常见的环境变量变体到LiteLLM期望的标准格式
        这样用户不需要知道LiteLLM的精确变量名
        """
        # 环境变量映射: {用户可能使用的变体: LiteLLM期望的标准名}
        env_mappings = {
            # OpenRouter
            "OPEN_ROUTER_API_KEY": "OPENROUTER_API_KEY",
            # 火山引擎
            "VOLCANO_API_KEY": "VOLCENGINE_API_KEY",
            # Google (多种可能的名称)
            "GOOGLE_CLOUD_API_KEY": "GEMINI_API_KEY",
            # Azure (常见变体)
            "AZURE_OPENAI_API_KEY": "AZURE_API_KEY",
        }
        
        for variant, standard in env_mappings.items():
            # 如果标准变量未设置，但变体已设置，则复制
            if not os.environ.get(standard) and os.environ.get(variant):
                os.environ[standard] = os.environ[variant]
    
    def _get_model_name(self) -> str:
        """
        获取LiteLLM格式的模型名称
        
        LiteLLM的模型命名规则:
        - 大多数provider: {provider_name}/{model_name}
        - OpenRouter: openrouter/{provider}/{model}
        - 参考: https://docs.litellm.ai/docs/providers
        """
        model = self.config.model
        
        if self.config.provider == LLMProvider.OPENROUTER:
            # OpenRouter模型需要添加前缀
            if not model.startswith("openrouter/"):
                return f"openrouter/{model}"
        elif self.config.provider == LLMProvider.VOLCANO:
            # 火山引擎使用openai兼容格式，但需要指定base_url
            # 模型名称不需要特殊处理
            pass
        # LiteLLM通用模式: 用户直接提供完整的模型名称
        # 如 openai/gpt-4o, anthropic/claude-3-5-sonnet
        
        return model
    
    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        enable_logprobs: Optional[bool] = None,
    ) -> LLMResponse:
        """异步生成LLM响应"""
        
        async def _do_generate():
            temp = temperature if temperature is not None else self.temperature
            logprobs_enabled = enable_logprobs if enable_logprobs is not None else self.enable_logprobs
            
            messages = [{"role": "user", "content": prompt}]
            model_name = self._get_model_name()
            
            # 构建请求参数
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temp,
            }
            
            # 添加logprobs参数 (部分provider不支持，会被忽略)
            if logprobs_enabled:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = self.top_logprobs
            
            # Provider特定参数
            if self.config.provider == LLMProvider.VOLCANO:
                # 火山引擎: 使用openai兼容格式
                kwargs["api_base"] = self.config.base_url
                kwargs["api_key"] = self.config.api_key
                kwargs["model"] = f"openai/{self.config.model}"
                
            elif self.config.provider == LLMProvider.OPENROUTER:
                # OpenRouter: API key已在setup中配置
                pass
                
            elif self.config.provider == LLMProvider.LITELLM:
                # 通用LiteLLM模式: 
                # - 如果用户提供了api_key，传递给LiteLLM
                # - 如果用户提供了base_url，传递给LiteLLM
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                if self.config.base_url:
                    kwargs["api_base"] = self.config.base_url
            
            start_time = time.time()
            response = await acompletion(**kwargs)
            latency = time.time() - start_time
            
            return self._parse_response(response, latency, model_name)
        
        return await self._retry_with_backoff(_do_generate)
    
    def _parse_response(self, response: Any, latency: float, model_name: str) -> LLMResponse:
        """解析LiteLLM响应"""
        content = response.choices[0].message.content or ""
        
        # 解析logprobs
        logprobs_list: list[LogProbInfo] = []
        avg_logprobs: Optional[float] = None
        logprob_diff: Optional[float] = None
        confidence: Optional[float] = None
        
        choice = response.choices[0]
        if hasattr(choice, 'logprobs') and choice.logprobs is not None:
            logprobs_content = getattr(choice.logprobs, 'content', None)
            
            if logprobs_content:
                token_logprobs = []
                
                for token_info in logprobs_content:
                    token = getattr(token_info, 'token', '')
                    logprob = getattr(token_info, 'logprob', 0.0)
                    
                    # 获取top_logprobs
                    top_lps = getattr(token_info, 'top_logprobs', [])
                    top_logprobs_list = [
                        {"token": getattr(tlp, 'token', ''), "logprob": getattr(tlp, 'logprob', 0.0)}
                        for tlp in (top_lps or [])
                    ]
                    
                    logprobs_list.append(LogProbInfo(
                        token=token,
                        logprob=logprob,
                        top_logprobs=top_logprobs_list if top_logprobs_list else None
                    ))
                    token_logprobs.append(logprob)
                
                # 计算平均logprobs
                if token_logprobs:
                    avg_logprobs = sum(token_logprobs) / len(token_logprobs)
                    confidence = self.calculate_confidence(avg_logprobs)
                
                # 计算第一个token的logprob差异
                if logprobs_list and logprobs_list[0].top_logprobs and len(logprobs_list[0].top_logprobs) >= 2:
                    logprob_diff = (
                        logprobs_list[0].top_logprobs[0]["logprob"] - 
                        logprobs_list[0].top_logprobs[1]["logprob"]
                    )
        
        # 解析token使用量
        token_usage = self._parse_token_usage(response)
        
        # 计算成本
        cost_info = self._calculate_cost(response, token_usage, model_name)
        
        return LLMResponse(
            content=content,
            logprobs=logprobs_list if logprobs_list else None,
            avg_logprobs=avg_logprobs,
            logprob_diff=logprob_diff,
            confidence=confidence,
            latency=latency,
            raw_response=response,
            token_usage=token_usage,
            cost_info=cost_info,
        )
    
    def _parse_token_usage(self, response: Any) -> TokenUsage:
        """解析token使用量"""
        usage = getattr(response, 'usage', None)
        if not usage:
            return TokenUsage()
        
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        total_tokens = getattr(usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens)
        
        # 尝试获取推理token (部分模型支持)
        reasoning_tokens = 0
        completion_tokens_details = getattr(usage, 'completion_tokens_details', None)
        if completion_tokens_details:
            reasoning_tokens = getattr(completion_tokens_details, 'reasoning_tokens', 0) or 0
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )
    
    # 备用定价表 (每百万token, USD) - 用于LiteLLM无法计算成本时
    FALLBACK_PRICING = {
        # OpenAI models (via OpenRouter)
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        # Claude models
        "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
        "claude-3-opus": {"prompt": 15.00, "completion": 75.00},
        "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        # DeepSeek
        "deepseek-v3": {"prompt": 0.14, "completion": 0.28},
        "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    }
    
    def _calculate_cost(
        self, 
        response: Any, 
        token_usage: TokenUsage,
        model_name: str
    ) -> CostInfo:
        """使用LiteLLM计算成本，如果失败则使用备用定价表"""
        prompt_price = 0.0
        completion_price = 0.0
        cost = 0.0
        
        # 1. 首先尝试使用LiteLLM的completion_cost
        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            pass
        
        # 2. 尝试从LiteLLM获取模型价格
        try:
            from litellm import model_cost
            model_info = model_cost.get(model_name, {})
            prompt_price = model_info.get("input_cost_per_token", 0) * 1_000_000
            completion_price = model_info.get("output_cost_per_token", 0) * 1_000_000
        except Exception:
            pass
        
        # 3. 如果LiteLLM无法获取价格，使用备用定价表
        if prompt_price == 0 and completion_price == 0:
            pricing = self._get_fallback_pricing(model_name)
            if pricing:
                prompt_price = pricing["prompt"]
                completion_price = pricing["completion"]
        
        # 4. 计算成本
        if prompt_price > 0 and completion_price > 0:
            prompt_cost = token_usage.prompt_tokens * prompt_price / 1_000_000
            completion_cost_val = token_usage.completion_tokens * completion_price / 1_000_000
            total_cost = prompt_cost + completion_cost_val
        elif cost > 0:
            # 使用LiteLLM返回的总成本按比例分配
            if token_usage.total_tokens > 0:
                prompt_ratio = token_usage.prompt_tokens / token_usage.total_tokens
                prompt_cost = cost * prompt_ratio
                completion_cost_val = cost * (1 - prompt_ratio)
                total_cost = cost
            else:
                prompt_cost = 0.0
                completion_cost_val = 0.0
                total_cost = 0.0
        else:
            prompt_cost = 0.0
            completion_cost_val = 0.0
            total_cost = 0.0
        
        return CostInfo(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost_val,
            total_cost=total_cost,
            prompt_price_per_m=prompt_price,
            completion_price_per_m=completion_price,
        )
    
    def _get_fallback_pricing(self, model_name: str) -> dict[str, float] | None:
        """从备用定价表获取模型价格"""
        # 移除前缀 (openrouter/, openai/ 等)
        model = model_name.lower()
        if "/" in model:
            model = model.split("/")[-1]
        
        # 精确匹配
        if model in self.FALLBACK_PRICING:
            return self.FALLBACK_PRICING[model]
        
        # 前缀匹配
        for key in self.FALLBACK_PRICING:
            if model.startswith(key):
                return self.FALLBACK_PRICING[key]
        
        return None
