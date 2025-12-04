"""
Google Vertex AI客户端 - 专门用于Gemini模型

使用Google GenAI SDK，支持完整的logprobs功能。
所有Gemini模型（gemini-2.5-flash, google/gemini-2.0-flash等）都走这个客户端。
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from google import genai
from google.genai import types
from loguru import logger

from llm_client.base import BaseLLMClient, VERTEX_AI_PRICING

if TYPE_CHECKING:
    from config import LLMConfig

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from utils.models_utils import LLMResponse, LogProbInfo, TokenUsage, CostInfo


class VertexAIClient(BaseLLMClient):
    """
    Google Vertex AI客户端
    专门用于Gemini模型调用，支持完整的logprobs功能
    
    支持的模型名称格式:
    - gemini-2.5-flash
    - google/gemini-2.5-flash
    - gemini-1.5-pro
    - gemini-2.0-flash-001
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_client()
        
        # 规范化模型名称
        self._model_name = self._normalize_model_name(config.model)
        logger.debug(f"VertexAI客户端初始化: 原始模型={config.model}, 规范化={self._model_name}")
    
    def _normalize_model_name(self, model: str) -> str:
        """
        规范化模型名称，移除provider前缀
        
        google/gemini-2.5-flash -> gemini-2.5-flash
        gemini-2.5-flash -> gemini-2.5-flash
        """
        if "/" in model:
            # 移除 provider 前缀 (如 google/, openrouter/google/ 等)
            model = model.split("/")[-1]
        return model
    
    def _setup_client(self) -> None:
        """初始化Google GenAI客户端 (Vertex AI模式，支持完整logprobs)"""
        self.client = genai.Client(
            api_key=self.config.api_key,
            vertexai=True,  # 使用Vertex AI模式以获得完整的logprobs支持
        )
        logger.debug(f"Google GenAI客户端初始化完成 (Vertex AI模式, API Key: {self.config.api_key[:8]}...)")
    
    def _get_generation_config(
        self,
        temperature: float,
        enable_logprobs: bool
    ) -> types.GenerateContentConfig:
        """构建生成配置"""
        config_kwargs = {
            "temperature": temperature,
        }
        
        if enable_logprobs:
            config_kwargs["response_logprobs"] = True
            config_kwargs["logprobs"] = self.top_logprobs
        
        # 安全设置 - 允许所有内容
        config_kwargs["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        ]
        
        return types.GenerateContentConfig(**config_kwargs)
    
    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        enable_logprobs: Optional[bool] = None,
    ) -> LLMResponse:
        """异步生成LLM响应"""
        import asyncio
        
        async def _do_generate():
            temp = temperature if temperature is not None else self.temperature
            logprobs_enabled = enable_logprobs if enable_logprobs is not None else self.enable_logprobs
            
            generation_config = self._get_generation_config(temp, logprobs_enabled)
            
            start_time = time.time()
            
            # Google GenAI SDK目前主要是同步的，使用run_in_executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=generation_config
                )
            )
            
            latency = time.time() - start_time
            return self._parse_response(response, latency)
        
        return await self._retry_with_backoff(_do_generate)
    
    def _parse_response(self, response: Any, latency: float) -> LLMResponse:
        """解析Vertex AI响应"""
        content = ""
        try:
            content = response.text if response.text else ""
        except Exception as e:
            logger.warning(f"获取响应文本失败: {e}")
            # 尝试从candidates获取
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        content += part.text
        
        # 解析logprobs
        logprobs_list: list[LogProbInfo] = []
        avg_logprobs: Optional[float] = None
        logprob_diff: Optional[float] = None
        confidence: Optional[float] = None
        
        try:
            if response.candidates:
                candidate = response.candidates[0]
                logprobs_result = getattr(candidate, "logprobs_result", None)
                
                if logprobs_result and logprobs_result.top_candidates:
                    chosen_logprobs = []
                    
                    for token_candidates in logprobs_result.top_candidates:
                        if token_candidates.candidates:
                            top_cand = token_candidates.candidates[0]
                            
                            # 获取token文本
                            token_text = getattr(top_cand, "token", "")
                            if not token_text:
                                token_text = getattr(top_cand, "text", "")
                            
                            log_prob = top_cand.log_probability
                            
                            # 获取top candidates
                            top_logprobs_list = [
                                {"token": getattr(c, "token", "") or getattr(c, "text", ""), 
                                 "logprob": c.log_probability}
                                for c in token_candidates.candidates
                            ]
                            
                            logprobs_list.append(LogProbInfo(
                                token=token_text,
                                logprob=log_prob,
                                top_logprobs=top_logprobs_list if top_logprobs_list else None
                            ))
                            chosen_logprobs.append(log_prob)
                    
                    # 计算平均logprobs
                    if chosen_logprobs:
                        avg_logprobs = sum(chosen_logprobs) / len(chosen_logprobs)
                        confidence = self.calculate_confidence(avg_logprobs)
                    
                    # 计算第一个token的logprob差异
                    if len(logprobs_result.top_candidates) > 0:
                        first_token_cands = logprobs_result.top_candidates[0].candidates
                        if len(first_token_cands) >= 2:
                            logprob_diff = (
                                first_token_cands[0].log_probability - 
                                first_token_cands[1].log_probability
                            )
        except Exception as e:
            # logprobs解析失败不应该影响主要功能
            logger.debug(f"LogProbs解析失败 (可忽略): {e}")
        
        # 解析token使用量
        token_usage = self._parse_token_usage(response)
        
        # 计算成本
        cost_info = self._calculate_cost(token_usage)
        
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
        usage_metadata = getattr(response, 'usage_metadata', None)
        if not usage_metadata:
            return TokenUsage()
        
        prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) or (prompt_tokens + completion_tokens)
        
        # Gemini目前不返回思考token，但预留字段
        reasoning_tokens = 0
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )
    
    def _calculate_cost(self, token_usage: TokenUsage) -> CostInfo:
        """计算Vertex AI / GenAI成本"""
        # 获取模型定价
        pricing = self._get_pricing()
        
        prompt_price = pricing["prompt"]
        completion_price = pricing["completion"]
        
        # 计算成本 (价格是每百万token)
        prompt_cost = token_usage.prompt_tokens * prompt_price / 1_000_000
        completion_cost = token_usage.completion_tokens * completion_price / 1_000_000
        total_cost = prompt_cost + completion_cost
        
        return CostInfo(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            prompt_price_per_m=prompt_price,
            completion_price_per_m=completion_price,
        )
    
    def _get_pricing(self) -> dict[str, float]:
        """获取模型定价"""
        model = self._model_name.lower()
        
        # 1. 精确匹配本地定价表
        if model in VERTEX_AI_PRICING:
            return VERTEX_AI_PRICING[model]
        
        # 2. 前缀匹配本地定价表
        for key in VERTEX_AI_PRICING:
            if key != "default" and model.startswith(key):
                return VERTEX_AI_PRICING[key]
        
        # 3. 模糊匹配
        if "2.5" in model and "flash" in model:
            return VERTEX_AI_PRICING["gemini-2.5-flash"]
        elif "2.0" in model and "flash" in model:
            return VERTEX_AI_PRICING["gemini-2.0-flash"]
        elif "1.5" in model and "flash" in model:
            return VERTEX_AI_PRICING["gemini-1.5-flash"]
        elif "1.5" in model and "pro" in model:
            return VERTEX_AI_PRICING["gemini-1.5-pro"]
        
        # 4. 使用LiteLLM获取价格 (通过openrouter/google/{model}格式)
        litellm_pricing = self._get_litellm_pricing()
        if litellm_pricing:
            return litellm_pricing
        
        # 5. 最后使用默认价格
        logger.warning(f"无法获取模型 {model} 的定价，使用默认值")
        return VERTEX_AI_PRICING["default"]
    
    def _get_litellm_pricing(self) -> Optional[dict[str, float]]:
        """
        使用LiteLLM获取模型价格
        
        尝试格式: openrouter/google/{model}
        """
        try:
            from litellm import model_cost
            
            # 尝试多种模型名称格式
            model_variants = [
                f"openrouter/google/{self._model_name}",
                f"google/{self._model_name}",
                self._model_name,
            ]
            
            for model_name in model_variants:
                model_info = model_cost.get(model_name, {})
                input_price = model_info.get("input_cost_per_token", 0)
                output_price = model_info.get("output_cost_per_token", 0)
                
                if input_price > 0 or output_price > 0:
                    # LiteLLM的价格是每token，转换为每百万token
                    return {
                        "prompt": input_price * 1_000_000,
                        "completion": output_price * 1_000_000,
                    }
            
            return None
        except Exception as e:
            logger.debug(f"LiteLLM价格查询失败: {e}")
            return None
