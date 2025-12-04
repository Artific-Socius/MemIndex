"""
数据模型定义 - 用于BoolQ评估实验

TODO: 这文件是得单独一个包吧，MemIndex也要用哦，只是调的提示词不一样
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class PromptStyle(str, Enum):
    """提示词风格枚举"""
    DIRECT = "direct"
    SSE = "sse"
    JSON = "json"


class EvalMode(str, Enum):
    """评估模式枚举"""
    VALIDATE = "validate"  # 验证模式：给段落+问题+答案，判断答案是否正确
    ANSWER = "answer"      # 回答模式：给段落+问题，直接回答True/False


class ReasonOrder(str, Enum):
    """推理顺序枚举"""
    REASON_FIRST = "reason-first"
    REASON_AFTER = "reason-after"


class LLMProvider(str, Enum):
    """LLM提供商枚举"""
    OPENROUTER = "openrouter"
    VOLCANO = "volcano"  # 字节火山引擎
    VERTEX_AI = "vertex_ai"  # Google Vertex AI / GenAI
    LITELLM = "litellm"  # 通用LiteLLM


@dataclass
class BoolQItem:
    """BoolQ数据集单条数据"""
    question: str
    passage: str
    answer: bool
    index: int = 0
    
    @property
    def hash(self) -> str:
        """生成数据项的哈希值"""
        return hashlib.sha256(
            f"{self.question}_{self.passage}_{self.answer}".encode("utf-8")
        ).hexdigest()


@dataclass
class LogProbInfo:
    """LogProb信息"""
    token: str
    logprob: float
    top_logprobs: Optional[list[dict[str, Any]]] = None


@dataclass
class TokenUsage:
    """Token使用量统计"""
    prompt_tokens: int = 0       # 输入token数
    completion_tokens: int = 0   # 输出token数
    total_tokens: int = 0        # 总token数
    
    # 推理token (部分模型支持，如o1, deepseek-reasoner等)
    reasoning_tokens: int = 0
    
    def __add__(self, other: TokenUsage) -> TokenUsage:
        """支持累加"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )
    
    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class CostInfo:
    """成本信息"""
    prompt_cost: float = 0.0        # 输入成本 (USD)
    completion_cost: float = 0.0    # 输出成本 (USD)
    total_cost: float = 0.0         # 总成本 (USD)
    
    # 价格信息 (每百万token)
    prompt_price_per_m: float = 0.0
    completion_price_per_m: float = 0.0
    
    def __add__(self, other: CostInfo) -> CostInfo:
        """支持累加"""
        return CostInfo(
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            total_cost=self.total_cost + other.total_cost,
            prompt_price_per_m=self.prompt_price_per_m,  # 保持原价格
            completion_price_per_m=self.completion_price_per_m,
        )
    
    def to_dict(self) -> dict[str, float]:
        return {
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
            "prompt_price_per_m": self.prompt_price_per_m,
            "completion_price_per_m": self.completion_price_per_m,
        }


@dataclass
class LLMResponse:
    """LLM响应数据模型"""
    content: str
    logprobs: Optional[list[LogProbInfo]] = None
    avg_logprobs: Optional[float] = None
    logprob_diff: Optional[float] = None
    confidence: Optional[float] = None
    latency: float = 0.0
    raw_response: Optional[Any] = None
    
    # Token使用量和成本
    token_usage: Optional[TokenUsage] = None
    cost_info: Optional[CostInfo] = None
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "content": self.content,
            "avg_logprobs": self.avg_logprobs,
            "logprob_diff": self.logprob_diff,
            "confidence": self.confidence,
            "latency": self.latency,
            "logprobs": [
                {"token": lp.token, "logprob": lp.logprob, "top_logprobs": lp.top_logprobs}
                for lp in (self.logprobs or [])
            ] if self.logprobs else None,
        }
        
        if self.token_usage:
            result["token_usage"] = self.token_usage.to_dict()
        if self.cost_info:
            result["cost_info"] = self.cost_info.to_dict()
        
        return result


@dataclass
class EvaluationResult:
    """单次评估结果"""
    status: str  # "success", "api_error", "parse_error"
    question: str
    passage: str
    expected: bool
    is_reversal: bool
    predicted: Optional[bool]
    is_correct: bool
    raw_response: str
    parsed_reason: str
    latency: float
    timestamp: str
    index: int
    item_hash: str
    
    # LogProb相关
    avg_logprobs: Optional[float] = None
    confidence: Optional[float] = None
    logprob_diff: Optional[float] = None
    logprobs: Optional[list[dict[str, Any]]] = None
    
    # Token和成本
    token_usage: Optional[dict[str, int]] = None
    cost_info: Optional[dict[str, float]] = None
    
    # 错误信息
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典，用于JSON序列化"""
        return {
            "status": self.status,
            "question": self.question,
            "passage": self.passage,
            "expected": self.expected,
            "is_reversal": self.is_reversal,
            "predicted": self.predicted,
            "is_correct": self.is_correct,
            "raw_response": self.raw_response,
            "parsed_reason": self.parsed_reason,
            "latency": self.latency,
            "timestamp": self.timestamp,
            "index": self.index,
            "hash": self.item_hash,
            "avg_logprobs": self.avg_logprobs,
            "confidence": self.confidence,
            "logprob_diff": self.logprob_diff,
            "logprobs": self.logprobs,
            "token_usage": self.token_usage,
            "cost_info": self.cost_info,
            "error": self.error,
        }


@dataclass
class ExperimentStats:
    """实验统计数据"""
    correct: int = 0
    total: int = 0
    errors: int = 0
    filter_acc: int = 0
    filter_total: int = 0
    avg_logprobs_list: list[float] = field(default_factory=list)
    avg_logprobs_list_fail: list[float] = field(default_factory=list)
    avg_logprobs_list_correct: list[float] = field(default_factory=list)
    
    # Token和成本累计
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    total_cost: CostInfo = field(default_factory=CostInfo)
    
    @property
    def accuracy(self) -> float:
        """计算准确率"""
        return self.correct / self.total if self.total > 0 else 0.0
    
    @property
    def filtered_accuracy(self) -> float:
        """计算过滤后准确率"""
        return self.filter_acc / self.filter_total if self.filter_total > 0 else 0.0
    
    @property
    def avg_logprobs_all(self) -> float:
        """所有样本的平均logprobs"""
        return sum(self.avg_logprobs_list) / len(self.avg_logprobs_list) if self.avg_logprobs_list else 0.0
    
    @property
    def avg_logprobs_correct_samples(self) -> float:
        """正确样本的平均logprobs"""
        return sum(self.avg_logprobs_list_correct) / len(self.avg_logprobs_list_correct) if self.avg_logprobs_list_correct else 0.0
    
    @property
    def avg_logprobs_fail_samples(self) -> float:
        """错误样本的平均logprobs"""
        return sum(self.avg_logprobs_list_fail) / len(self.avg_logprobs_list_fail) if self.avg_logprobs_list_fail else 0.0
    
    def update(
        self,
        is_correct: bool,
        parsed_successfully: bool,
        avg_logprobs: Optional[float] = None,
        filter_threshold: float = -1e-6,
        token_usage: Optional[TokenUsage] = None,
        cost_info: Optional[CostInfo] = None,
    ) -> None:
        """更新统计数据"""
        if parsed_successfully:
            self.total += 1
            if is_correct:
                self.correct += 1
            
            if avg_logprobs is not None:
                self.avg_logprobs_list.append(avg_logprobs)
                if is_correct:
                    self.avg_logprobs_list_correct.append(avg_logprobs)
                else:
                    self.avg_logprobs_list_fail.append(avg_logprobs)
                
                if avg_logprobs > filter_threshold:
                    self.filter_total += 1
                    if is_correct:
                        self.filter_acc += 1
        else:
            self.errors += 1
        
        # 累计token和成本
        if token_usage:
            self.total_token_usage = self.total_token_usage + token_usage
        if cost_info:
            self.total_cost = self.total_cost + cost_info
