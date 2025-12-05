"""
配置管理 - BoolQ评估实验配置
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from utils.models_utils import LLMProvider, PromptStyle, ReasonOrder, EvalMode

# 加载环境变量
load_dotenv()


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    temperature: float = 0.0
    enable_logprobs: bool = True
    top_logprobs: int = 5
    max_retries: int = 5
    base_retry_delay: float = 5.0
    
    # API配置 - 从环境变量加载
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Provider特定的环境变量名映射
    # 参考: https://docs.litellm.ai/docs/providers
    PROVIDER_ENV_KEYS: dict = field(default_factory=lambda: {
        LLMProvider.OPENROUTER: ["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"],
        LLMProvider.VOLCANO: ["VOLCENGINE_API_KEY", "VOLCANO_API_KEY"],
        LLMProvider.VERTEX_AI: ["GOOGLE_API_KEY", "GOOGLE_CLOUD_API_KEY"],
        # LiteLLM通用模式下，API key由LiteLLM自己从标准环境变量读取
        # 如 OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY 等
        LLMProvider.LITELLM: [],
    })
    
    PROVIDER_BASE_URLS: dict = field(default_factory=lambda: {
        LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
        LLMProvider.VOLCANO: "https://ark.cn-beijing.volces.com/api/v3",
        LLMProvider.VERTEX_AI: None,  # 由genai库处理
        LLMProvider.LITELLM: None,  # 由LiteLLM处理
    })
    
    def __post_init__(self):
        """根据provider自动设置API配置"""
        # 尝试从环境变量加载API key
        if not self.api_key:
            env_keys = self.PROVIDER_ENV_KEYS.get(self.provider, [])
            for env_key in env_keys:
                self.api_key = os.getenv(env_key)
                if self.api_key:
                    break
        
        # 设置base_url (如果需要)
        if not self.base_url:
            self.base_url = self.PROVIDER_BASE_URLS.get(self.provider)
        
        # 只有特定provider需要强制API key
        # LITELLM通用模式下，由LiteLLM自己处理API key
        requires_api_key = self.provider in [
            LLMProvider.OPENROUTER, 
            LLMProvider.VOLCANO, 
            LLMProvider.VERTEX_AI
        ]
        
        if requires_api_key and not self.api_key:
            env_names = ", ".join(self.PROVIDER_ENV_KEYS.get(self.provider, []))
            raise ValueError(
                f"API key not found for provider: {self.provider}. "
                f"Please set one of: {env_names}"
            )


@dataclass  
class ExperimentConfig:
    """实验配置"""
    # 模型配置
    model: str = "google/gemini-2.0-flash-001"
    provider: LLMProvider = LLMProvider.OPENROUTER
    
    # 提示词配置
    style: PromptStyle = PromptStyle.SSE
    eval_mode: EvalMode = EvalMode.VALIDATE  # 评估模式：validate验证/answer回答
    reason_order: ReasonOrder = ReasonOrder.REASON_AFTER
    use_reasoning: bool = False
    
    # 数据集配置
    split: str = "validation"
    limit: int = 0  # 0表示不限制
    reversal_ratio: float = 0.3
    dirty_data_path: Optional[str] = "BoolQuestion_LLM_Test/datasets/google_boolq/dirty_data"
    
    # 输出配置
    output_dir: str = "BoolQuestion_LLM_Test/outputs"
    
    # 过滤阈值
    filter_threshold: float = -1e-6
    
    # 并发配置
    concurrency: int = 10  # 最大并发数
    
    def get_llm_config(self) -> LLMConfig:
        """获取LLM配置"""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
        )
    
    @classmethod
    def from_args(cls, args) -> ExperimentConfig:
        """从argparse参数创建配置"""
        # 根据模型名称自动判断provider
        provider = cls._detect_provider(args.model)
        
        return cls(
            model=args.model,
            provider=provider,
            style=PromptStyle(args.style),
            eval_mode=EvalMode(getattr(args, 'eval_mode', None) or 'validate'),
            reason_order=ReasonOrder(args.reason_order),
            use_reasoning=args.reasoning,
            split=args.split,
            limit=args.limit,
            reversal_ratio=args.reversal,
            concurrency=getattr(args, 'concurrency', 10),
        )
    
    @staticmethod
    def _detect_provider(model: str) -> LLMProvider:
        """
        根据模型名称检测provider
        
        遵循LiteLLM的模型命名规范:
        https://docs.litellm.ai/docs/providers
        
        规则:
        1. 火山引擎模型 (显式列表，使用自定义base_url) -> VOLCANO
        2. Gemini模型 (使用Google GenAI SDK获取完整logprobs) -> VERTEX_AI
        3. 其他所有模型 -> LITELLM (让LiteLLM根据模型名自动处理)
        
        LiteLLM会根据模型名前缀自动选择provider和API key:
        - openai/gpt-4o -> OpenAI (OPENAI_API_KEY)
        - anthropic/claude-3.5-sonnet -> Anthropic (ANTHROPIC_API_KEY)
        - openrouter/openai/gpt-4o -> OpenRouter (OPENROUTER_API_KEY)
        - azure/deployment-name -> Azure (AZURE_API_KEY)
        - ollama/llama3 -> Ollama (本地)
        - gpt-4o (无前缀) -> OpenAI (OPENAI_API_KEY)
        """
        model_lower = model.lower()
        
        # 1. 火山引擎模型 (显式列表)
        # 这些模型使用火山引擎的自定义base_url，需要特殊处理
        volcano_models = ["deepseek-v3-250324", "deepseek-v3-1-terminus", "kimi-k2-thinking-251104"]
        if model in volcano_models:
            return LLMProvider.VOLCANO
        
        # 2. Gemini模型 -> Vertex AI
        # 使用Google GenAI SDK以获取完整的logprobs支持
        # 支持格式: gemini-2.5-flash, google/gemini-2.5-flash 等
        if "gemini" in model_lower:
            return LLMProvider.VERTEX_AI
        
        # 3. 其他所有模型 -> LITELLM
        # LiteLLM会根据模型名前缀自动处理:
        # - 带前缀 (openai/, anthropic/, openrouter/等): 使用对应provider
        # - 无前缀 (gpt-4o, claude-3.5-sonnet等): 默认使用OpenAI格式
        return LLMProvider.LITELLM


# 模型名称到LiteLLM格式的映射
LITELLM_MODEL_MAPPING: dict[str, str] = {
    # OpenRouter models (需要添加openrouter/前缀)
    "google/gemini-2.0-flash-001": "openrouter/google/gemini-2.0-flash-001",
    "openai/gpt-4o": "openrouter/openai/gpt-4o",
    "anthropic/claude-3-5-sonnet": "openrouter/anthropic/claude-3-5-sonnet",
    
    # 火山引擎模型 (需要使用volcengine前缀)
    "deepseek-v3-250324": "volcengine/deepseek-v3-250324",
    "deepseek-v3-1-terminus": "volcengine/deepseek-v3-1-terminus",
    "kimi-k2-thinking-251104": "volcengine/kimi-k2-thinking-251104",
}


def get_litellm_model_name(model: str, provider: LLMProvider) -> str:
    """获取LiteLLM格式的模型名称"""
    if model in LITELLM_MODEL_MAPPING:
        return LITELLM_MODEL_MAPPING[model]
    
    # 根据provider添加前缀
    if provider == LLMProvider.OPENROUTER:
        if not model.startswith("openrouter/"):
            return f"openrouter/{model}"
    elif provider == LLMProvider.VOLCANO:
        if not model.startswith("volcengine/"):
            return f"volcengine/{model}"
    
    return model


# ============================================================
# 批量任务配置
# ============================================================

@dataclass
class TaskConfig:
    """单个任务配置"""
    name: str = ""
    model: str = "gemini-2.5-flash"
    style: str = "direct"
    eval_mode: str = "validate"
    limit: int = 0
    reasoning: bool = False
    split: str = "validation"
    reversal: float = 0.3
    reason_order: str = "reason-after"
    concurrency: int = 10
    enabled: bool = True
    
    def to_experiment_config(self) -> ExperimentConfig:
        """转换为ExperimentConfig"""
        # 自动检测provider
        provider = ExperimentConfig._detect_provider(self.model)
        return ExperimentConfig(
            model=self.model,
            provider=provider,
            style=PromptStyle(self.style),
            eval_mode=EvalMode(self.eval_mode),
            limit=self.limit,
            use_reasoning=self.reasoning,
            split=self.split,
            reversal_ratio=self.reversal,
            reason_order=ReasonOrder(self.reason_order),
            concurrency=self.concurrency,
        )


@dataclass
class BatchConfig:
    """批量任务配置"""
    lang: str = "auto"
    output_dir: str = "outputs"
    dirty_data_path: str = "datasets/google_boolq/dirty_data"
    tasks: list[TaskConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> BatchConfig:
        """从YAML文件加载配置"""
        import yaml
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # 全局设置
        global_settings = data.get('global', {})
        config.lang = global_settings.get('lang', 'auto')
        config.output_dir = global_settings.get('output_dir', 'outputs')
        config.dirty_data_path = global_settings.get('dirty_data_path', 'datasets/google_boolq/dirty_data')
        
        # 解析任务列表
        for i, task_data in enumerate(data.get('tasks', [])):
            task = TaskConfig(
                name=task_data.get('name', f'Task-{i}'),
                model=task_data.get('model', 'gemini-2.5-flash'),
                style=task_data.get('style', 'direct'),
                eval_mode=task_data.get('eval_mode', 'validate'),
                limit=task_data.get('limit', 0),
                reasoning=task_data.get('reasoning', False),
                split=task_data.get('split', 'validation'),
                reversal=task_data.get('reversal', 0.3),
                reason_order=task_data.get('reason_order', 'reason-after'),
                concurrency=task_data.get('concurrency', 10),
                enabled=task_data.get('enabled', True),
            )
            config.tasks.append(task)
        
        return config
