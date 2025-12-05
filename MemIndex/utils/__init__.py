"""
Utils 模块 - 工具函数和数据处理

该模块提供了数据加载、数据集编译、废话生成、LLM控制器等工具功能。
"""

from .data_loader import (
    BenchmarkItem,
    BenchmarkItemExtra,
    BenchmarkSequence,
    BenchmarkDataset,
    BenchmarkDatasetFile,
    ScoreCondition,
    RefData,
    load_dataset,
    save_dataset,
    format_text,
    format_data,
    parse_content,
    ref_change,
)
from .dataset_compiler import DatasetCompiler
from .nonsense_generator import NonsenseGenerator, filler_no_response_tokens_trivia

# 使用 LiteLLM 控制器作为默认
from .litellm_controller import (
    LiteLLMController, 
    LLMResult, 
    TokenUseInformation,
    CostInformation,
    CostTracker,
    get_cost_tracker,
)
# 保留旧控制器的别名以便兼容
from .litellm_controller import LiteLLMController as LLMController

# 日志工具
from .logging import (
    logger,
    get_console,
    setup_logging,
    setup_simple_logging,
    init_logging,
)

__all__ = [
    # 数据模型
    "BenchmarkItem",
    "BenchmarkItemExtra",
    "BenchmarkSequence",
    "BenchmarkDataset",
    "BenchmarkDatasetFile",
    "ScoreCondition",
    "RefData",
    # 数据加载
    "load_dataset",
    "save_dataset",
    "format_text",
    "format_data",
    "parse_content",
    "ref_change",
    # 工具类
    "DatasetCompiler",
    "NonsenseGenerator",
    "filler_no_response_tokens_trivia",
    # LLM 控制器
    "LLMController",
    "LiteLLMController",
    "LLMResult",
    "TokenUseInformation",
    "CostInformation",
    "CostTracker",
    "get_cost_tracker",
    # 日志
    "logger",
    "get_console",
    "setup_logging",
    "setup_simple_logging",
    "init_logging",
]
