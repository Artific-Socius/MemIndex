"""
RunningConfig - 运行配置

提供运行时配置的加载和管理功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


class RunningConfig(BaseModel):
    """实验运行配置"""
    
    # 模型相关配置
    chat_model: str = Field(
        "openai/gpt-4o-mini",
        description="Chat Model - 用于对话的目标模型"
    )
    eval_model: str = Field(
        "volcano/deepseek-v3-250324",
        description="Eval Model - 用于评估的模型"
    )
    
    # Agent/Memory Provider 配置
    memory_provider: str = Field(
        "llm",
        description="Memory Provider - Agent类型 (llm, memecho, example, mem0, mem0_graph, memobase, letta)"
    )
    
    # 上下文窗口配置
    context_window: int = Field(
        16384,
        description="Context Window - 上下文窗口大小(tokens)"
    )
    
    # 数据集配置
    benchmark_config: str = Field(
        "./data/config/1k.json",
        description="Benchmark Config - 基准测试配置文件路径"
    )
    
    # 报告目录
    report_dir: str = Field(
        "./data/reports",
        description="Report Directory - 报告输出目录"
    )
    
    # Prompt 配置
    chat_prompt: Optional[str] = Field(
        None,
        description="Chat Prompt - Chat模型使用的提示词key (默认使用prompts.yaml中的defaults.chat)"
    )
    eval_prompt: Optional[str] = Field(
        None,
        description="Eval Prompt - Eval模型使用的提示词key (默认使用prompts.yaml中的defaults.eval)"
    )
    
    # 评估模式配置
    eval_mode: str = Field(
        "binary",
        description="Eval Mode - 评估模式 (binary: 二元评估, score: 0-1分数评估)"
    )


T = TypeVar("T", bound=BaseModel)


class RunningConfigManager:
    """
    运行配置管理器
    
    提供运行配置的加载和更新功能。
    """
    
    def __init__(self, config_path: str, default_model: Type[T] = RunningConfig):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            default_model: 默认的配置模型
        """
        self.config_path = Path(config_path)
        self.default_model = default_model
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
    
    def _load_yaml(self) -> dict:
        """加载 YAML 文件"""
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                return self.yaml.load(f) or {}
        return {}
    
    def _save_yaml(self, data: dict, model: BaseModel) -> None:
        """保存 YAML 文件"""
        commented_map = self._ensure_commented_map(data)
        self._add_comments(commented_map, model)
        
        with self.config_path.open("w", encoding="utf-8") as f:
            self.yaml.dump(commented_map, f)
    
    def _ensure_commented_map(self, data: dict) -> CommentedMap:
        """确保数据是 CommentedMap 类型"""
        commented_map = CommentedMap()
        for key, value in data.items():
            if isinstance(value, dict):
                value = self._ensure_commented_map(value)
            commented_map[key] = value
        return commented_map
    
    def _add_comments(self, commented_map: CommentedMap, model: BaseModel) -> None:
        """添加注释"""
        for field_name, field in model.model_fields.items():
            if field_name in commented_map and field.description:
                commented_map.yaml_add_eol_comment(field.description, key=field_name)
    
    def get_config(self) -> T:
        """
        获取配置
        
        Returns:
            配置对象
        """
        yaml_data = self._load_yaml()
        config = self.default_model(**yaml_data)
        
        merged_data = config.model_dump()
        need_save = False
        for key, value in merged_data.items():
            if key not in yaml_data:
                yaml_data[key] = value
                need_save = True
        
        if need_save or not self.config_path.exists():
            self._save_yaml(yaml_data, self.default_model())
        
        return config
    
    def update_config(self, updates: dict) -> None:
        """
        更新配置
        
        Args:
            updates: 更新内容
        """
        current_data = self._load_yaml()
        current_data.update(updates)
        self._save_yaml(current_data, self.default_model())


def merge_config_with_args(config: RunningConfig, args) -> RunningConfig:
    """
    将命令行参数合并到配置中
    
    Args:
        config: 配置对象
        args: 命令行参数
        
    Returns:
        合并后的配置
    """
    config_dict = config.model_dump()
    
    arg_to_config_map = {
        'model': 'chat_model',
        'eval_model': 'eval_model',
        'agent': 'memory_provider',
        'context_window': 'context_window',
        'benchmark_config': 'benchmark_config',
        'report_dir': 'report_dir',
        'chat_prompt': 'chat_prompt',
        'eval_prompt': 'eval_prompt',
        'eval_mode': 'eval_mode',
    }
    
    arg_defaults = {
        'model': '',
        'eval_model': 'volcano/deepseek-v3-250324',
        'agent': None,
        'context_window': 16384,
        'benchmark_config': './data/config/2k.json',
        'report_dir': './data/reports',
        'chat_prompt': None,
        'eval_prompt': None,
        'eval_mode': 'binary',
    }
    
    for arg_name, config_key in arg_to_config_map.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            default_value = arg_defaults.get(arg_name)
            
            if arg_value is not None and arg_value != default_value:
                config_dict[config_key] = arg_value
    
    return RunningConfig(**config_dict)
