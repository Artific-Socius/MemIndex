"""
Config - 系统配置

提供系统配置的加载和管理功能。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML, CommentToken
from ruamel.yaml.comments import CommentedMap


class LLMProvider(BaseModel):
    """LLM 提供商配置"""
    name: str = Field("openai", description="LLM Provider")
    base_url: str = Field("https://api.openai.com/v1", description="Base URL")
    api_key: str = Field("", description="API Key")
    api_key_env: str = Field("OPENAI_API_KEY", description="API Key environment variable name")


class LLMConfig(BaseModel):
    """LLM 配置"""
    llm_retry_times: int = Field(3, description="LLM Retry Times")


class Config(BaseModel):
    """系统配置"""
    base_path: str = Field("./", description="Base Path")
    env_file: str = Field(".env", description="Environment file")
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM Config")
    providers: list[LLMProvider] = [
        LLMProvider(
            name="openai", 
            base_url="https://api.openai.com/v1", 
            api_key="", 
            api_key_env="OPENAI_API_KEY"
        ),
    ]


T = TypeVar("T", bound=BaseModel)


class ConfigManager(Generic[T]):
    """
    配置管理器
    
    提供配置的加载、保存和更新功能。
    """
    
    def __init__(self, config_path: str, default_model: Type[T] = Config):
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
    
    def _save_yaml(self, data: dict, model: BaseModel, depth_limit: int = 1) -> None:
        """保存 YAML 文件"""
        commented_map = self._ensure_commented_map(data)
        self._add_comments(commented_map, commented_map, model)
        self._add_newlines(commented_map, depth=0, depth_limit=depth_limit)
        
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
    
    def _add_comments(
        self, 
        commented_map: CommentedMap, 
        commented_map_root: CommentedMap, 
        model: type[BaseModel] | BaseModel, 
        depth: int = 0
    ) -> None:
        """添加注释"""
        # 获取模型类（支持类型或实例）
        model_class = model if isinstance(model, type) else model.__class__
        
        for field_name, field in model_class.model_fields.items():
            if field_name in commented_map and model_class != field.annotation:
                existing_comment = commented_map.ca.comment if commented_map.ca else None
                existing_comment_result = []
                
                if existing_comment:
                    queued = [*existing_comment]
                    while len(queued) > 0:
                        item = queued.pop(0)
                        if isinstance(item, CommentToken):
                            existing_comment_result.append(item)
                        elif isinstance(item, list):
                            queued = [*item] + queued
                
                if field.description:
                    current_comment = (
                        re.sub(r"^#\x20", "", existing_comment_result[0].value.strip()) 
                        if existing_comment_result and existing_comment_result[0] 
                        else ""
                    )
                    if current_comment != field.description or True:
                        commented_map.yaml_add_eol_comment(field.description, key=field_name)
            
            if hasattr(field.annotation, "model_fields") and model_class != field.annotation:
                self._add_comments(
                    commented_map[field_name], 
                    commented_map_root, 
                    field.annotation, 
                    depth + 1
                )
    
    def _add_newlines(
        self, 
        commented_map: CommentedMap, 
        depth: int, 
        depth_limit: int
    ) -> None:
        """添加换行"""
        for key in list(commented_map.keys()):
            if isinstance(commented_map[key], CommentedMap):
                self._add_newlines(commented_map[key], depth + 1, depth_limit)
            
            if depth < depth_limit:
                commented_map.yaml_set_comment_before_after_key(key, before="\n")
    
    def get_config(self) -> T:
        """
        获取配置
        
        Returns:
            配置对象
        """
        yaml_data = self._load_yaml()
        
        try:
            config = self.default_model(**yaml_data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")
        
        merged_data = config.model_dump()
        for key, value in merged_data.items():
            if key not in yaml_data:
                yaml_data[key] = value
        
        queue = [(k, v, k) for k, v in merged_data.items()]
        while len(queue) > 0:
            key, value, path = queue.pop(0)
            if isinstance(value, BaseModel):
                for k, v in value.model_dump().items():
                    queue.append((k, v, f"{path}.{k}"))
            else:
                keys = path.split(".")
                current = yaml_data
                for k in keys[:-1]:
                    current = current[k]
                if keys[-1] in current:
                    current[keys[-1]] = value
        
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



