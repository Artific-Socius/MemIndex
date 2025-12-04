"""
Prompt管理器 - 从YAML加载并格式化Prompt
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models_utils import PromptStyle, ReasonOrder, EvalMode
from i18n import t


class PromptManager:
    """
    Prompt管理器
    
    功能:
    - 从YAML文件加载prompt模板
    - 根据风格和配置生成prompt
    - 支持多种输出格式 (direct, sse, json)
    - 支持多种评估模式 (validate, answer)
    """
    
    DEFAULT_YAML_PATH = Path(__file__).parent / "prompts.yaml"
    
    def __init__(
        self,
        style: PromptStyle = PromptStyle.SSE,
        eval_mode: EvalMode = EvalMode.VALIDATE,
        use_reasoning: bool = False,
        reason_order: ReasonOrder = ReasonOrder.REASON_AFTER,
        yaml_path: Optional[Path] = None,
    ):
        """
        初始化Prompt管理器
        
        Args:
            style: 提示词风格 (direct, sse, json)
            eval_mode: 评估模式 (validate, answer)
            use_reasoning: 是否使用详细推理
            reason_order: 推理顺序 (reason-first, reason-after)
            yaml_path: YAML配置文件路径
        """
        self.style = style
        self.eval_mode = eval_mode
        self.use_reasoning = use_reasoning
        self.reason_order = reason_order
        self.yaml_path = yaml_path or self.DEFAULT_YAML_PATH
        
        # 加载配置
        self._config = self._load_yaml()
        logger.debug(
            t("PromptManager初始化: mode={mode}, style={style}, reasoning={reasoning}, order={order}",
              mode=eval_mode.value, style=style.value, reasoning=use_reasoning, order=reason_order.value)
        )
    
    def _load_yaml(self) -> dict[str, Any]:
        """加载YAML配置文件"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Prompt配置文件不存在: {self.yaml_path}")
        
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"{t('已加载Prompt配置')}: {self.yaml_path}")
        return config
    
    def _get_mode_config(self) -> dict[str, Any]:
        """获取当前评估模式的配置"""
        mode_config = self._config.get(self.eval_mode.value)
        if not mode_config:
            raise ValueError(f"未知的评估模式: {self.eval_mode}")
        return mode_config
    
    def create_prompt(
        self,
        question: str,
        passage: str,
        preset_answer: Optional[bool] = None,
    ) -> str:
        """
        创建完整的prompt
        
        Args:
            question: 问题
            passage: 段落
            preset_answer: 预设答案 (validate模式需要，answer模式忽略)
            
        Returns:
            str: 格式化后的prompt
        """
        mode_config = self._get_mode_config()
        
        # 基础指令
        base_instruction = mode_config["base_instruction"]
        
        # 数据部分
        if self.eval_mode == EvalMode.VALIDATE:
            data = mode_config["data_template"].format(
                question=question,
                passage=passage,
                preset_answer=preset_answer,
            )
        else:  # ANSWER模式
            data = mode_config["data_template"].format(
                question=question,
                passage=passage,
            )
        
        # 获取风格配置
        style_config = mode_config["styles"].get(self.style.value)
        if not style_config:
            raise ValueError(f"未知的风格: {self.style}")
        
        # 指令部分 (非direct风格使用)
        instructions = ""
        if style_config.get("use_instructions", True):
            instructions = style_config.get("instructions", "")
        
        # 约束部分
        constraint = self._get_constraint(style_config)
        
        # 组装prompt
        parts = [base_instruction, data]
        if instructions:
            parts.append(instructions)
        parts.append(constraint)
        
        return "\n\n".join(parts)
    
    def _get_constraint(self, style_config: dict[str, Any]) -> str:
        """获取约束文本"""
        # direct风格直接返回约束
        if "constraint" in style_config and not isinstance(style_config["constraint"], dict):
            return style_config["constraint"]
        
        # 其他风格根据reason_order选择
        order_key = self.reason_order.value.replace("-", "_")
        
        if order_key in style_config:
            constraint_config = style_config[order_key]
            constraint = constraint_config["constraint"]
            
            # 添加推理后缀
            if self.use_reasoning and "reasoning_suffix" in constraint_config:
                constraint += constraint_config["reasoning_suffix"]
            elif not self.use_reasoning and "no_reasoning_suffix" in constraint_config:
                constraint += constraint_config["no_reasoning_suffix"]
            
            return constraint
        
        raise ValueError(f"找不到约束配置: style={self.style}, order={self.reason_order}")
    
    def create_message_list(
        self,
        question: str,
        passage: str,
        preset_answer: Optional[bool] = None,
    ) -> list[dict[str, str]]:
        """
        创建消息列表格式 (用于OpenAI兼容API)
        
        Args:
            question: 问题
            passage: 段落
            preset_answer: 预设答案 (validate模式需要)
            
        Returns:
            list: 消息列表
        """
        prompt = self.create_prompt(question, passage, preset_answer)
        return [{"role": "user", "content": prompt}]
    
    @classmethod
    def from_config(cls, config) -> PromptManager:
        """从ExperimentConfig创建PromptManager"""
        return cls(
            style=config.style,
            eval_mode=getattr(config, 'eval_mode', EvalMode.VALIDATE),
            use_reasoning=config.use_reasoning,
            reason_order=config.reason_order,
        )
