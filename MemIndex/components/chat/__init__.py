"""
Chat 模块 - 负责与 LLM 进行对话交互

该模块提供了与 LLM 进行对话的基础抽象和实现。
"""

from .base_chat import BaseChat
from .llm_chat import LLMChat

__all__ = ["BaseChat", "LLMChat"]



