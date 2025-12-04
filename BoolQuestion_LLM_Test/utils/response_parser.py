"""
LLM响应解析器 - 支持多种输出格式
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models_utils import PromptStyle


@dataclass
class ParseResult:
    """解析结果"""
    answer: Optional[bool]
    reason: str
    success: bool
    error_message: str = ""
    
    @classmethod
    def success_result(cls, answer: bool, reason: str = "") -> ParseResult:
        """创建成功结果"""
        return cls(answer=answer, reason=reason, success=True)
    
    @classmethod
    def error_result(cls, error_message: str) -> ParseResult:
        """创建错误结果"""
        return cls(answer=None, reason="", success=False, error_message=error_message)


class ResponseParser:
    """
    LLM响应解析器
    
    支持的格式:
    - direct: 直接输出true/false
    - sse: 结构化输出 (answer: true/false, reason: ...)
    - json: JSON格式输出
    """
    
    def __init__(self, style: PromptStyle = PromptStyle.SSE):
        """
        初始化解析器
        
        Args:
            style: 解析风格
        """
        self.style = style
    
    def parse(self, text: str) -> ParseResult:
        """
        解析LLM响应
        
        Args:
            text: LLM响应文本
            
        Returns:
            ParseResult: 解析结果
        """
        if not text:
            return ParseResult.error_result("空响应")
        
        text = text.strip()
        
        if self.style == PromptStyle.DIRECT:
            return self._parse_direct(text)
        elif self.style == PromptStyle.SSE:
            return self._parse_sse(text)
        elif self.style == PromptStyle.JSON:
            return self._parse_json(text)
        else:
            return ParseResult.error_result(f"未知的解析风格: {self.style}")
    
    def _parse_direct(self, text: str) -> ParseResult:
        """解析direct风格响应"""
        lower_text = text.lower()
        
        # 检查是否只包含true或false
        if "true" in lower_text and "false" not in lower_text:
            return ParseResult.success_result(True)
        
        if "false" in lower_text and "true" not in lower_text:
            return ParseResult.success_result(False)
        
        # 正则回退 - 查找独立的true/false
        if re.search(r"\btrue\b", lower_text):
            return ParseResult.success_result(True)
        
        if re.search(r"\bfalse\b", lower_text):
            return ParseResult.success_result(False)
        
        return ParseResult.error_result(f"无法从响应中解析true/false: {text[:100]}")
    
    def _parse_sse(self, text: str) -> ParseResult:
        """解析sse风格响应"""
        # 查找answer字段
        answer_match = re.search(r"answer:\s*(true|false)", text, re.IGNORECASE)
        reason_match = re.search(r"reason:\s*(.*?)(?=\n(?:answer:|rewrite:)|$)", text, re.IGNORECASE | re.DOTALL)
        
        answer = None
        if answer_match:
            val = answer_match.group(1).lower()
            answer = val == "true"
        else:
            return ParseResult.error_result(f"无法找到answer字段: {text[:100]}")
        
        reason = reason_match.group(1).strip() if reason_match else ""
        
        return ParseResult.success_result(answer, reason)
    
    def _parse_json(self, text: str) -> ParseResult:
        """解析json风格响应"""
        try:
            json_str = text
            
            # 尝试提取markdown代码块中的JSON
            code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # 尝试找到第一个{和最后一个}
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = text[start:end + 1]
            
            data = json.loads(json_str)
            
            answer_raw = data.get("answer")
            reason = data.get("reason", "")
            
            # 处理answer值
            if isinstance(answer_raw, bool):
                return ParseResult.success_result(answer_raw, reason)
            elif isinstance(answer_raw, str):
                if answer_raw.lower() == "true":
                    return ParseResult.success_result(True, reason)
                if answer_raw.lower() == "false":
                    return ParseResult.success_result(False, reason)
            
            return ParseResult.error_result(f"无效的answer值: {answer_raw}")
            
        except json.JSONDecodeError as e:
            return ParseResult.error_result(f"JSON解析错误: {str(e)}")
        except Exception as e:
            return ParseResult.error_result(f"解析错误: {str(e)}")
    
    @staticmethod
    def parse_with_style(text: str, style: PromptStyle) -> ParseResult:
        """静态方法: 使用指定风格解析"""
        parser = ResponseParser(style)
        return parser.parse(text)
    
    @staticmethod
    def parse_auto(text: str) -> ParseResult:
        """
        自动检测格式并解析
        按顺序尝试: JSON -> SSE -> Direct
        """
        if not text:
            return ParseResult.error_result("空响应")
        
        text = text.strip()
        
        # 尝试JSON
        if "{" in text and "}" in text:
            result = ResponseParser(PromptStyle.JSON).parse(text)
            if result.success:
                return result
        
        # 尝试SSE
        if "answer:" in text.lower():
            result = ResponseParser(PromptStyle.SSE).parse(text)
            if result.success:
                return result
        
        # 回退到Direct
        return ResponseParser(PromptStyle.DIRECT).parse(text)

