"""
ResponseParser单元测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.response_parser import ResponseParser, ParseResult
from utils.models_utils import PromptStyle


class TestResponseParser:
    """ResponseParser测试"""
    
    # ==================== Direct风格测试 ====================
    
    def test_direct_true_simple(self):
        """测试direct风格解析true"""
        parser = ResponseParser(PromptStyle.DIRECT)
        result = parser.parse("true")
        
        assert result.success is True
        assert result.answer is True
    
    def test_direct_false_simple(self):
        """测试direct风格解析false"""
        parser = ResponseParser(PromptStyle.DIRECT)
        result = parser.parse("false")
        
        assert result.success is True
        assert result.answer is False
    
    def test_direct_true_case_insensitive(self):
        """测试大小写不敏感"""
        parser = ResponseParser(PromptStyle.DIRECT)
        
        assert parser.parse("True").answer is True
        assert parser.parse("TRUE").answer is True
        assert parser.parse("False").answer is False
        assert parser.parse("FALSE").answer is False
    
    def test_direct_with_noise(self):
        """测试带噪声的输入"""
        parser = ResponseParser(PromptStyle.DIRECT)
        
        result = parser.parse("The answer is true.")
        assert result.success is True
        assert result.answer is True
        
        result = parser.parse("I believe it's false based on the passage.")
        assert result.success is True
        assert result.answer is False
    
    def test_direct_empty(self):
        """测试空输入"""
        parser = ResponseParser(PromptStyle.DIRECT)
        result = parser.parse("")
        
        assert result.success is False
    
    def test_direct_no_answer(self):
        """测试无法解析的输入"""
        parser = ResponseParser(PromptStyle.DIRECT)
        result = parser.parse("I cannot determine the answer.")
        
        assert result.success is False
    
    # ==================== SSE风格测试 ====================
    
    def test_sse_basic(self):
        """测试SSE风格基本解析"""
        parser = ResponseParser(PromptStyle.SSE)
        
        text = """
rewrite: The sky is blue.
answer: True
reason: The passage states that the sky appears blue.
"""
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
        assert "passage states" in result.reason.lower()
    
    def test_sse_false(self):
        """测试SSE风格解析false"""
        parser = ResponseParser(PromptStyle.SSE)
        
        text = """
rewrite: The sky is green.
answer: false
reason: This contradicts the passage.
"""
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is False
    
    def test_sse_reason_first(self):
        """测试reason在前的SSE格式"""
        parser = ResponseParser(PromptStyle.SSE)
        
        text = """
rewrite: Test statement.
reason: Based on the passage analysis.
answer: True
"""
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
    
    def test_sse_missing_answer(self):
        """测试缺少answer字段"""
        parser = ResponseParser(PromptStyle.SSE)
        
        text = """
rewrite: Test statement.
reason: Some reasoning.
"""
        result = parser.parse(text)
        
        assert result.success is False
    
    # ==================== JSON风格测试 ====================
    
    def test_json_basic(self):
        """测试JSON风格基本解析"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = '{"answer": true, "reason": "Test reason"}'
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
        assert result.reason == "Test reason"
    
    def test_json_false(self):
        """测试JSON风格解析false"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = '{"answer": false, "reason": "Contradiction found"}'
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is False
    
    def test_json_string_answer(self):
        """测试字符串形式的answer"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = '{"answer": "True", "reason": "Test"}'
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
        
        text = '{"answer": "false", "reason": "Test"}'
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is False
    
    def test_json_with_markdown(self):
        """测试带markdown代码块的JSON"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = """
Here is my analysis:
```json
{
  "answer": true,
  "reason": "The passage supports this."
}
```
"""
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
    
    def test_json_with_surrounding_text(self):
        """测试JSON前后有文本"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = """
Based on my analysis, here is the result:
{"answer": false, "reason": "Not supported"}
That's my conclusion.
"""
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is False
    
    def test_json_invalid(self):
        """测试无效JSON"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = '{"answer": true, "reason": }'  # 无效JSON
        result = parser.parse(text)
        
        assert result.success is False
    
    def test_json_reason_first(self):
        """测试reason在前的JSON"""
        parser = ResponseParser(PromptStyle.JSON)
        
        text = '{"reason": "Analysis here", "answer": true}'
        result = parser.parse(text)
        
        assert result.success is True
        assert result.answer is True
    
    # ==================== 静态方法测试 ====================
    
    def test_parse_with_style(self):
        """测试静态方法parse_with_style"""
        result = ResponseParser.parse_with_style("true", PromptStyle.DIRECT)
        assert result.answer is True
        
        result = ResponseParser.parse_with_style('{"answer": false}', PromptStyle.JSON)
        assert result.answer is False
    
    def test_parse_auto(self):
        """测试自动格式检测"""
        # JSON格式
        result = ResponseParser.parse_auto('{"answer": true, "reason": "test"}')
        assert result.answer is True
        
        # SSE格式
        result = ResponseParser.parse_auto("answer: false\nreason: test")
        assert result.answer is False
        
        # Direct格式
        result = ResponseParser.parse_auto("The answer is true")
        assert result.answer is True


class TestParseResult:
    """ParseResult测试"""
    
    def test_success_result(self):
        """测试成功结果创建"""
        result = ParseResult.success_result(True, "Test reason")
        
        assert result.success is True
        assert result.answer is True
        assert result.reason == "Test reason"
        assert result.error_message == ""
    
    def test_error_result(self):
        """测试错误结果创建"""
        result = ParseResult.error_result("Parse failed")
        
        assert result.success is False
        assert result.answer is None
        assert result.error_message == "Parse failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
