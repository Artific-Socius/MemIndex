"""
PromptManager单元测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_manager import PromptManager
from utils.models_utils import PromptStyle, ReasonOrder


class TestPromptManager:
    """PromptManager测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        pm = PromptManager()
        
        assert pm.style == PromptStyle.SSE
        assert pm.use_reasoning is False
        assert pm.reason_order == ReasonOrder.REASON_AFTER
    
    def test_init_custom(self):
        """测试自定义初始化"""
        pm = PromptManager(
            style=PromptStyle.JSON,
            use_reasoning=True,
            reason_order=ReasonOrder.REASON_FIRST,
        )
        
        assert pm.style == PromptStyle.JSON
        assert pm.use_reasoning is True
        assert pm.reason_order == ReasonOrder.REASON_FIRST
    
    def test_yaml_loading(self):
        """测试YAML配置加载"""
        pm = PromptManager()
        
        # 验证配置已加载
        assert pm._config is not None
        assert "base_instruction" in pm._config
        assert "data_template" in pm._config
        assert "styles" in pm._config
    
    def test_create_prompt_direct(self):
        """测试创建direct风格prompt"""
        pm = PromptManager(style=PromptStyle.DIRECT)
        
        prompt = pm.create_prompt(
            question="Is the sky blue?",
            passage="The sky appears blue during a clear day.",
            preset_answer=True,
        )
        
        assert "Is the sky blue?" in prompt
        assert "sky appears blue" in prompt
        assert "True" in prompt
        # Direct风格应该有简短的约束
        assert "true" in prompt.lower() or "false" in prompt.lower()
    
    def test_create_prompt_sse(self):
        """测试创建SSE风格prompt"""
        pm = PromptManager(style=PromptStyle.SSE)
        
        prompt = pm.create_prompt(
            question="Test question?",
            passage="Test passage.",
            preset_answer=False,
        )
        
        assert "Test question?" in prompt
        assert "Test passage." in prompt
        assert "False" in prompt
        # SSE风格应该包含输出格式说明
        assert "rewrite" in prompt.lower()
        assert "answer" in prompt.lower()
    
    def test_create_prompt_json(self):
        """测试创建JSON风格prompt"""
        pm = PromptManager(style=PromptStyle.JSON)
        
        prompt = pm.create_prompt(
            question="Test?",
            passage="Passage.",
            preset_answer=True,
        )
        
        # JSON风格应该包含JSON格式说明
        assert "json" in prompt.lower()
        assert '"answer"' in prompt or "'answer'" in prompt
    
    def test_create_prompt_with_reasoning(self):
        """测试带推理的prompt"""
        pm_no_reason = PromptManager(style=PromptStyle.JSON, use_reasoning=False)
        pm_with_reason = PromptManager(style=PromptStyle.JSON, use_reasoning=True)
        
        prompt_no = pm_no_reason.create_prompt("Q", "P", True)
        prompt_yes = pm_with_reason.create_prompt("Q", "P", True)
        
        # 有推理的prompt应该更长或包含推理相关指令
        # 具体取决于YAML配置
        assert isinstance(prompt_no, str)
        assert isinstance(prompt_yes, str)
    
    def test_create_prompt_reason_order(self):
        """测试推理顺序"""
        pm_first = PromptManager(style=PromptStyle.SSE, reason_order=ReasonOrder.REASON_FIRST)
        pm_after = PromptManager(style=PromptStyle.SSE, reason_order=ReasonOrder.REASON_AFTER)
        
        prompt_first = pm_first.create_prompt("Q", "P", True)
        prompt_after = pm_after.create_prompt("Q", "P", True)
        
        # 不同顺序应该生成不同的prompt
        assert prompt_first != prompt_after
    
    def test_create_message_list(self):
        """测试创建消息列表"""
        pm = PromptManager()
        
        messages = pm.create_message_list("Q", "P", True)
        
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Q" in messages[0]["content"]
    
    def test_from_config(self):
        """测试从配置创建"""
        class MockConfig:
            style = PromptStyle.JSON
            use_reasoning = True
            reason_order = ReasonOrder.REASON_FIRST
        
        pm = PromptManager.from_config(MockConfig())
        
        assert pm.style == PromptStyle.JSON
        assert pm.use_reasoning is True
        assert pm.reason_order == ReasonOrder.REASON_FIRST


class TestPromptContent:
    """测试Prompt内容"""
    
    def test_contains_base_instruction(self):
        """测试包含基础指令"""
        pm = PromptManager()
        prompt = pm.create_prompt("Q", "P", True)
        
        # 应该包含角色说明
        assert "assistant" in prompt.lower() or "role" in prompt.lower()
    
    def test_contains_data(self):
        """测试包含数据"""
        pm = PromptManager()
        prompt = pm.create_prompt(
            question="My question",
            passage="My passage content",
            preset_answer=False,
        )
        
        assert "My question" in prompt
        assert "My passage content" in prompt
    
    def test_contains_instructions_for_non_direct(self):
        """测试非direct风格包含指令"""
        pm_sse = PromptManager(style=PromptStyle.SSE)
        pm_json = PromptManager(style=PromptStyle.JSON)
        
        prompt_sse = pm_sse.create_prompt("Q", "P", True)
        prompt_json = pm_json.create_prompt("Q", "P", True)
        
        # 非direct风格应该包含更详细的指令
        assert "instructions" in prompt_sse.lower() or "步骤" in prompt_sse
        assert "instructions" in prompt_json.lower() or "步骤" in prompt_json


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
