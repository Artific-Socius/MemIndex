"""
数据模型单元测试
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models_utils import (
    BoolQItem,
    LLMResponse,
    LogProbInfo,
    ExperimentStats,
    PromptStyle,
    ReasonOrder,
    LLMProvider,
)


class TestBoolQItem:
    """BoolQItem测试"""
    
    def test_create_item(self):
        """测试创建数据项"""
        item = BoolQItem(
            question="Is the sky blue?",
            passage="The sky appears blue during a clear day.",
            answer=True,
            index=0,
        )
        
        assert item.question == "Is the sky blue?"
        assert item.passage == "The sky appears blue during a clear day."
        assert item.answer is True
        assert item.index == 0
    
    def test_hash_generation(self):
        """测试哈希生成"""
        item1 = BoolQItem(
            question="Q1",
            passage="P1",
            answer=True,
            index=0,
        )
        
        item2 = BoolQItem(
            question="Q1",
            passage="P1",
            answer=True,
            index=1,  # 不同index
        )
        
        item3 = BoolQItem(
            question="Q2",
            passage="P1",
            answer=True,
            index=0,
        )
        
        # 相同内容应该有相同的hash
        assert item1.hash == item2.hash
        # 不同内容应该有不同的hash
        assert item1.hash != item3.hash
    
    def test_hash_consistency(self):
        """测试哈希一致性"""
        item = BoolQItem(
            question="Test question",
            passage="Test passage",
            answer=False,
            index=0,
        )
        
        # 多次调用应该返回相同结果
        hash1 = item.hash
        hash2 = item.hash
        assert hash1 == hash2


class TestLogProbInfo:
    """LogProbInfo测试"""
    
    def test_create_logprob_info(self):
        """测试创建LogProb信息"""
        info = LogProbInfo(
            token="true",
            logprob=-0.5,
            top_logprobs=[{"token": "true", "logprob": -0.5}, {"token": "false", "logprob": -1.0}]
        )
        
        assert info.token == "true"
        assert info.logprob == -0.5
        assert len(info.top_logprobs) == 2


class TestLLMResponse:
    """LLMResponse测试"""
    
    def test_create_response(self):
        """测试创建响应"""
        response = LLMResponse(
            content="true",
            avg_logprobs=-0.5,
            confidence=0.6,
            latency=1.0,
        )
        
        assert response.content == "true"
        assert response.avg_logprobs == -0.5
        assert response.confidence == 0.6
        assert response.latency == 1.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        response = LLMResponse(
            content="test",
            avg_logprobs=-0.3,
            confidence=0.7,
            logprob_diff=0.5,
            latency=0.5,
        )
        
        d = response.to_dict()
        
        assert d["content"] == "test"
        assert d["avg_logprobs"] == -0.3
        assert d["confidence"] == 0.7
        assert d["logprob_diff"] == 0.5
        assert d["latency"] == 0.5


class TestExperimentStats:
    """ExperimentStats测试"""
    
    def test_initial_state(self):
        """测试初始状态"""
        stats = ExperimentStats()
        
        assert stats.correct == 0
        assert stats.total == 0
        assert stats.errors == 0
        assert stats.accuracy == 0.0
        assert stats.filtered_accuracy == 0.0
    
    def test_update_correct(self):
        """测试更新正确统计"""
        stats = ExperimentStats()
        
        stats.update(is_correct=True, parsed_successfully=True, avg_logprobs=-0.3)
        
        assert stats.correct == 1
        assert stats.total == 1
        assert stats.accuracy == 1.0
    
    def test_update_incorrect(self):
        """测试更新错误统计"""
        stats = ExperimentStats()
        
        stats.update(is_correct=False, parsed_successfully=True, avg_logprobs=-0.5)
        
        assert stats.correct == 0
        assert stats.total == 1
        assert stats.accuracy == 0.0
    
    def test_update_parse_error(self):
        """测试解析错误统计"""
        stats = ExperimentStats()
        
        stats.update(is_correct=False, parsed_successfully=False)
        
        assert stats.errors == 1
        assert stats.total == 0
    
    def test_accuracy_calculation(self):
        """测试准确率计算"""
        stats = ExperimentStats()
        
        # 添加3个正确, 2个错误
        for _ in range(3):
            stats.update(is_correct=True, parsed_successfully=True)
        for _ in range(2):
            stats.update(is_correct=False, parsed_successfully=True)
        
        assert stats.total == 5
        assert stats.correct == 3
        assert stats.accuracy == 0.6
    
    def test_filter_accuracy(self):
        """测试过滤准确率"""
        stats = ExperimentStats()
        
        # 添加高logprobs的结果
        stats.update(is_correct=True, parsed_successfully=True, avg_logprobs=-1e-7, filter_threshold=-1e-6)
        stats.update(is_correct=False, parsed_successfully=True, avg_logprobs=-1e-7, filter_threshold=-1e-6)
        
        # 添加低logprobs的结果
        stats.update(is_correct=True, parsed_successfully=True, avg_logprobs=-1.0, filter_threshold=-1e-6)
        
        assert stats.filter_total == 2
        assert stats.filter_acc == 1
        assert stats.filtered_accuracy == 0.5


class TestEnums:
    """枚举测试"""
    
    def test_prompt_style(self):
        """测试PromptStyle枚举"""
        assert PromptStyle.DIRECT.value == "direct"
        assert PromptStyle.SSE.value == "sse"
        assert PromptStyle.JSON.value == "json"
    
    def test_reason_order(self):
        """测试ReasonOrder枚举"""
        assert ReasonOrder.REASON_FIRST.value == "reason-first"
        assert ReasonOrder.REASON_AFTER.value == "reason-after"
    
    def test_llm_provider(self):
        """测试LLMProvider枚举"""
        assert LLMProvider.OPENROUTER.value == "openrouter"
        assert LLMProvider.VOLCANO.value == "volcano"
        assert LLMProvider.VERTEX_AI.value == "vertex_ai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
