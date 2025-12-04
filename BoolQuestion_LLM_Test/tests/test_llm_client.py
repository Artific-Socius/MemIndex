"""
LLM客户端单元测试

注意: 这些测试主要测试客户端结构，不实际调用API
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models_utils import LLMProvider, LLMResponse, LogProbInfo
from config import LLMConfig


class TestLLMConfig:
    """LLMConfig测试"""
    
    def test_openrouter_config(self):
        """测试OpenRouter配置"""
        with patch.dict('os.environ', {'OPEN_ROUTER_API_KEY': 'test_key'}):
            config = LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model="google/gemini-2.0-flash-001",
            )
            
            assert config.api_key == "test_key"
            assert "openrouter" in config.base_url.lower()
    
    def test_volcano_config(self):
        """测试火山引擎配置"""
        with patch.dict('os.environ', {'VOLCANO_API_KEY': 'volcano_key'}):
            config = LLMConfig(
                provider=LLMProvider.VOLCANO,
                model="deepseek-v3-250324",
            )
            
            assert config.api_key == "volcano_key"
            assert "volces" in config.base_url.lower()
    
    def test_missing_api_key(self):
        """测试缺少API Key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError):
                LLMConfig(
                    provider=LLMProvider.OPENROUTER,
                    model="test",
                )


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
    
    def test_response_with_logprobs(self):
        """测试带logprobs的响应"""
        logprobs = [
            LogProbInfo(token="true", logprob=-0.3, top_logprobs=[]),
            LogProbInfo(token=".", logprob=-0.1, top_logprobs=[]),
        ]
        
        response = LLMResponse(
            content="true.",
            logprobs=logprobs,
            avg_logprobs=-0.2,
            latency=0.5,
        )
        
        assert len(response.logprobs) == 2
        assert response.logprobs[0].token == "true"
    
    def test_to_dict(self):
        """测试转换为字典"""
        response = LLMResponse(
            content="test",
            avg_logprobs=-0.3,
            confidence=0.7,
            latency=0.5,
        )
        
        d = response.to_dict()
        
        assert "content" in d
        assert "avg_logprobs" in d
        assert "latency" in d


class TestBaseLLMClient:
    """BaseLLMClient测试"""
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        from llm_client.base import BaseLLMClient
        import math
        
        confidence = BaseLLMClient.calculate_confidence(-0.5)
        expected = math.exp(-0.5)
        
        assert abs(confidence - expected) < 0.001
    
    def test_create_client_factory(self):
        """测试客户端工厂"""
        from llm_client.base import create_llm_client
        
        with patch.dict('os.environ', {'OPEN_ROUTER_API_KEY': 'test'}):
            config = LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model="test",
            )
            
            client = create_llm_client(config)
            
            from llm_client import LiteLLMClient
            assert isinstance(client, LiteLLMClient)


class TestLiteLLMClient:
    """LiteLLMClient测试"""
    
    def test_get_model_name_openrouter(self):
        """测试OpenRouter模型名称"""
        from llm_client import LiteLLMClient
        
        with patch.dict('os.environ', {'OPEN_ROUTER_API_KEY': 'test'}):
            config = LLMConfig(
                provider=LLMProvider.OPENROUTER,
                model="google/gemini-2.0-flash-001",
            )
            
            client = LiteLLMClient(config)
            model_name = client._get_model_name()
            
            assert model_name.startswith("openrouter/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
