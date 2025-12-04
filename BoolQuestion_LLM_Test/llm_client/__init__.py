"""
LLM客户端抽象层
支持多种LLM提供商: LiteLLM (OpenRouter, 火山引擎等), Google Vertex AI
"""
from llm_client.base import BaseLLMClient, create_llm_client
from llm_client.litellm_client import LiteLLMClient
from llm_client.vertex_client import VertexAIClient

__all__ = [
    "BaseLLMClient",
    "LiteLLMClient", 
    "VertexAIClient",
    "create_llm_client",
]
