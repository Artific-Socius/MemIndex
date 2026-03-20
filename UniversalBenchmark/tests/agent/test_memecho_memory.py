"""MemechoMemory 测试

依赖 Memecho 云 API，需要 MEMECHO_API_KEY 环境变量。
模型: openrouter/google/gemini-2.5-flash-lite
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from agent.memory.memecho import MemechoMemory

MODEL = "openrouter/google/gemini-2.5-flash-lite"

SKIP_REASON = "需要 MEMECHO_API_KEY 环境变量"
needs_memecho = pytest.mark.skipif(
    not os.getenv("MEMECHO_API_KEY"), reason=SKIP_REASON,
)


# =====================================================================
# 单元测试 - 不依赖 API 的基础逻辑
# =====================================================================


class TestMemechoMemoryUnit:
    """MemechoMemory 不涉及网络调用的基础测试。"""

    def test_init_defaults(self) -> None:
        """默认参数应正确设置。"""
        mem = MemechoMemory(api_key="fake_key")

        assert mem._api_base_url == "https://api.memecho.cloud"
        assert mem._api_key == "fake_key"
        assert mem.memory_lib_id is None
        assert mem._include_user_query is True
        assert mem._read_only is False
        assert mem._max_retries == 3

    def test_init_custom_params(self) -> None:
        """自定义参数应正确传入。"""
        mem = MemechoMemory(
            api_base_url="https://custom.api.com/",
            api_key="my_key",
            memory_lib_id="existing_lib_123",
            include_user_query=False,
            read_only=True,
            max_retries=5,
        )

        assert mem._api_base_url == "https://custom.api.com"  # 去尾 /
        assert mem.memory_lib_id == "existing_lib_123"
        assert mem._include_user_query is False
        assert mem._read_only is True
        assert mem._max_retries == 5

    def test_reset_clears_lib_id(self) -> None:
        """reset 应清空 memory_lib_id。"""
        mem = MemechoMemory(api_key="fake", memory_lib_id="lib_123")
        assert mem.memory_lib_id == "lib_123"

        mem.reset()
        assert mem.memory_lib_id is None

    def test_memory_identifier(self) -> None:
        """memory_identifier 应返回正确的类名。"""
        mem = MemechoMemory(api_key="fake", memory_tag="prod")
        assert mem.memory_identifier == "MemechoMemory:prod"

        mem2 = MemechoMemory(api_key="fake")
        assert mem2.memory_identifier == "MemechoMemory"

    def test_initialize_with_existing_id(self) -> None:
        """传入 memory_lib_id 时 initialize 应直接使用。"""
        mem = MemechoMemory(api_key="fake")
        result = mem.initialize(memory_lib_id="my_lib_id")

        assert result == "my_lib_id"
        assert mem.memory_lib_id == "my_lib_id"

    def test_system_prompt_preserved(self) -> None:
        """system_prompt 应在 reset 后保留。"""
        mem = MemechoMemory(
            api_key="fake",
            memory_lib_id="lib",
            system_prompt="你是助手",
        )
        mem.reset()
        assert mem.system_prompt == "你是助手"


# =====================================================================
# 集成测试 - 依赖 Memecho API
# =====================================================================


@needs_memecho
class TestMemechoMemoryIntegration:
    """MemechoMemory 与真实 Memecho API 的集成测试。"""

    def test_initialize_creates_lib(self) -> None:
        """initialize 应成功创建记忆库并返回 ID。"""
        mem = MemechoMemory()
        lib_id = mem.initialize()

        assert lib_id is not None
        assert len(lib_id) > 0
        assert mem.memory_lib_id == lib_id

    def test_get_messages_auto_initializes(self) -> None:
        """未初始化时调用 get_messages 应自动创建记忆库。"""
        mem = MemechoMemory()
        assert mem.memory_lib_id is None

        msgs = mem.get_messages("你好")

        assert mem.memory_lib_id is not None
        assert isinstance(msgs, list)
        assert len(msgs) > 0

    def test_single_turn_flow(self) -> None:
        """单轮对话: get_messages → add_response 完整流程。"""
        mem = MemechoMemory()
        mem.initialize()

        msgs = mem.get_messages("今天天气怎么样？")
        assert isinstance(msgs, list)
        assert any(m["role"] == "user" for m in msgs)

        mem.add_response("今天天气不错！")  # 不应抛异常

    def test_multi_turn_context_retained(self) -> None:
        """多轮对话后 Memecho 应保持上下文。"""
        mem = MemechoMemory()
        mem.initialize()

        mem.get_messages("我叫张三")
        mem.add_response("你好张三！")

        msgs = mem.get_messages("我叫什么？")
        all_content = " ".join(m.get("content", "") for m in msgs)
        assert "张三" in all_content


# =====================================================================
# 端到端测试 - MemechoMemory + Agent + 真实 LLM
# =====================================================================


@pytest.mark.skipif(
    not os.getenv("MEMECHO_API_KEY") or not os.getenv("OPENROUTER_API_KEY"),
    reason="需要 MEMECHO_API_KEY 和 OPENROUTER_API_KEY 环境变量",
)
class TestMemechoMemoryWithLLM:
    """MemechoMemory + Agent + 真实 LLM 的端到端测试。"""

    def _make_agent(self, **kwargs: object) -> Agent:
        AgentCls = Agent.compose(MemechoMemory)
        return AgentCls(
            model=MODEL,
            system_prompt="你是一个简洁的助手，用中文回答，每次回复不超过50字。",
            **kwargs,
        )

    def test_single_turn_chat(self) -> None:
        """单轮对话应能正常返回非空回复。"""
        agent = self._make_agent()
        response = agent.chat("1+1等于几？")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_multi_turn_chat(self) -> None:
        """多轮对话应保持上下文。"""
        agent = self._make_agent()

        agent.chat("请记住：我最喜欢的颜色是蓝色。")
        response = agent.chat("我最喜欢什么颜色？")

        assert "蓝色" in response

    def test_agent_identifier(self) -> None:
        """Agent 组合后标识符应正确。"""
        agent = self._make_agent(agent_tag="memecho_test")
        ident = agent.identifier

        assert "MemechoMemory" in ident
        assert "gemini" in ident
        assert "memecho_test" in ident
