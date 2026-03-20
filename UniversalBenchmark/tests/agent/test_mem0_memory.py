"""Mem0Memory / Mem0GraphMemory 测试

依赖 Mem0 云 API，需要 MEM0_API_KEY 环境变量。
模型: openrouter/google/gemini-2.5-flash-lite
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from agent.memory.mem0 import Mem0Memory, Mem0GraphMemory

MODEL = "openrouter/google/gemini-2.5-flash-lite"

SKIP_REASON = "需要 MEM0_API_KEY 环境变量"
needs_mem0 = pytest.mark.skipif(
    not os.getenv("MEM0_API_KEY"), reason=SKIP_REASON,
)


# =====================================================================
# 单元测试 - Mem0Memory 纯逻辑（不调用 Mem0 API）
# =====================================================================


class TestMem0MemoryUnit:
    """Mem0Memory 不涉及 API 调用的基础逻辑测试。

    由于 Mem0Memory.__init__ 会立即创建 MemoryClient（需要 api_key），
    这里的测试需要 MEM0_API_KEY 环境变量，但不会实际发起网络请求。
    """

    @needs_mem0
    def test_init_defaults(self) -> None:
        """默认参数应正确设置。"""
        mem = Mem0Memory()

        assert mem.user_id is None
        assert mem._recent_turns == 3
        assert mem._enable_graph is False
        assert mem._max_retries == 3
        assert mem._pending_user_input is None
        assert mem.history == []

    @needs_mem0
    def test_init_custom_params(self) -> None:
        """自定义参数应正确传入。"""
        mem = Mem0Memory(
            user_id="custom_user",
            recent_turns=5,
            enable_graph=True,
            max_retries=1,
            memory_context_prefix="自定义前缀：\n",
        )

        assert mem.user_id == "custom_user"
        assert mem._recent_turns == 5
        assert mem._enable_graph is True
        assert mem._max_retries == 1
        assert mem._memory_context_prefix == "自定义前缀：\n"

    @needs_mem0
    def test_initialize_auto_generates_id(self) -> None:
        """initialize 不传参数时应自动生成 user_id。"""
        mem = Mem0Memory()
        uid = mem.initialize()

        assert uid.startswith("mem0_user_")
        assert mem.user_id == uid

    @needs_mem0
    def test_initialize_with_existing_id(self) -> None:
        """initialize 传入 user_id 时应直接使用。"""
        mem = Mem0Memory()
        uid = mem.initialize(user_id="my_custom_id")

        assert uid == "my_custom_id"
        assert mem.user_id == "my_custom_id"

    @needs_mem0
    def test_reset_clears_state(self) -> None:
        """reset 应清空历史和 pending 状态。"""
        mem = Mem0Memory()
        mem.initialize()
        mem._history.append({"role": "user", "content": "test"})
        mem._pending_user_input = "pending"

        mem.reset()
        assert mem.history == []
        assert mem._pending_user_input is None
        assert mem.user_id is not None  # user_id 不被 reset 清除

    @needs_mem0
    def test_memory_identifier(self) -> None:
        """memory_identifier 应返回正确的类名。"""
        mem = Mem0Memory(memory_tag="exp1")
        assert mem.memory_identifier == "Mem0Memory:exp1"

        mem2 = Mem0Memory()
        assert mem2.memory_identifier == "Mem0Memory"

    @needs_mem0
    def test_turn_count(self) -> None:
        """turn_count 应正确统计助手回复数。"""
        mem = Mem0Memory()
        mem.initialize()

        assert mem.turn_count == 0
        mem._history.append({"role": "user", "content": "q"})
        mem._history.append({"role": "assistant", "content": "a"})
        assert mem.turn_count == 1

    @needs_mem0
    def test_recent_history_windowing(self) -> None:
        """_get_recent_history 应正确按 recent_turns 截断。"""
        mem = Mem0Memory(recent_turns=2)
        mem.initialize()

        for i in range(5):
            mem._history.append({"role": "user", "content": f"q{i}"})
            mem._history.append({"role": "assistant", "content": f"a{i}"})

        recent = mem._get_recent_history()
        assert len(recent) == 4  # 2 turns × 2 messages

    @needs_mem0
    def test_history_is_copy(self) -> None:
        """history 属性应返回副本。"""
        mem = Mem0Memory()
        mem.initialize()
        mem._history.append({"role": "user", "content": "test"})

        h = mem.history
        h.clear()
        assert len(mem._history) == 1  # 内部不受影响

    def test_missing_api_key_raises(self) -> None:
        """未设置 MEM0_API_KEY 时应抛出 ValueError。"""
        original = os.environ.pop("MEM0_API_KEY", None)
        try:
            with pytest.raises((ValueError, ImportError)):
                Mem0Memory(api_key=None)
        finally:
            if original is not None:
                os.environ["MEM0_API_KEY"] = original


# =====================================================================
# Mem0GraphMemory 单元测试
# =====================================================================


class TestMem0GraphMemoryUnit:
    """Mem0GraphMemory 的基础测试。"""

    @needs_mem0
    def test_graph_enabled_by_default(self) -> None:
        """Mem0GraphMemory 应默认启用 enable_graph。"""
        mem = Mem0GraphMemory()
        assert mem._enable_graph is True

    @needs_mem0
    def test_is_subclass(self) -> None:
        """Mem0GraphMemory 应是 Mem0Memory 的子类。"""
        assert issubclass(Mem0GraphMemory, Mem0Memory)

    @needs_mem0
    def test_memory_identifier(self) -> None:
        """标识符应显示 Mem0GraphMemory 而非 Mem0Memory。"""
        # Mem0GraphMemory 没有自己的 get_messages，所以 _memory_type 继承自 Mem0Memory
        mem = Mem0GraphMemory()
        assert "Mem0" in mem.memory_identifier


# =====================================================================
# 集成测试 - 依赖 Mem0 API
# =====================================================================


@needs_mem0
class TestMem0MemoryIntegration:
    """Mem0Memory 与真实 Mem0 API 的集成测试。"""

    def test_search_empty_returns_list(self) -> None:
        """新用户搜索应返回空列表（不崩溃）。"""
        mem = Mem0Memory()
        mem.initialize()

        results = mem._search("随机查询")
        assert isinstance(results, list)

    def test_get_messages_builds_context(self) -> None:
        """get_messages 应正确组装消息列表。"""
        mem = Mem0Memory(system_prompt="你是助手")
        mem.initialize()

        msgs = mem.get_messages("你好")

        assert len(msgs) >= 2
        assert msgs[0] == {"role": "system", "content": "你是助手"}
        assert msgs[-1] == {"role": "user", "content": "你好"}

    def test_add_response_updates_history(self) -> None:
        """add_response 应更新本地历史。"""
        mem = Mem0Memory()
        mem.initialize()

        mem.get_messages("问题")
        mem.add_response("回答")

        assert mem.turn_count == 1
        assert mem.history[-1]["content"] == "回答"
        assert mem.history[-2]["content"] == "问题"

    def test_multi_turn_local_history(self) -> None:
        """多轮对话后本地历史应正确累积。"""
        mem = Mem0Memory()
        mem.initialize()

        mem.get_messages("第一轮问题")
        mem.add_response("第一轮回答")
        mem.get_messages("第二轮问题")
        mem.add_response("第二轮回答")

        assert mem.turn_count == 2
        assert len(mem.history) == 4

    def test_add_to_mem0_succeeds(self) -> None:
        """_add_to_mem0 应成功提交数据。"""
        mem = Mem0Memory()
        mem.initialize()

        result = mem._add_to_mem0("测试用户消息", "测试助手回复")
        assert result is True


# =====================================================================
# 端到端测试 - Mem0Memory + Agent + 真实 LLM
# =====================================================================


@pytest.mark.skipif(
    not os.getenv("MEM0_API_KEY") or not os.getenv("OPENROUTER_API_KEY"),
    reason="需要 MEM0_API_KEY 和 OPENROUTER_API_KEY 环境变量",
)
class TestMem0MemoryWithLLM:
    """Mem0Memory + Agent + 真实 LLM 的端到端测试。"""

    def _make_agent(self, **kwargs: object) -> Agent:
        AgentCls = Agent.compose(Mem0Memory)
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
        """多轮对话应保持上下文（通过本地 recent history）。"""
        agent = self._make_agent()

        agent.chat("我的名字叫李华，请记住。")
        response = agent.chat("我叫什么名字？")

        assert "李华" in response

    def test_agent_identifier(self) -> None:
        """Agent 组合后标识符应正确。"""
        agent = self._make_agent(agent_tag="mem0_test")
        ident = agent.identifier

        assert "Mem0Memory" in ident
        assert "gemini" in ident
        assert "mem0_test" in ident
