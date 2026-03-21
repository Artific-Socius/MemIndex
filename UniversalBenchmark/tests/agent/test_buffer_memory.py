"""BufferMemory 测试

纯本地内存，无外部依赖。同时测试与 Agent + LLM 组合后的端到端对话。
模型: openrouter/google/gemini-2.5-flash-lite
"""

import os
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from agent.memory.buffer import BufferMemory

MODEL = "openrouter/google/gemini-2.5-flash-lite"


# =====================================================================
# 单元测试 - 纯 Memory 逻辑，不调用 LLM
# =====================================================================


class TestBufferMemoryUnit:
    """BufferMemory 的纯逻辑测试，不涉及任何外部调用。"""

    def test_first_message_returns_system_and_user(self) -> None:
        """首条消息应返回 [system, user]。"""
        mem = BufferMemory(system_prompt="你是助手")
        msgs = mem.get_messages("你好")

        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "你是助手"}
        assert msgs[1] == {"role": "user", "content": "你好"}

    def test_no_system_prompt(self) -> None:
        """未设置 system_prompt 时不应有 system 消息。"""
        mem = BufferMemory()
        msgs = mem.get_messages("你好")

        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_history_accumulates(self) -> None:
        """多轮对话后历史应正确累积。"""
        mem = BufferMemory(system_prompt="sys")

        mem.get_messages("第一轮")
        mem.add_response("回复一")
        msgs = mem.get_messages("第二轮")

        # get_messages 先 append user 到 history, 然后返回 system + history
        # history: [user:第一轮, assistant:回复一, user:第二轮]
        # msgs: [system:sys, user:第一轮, assistant:回复一, user:第二轮]
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_history_accumulates_correctly(self) -> None:
        """验证完整的两轮对话后消息列表的正确性。"""
        mem = BufferMemory(system_prompt="sys")

        # 第一轮
        msgs1 = mem.get_messages("问题1")
        assert len(msgs1) == 2  # system + user
        mem.add_response("回答1")

        # 第二轮
        msgs2 = mem.get_messages("问题2")
        # history: [user:问题1, assistant:回答1, user:问题2]
        # 返回: [system:sys] + history = 4条
        assert len(msgs2) == 4
        assert msgs2[0]["role"] == "system"
        assert msgs2[1]["content"] == "问题1"
        assert msgs2[2]["content"] == "回答1"
        assert msgs2[3]["content"] == "问题2"

    def test_max_turns_windowing(self) -> None:
        """max_turns 应正确截断历史。"""
        mem = BufferMemory(system_prompt="sys", max_turns=2)

        for i in range(5):
            mem.get_messages(f"问题{i}")
            mem.add_response(f"回答{i}")

        msgs = mem.get_messages("最新问题")
        # max_turns=2, 所以 history 切片取最后 4 条 + 刚 append 的 user
        # history 共 11 条 (5 user + 5 assistant + 1 最新 user)
        # 切片 [-4:] = [assistant:回答3, user:问题4, assistant:回答4, 最新user]
        # 等等, 实际上 get_messages 先 append "最新问题" 到 history
        # history 有 11 条, 取 [-4:] = 最后 4 条
        non_system = [m for m in msgs if m["role"] != "system"]
        assert len(non_system) <= 4  # max_turns=2 → 最多 4 条非 system 消息

    def test_turn_count(self) -> None:
        """turn_count 应统计已完成的对话轮数。"""
        mem = BufferMemory()

        assert mem.turn_count == 0
        mem.get_messages("问题")
        assert mem.turn_count == 0  # 还没有 assistant 回复
        mem.add_response("回答")
        assert mem.turn_count == 1

    def test_reset(self) -> None:
        """reset 应清空历史但保留配置。"""
        mem = BufferMemory(system_prompt="sys", max_turns=3)

        mem.get_messages("问题")
        mem.add_response("回答")
        assert mem.turn_count == 1

        mem.reset()
        assert mem.turn_count == 0
        assert len(mem.history) == 0
        assert mem.system_prompt == "sys"  # 配置保留

    def test_memory_identifier(self) -> None:
        """memory_identifier 应返回正确的类名。"""
        mem = BufferMemory(memory_tag="v1")
        assert mem.memory_identifier == "BufferMemory:v1"

        mem2 = BufferMemory()
        assert mem2.memory_identifier == "BufferMemory"

    def test_history_is_copy(self) -> None:
        """history 属性应返回副本，修改副本不影响内部状态。"""
        mem = BufferMemory()
        mem.get_messages("问题")
        mem.add_response("回答")

        h = mem.history
        h.clear()
        assert mem.turn_count == 1  # 内部不受影响


# =====================================================================
# 集成测试 - BufferMemory + Agent + 真实 LLM
# =====================================================================


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="需要 OPENROUTER_API_KEY 环境变量",
)
class TestBufferMemoryWithLLM:
    """BufferMemory 与真实 LLM 的端到端测试。"""

    def _make_agent(self, **kwargs: Any) -> Agent:
        AgentCls = Agent.compose(BufferMemory)
        return AgentCls(
            model=MODEL,
            system_prompt="你是一个简洁的助手，用中文回答，每次回复不超过50字。",
            **kwargs,
        )

    def test_single_turn(self) -> None:
        """单轮对话应能正常返回非空回复。"""
        agent = self._make_agent()
        response = agent.chat("1+1等于几？")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_multi_turn_context(self) -> None:
        """多轮对话应保持上下文。"""
        agent = self._make_agent()

        agent.chat("我的名字叫小明，请记住。")
        response = agent.chat("我叫什么名字？")

        assert "小明" in response

    def test_agent_identifier(self) -> None:
        """Agent 组合后标识符应正确。"""
        agent = self._make_agent(agent_tag="test")
        ident = agent.identifier

        assert "BufferMemory" in ident
        assert "gemini" in ident
        assert "test" in ident

    def test_max_turns_with_llm(self) -> None:
        """带 max_turns 的 Agent 应能正常运作。"""
        agent = self._make_agent(max_turns=2)

        agent.chat("记住数字42。")
        agent.chat("记住数字99。")
        response = agent.chat("你好")

        assert isinstance(response, str)
        assert len(response) > 0
