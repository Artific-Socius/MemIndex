from __future__ import annotations

from typing import Any, Optional, Type

from .llm.base import LLMMixin
from .memory.base import MemoryMixin, TurnTrace


class Agent(LLMMixin, MemoryMixin):
    """由 **LLM** 和 **Memory** Mixin 组合而成的对话 Agent。

    ``Agent`` 本身是抽象的（继承了 :class:`MemoryMixin` 中未实现的方法）。
    通过以下方式创建具体的 Agent：

    **方式 1 – 显式子类化** ::

        class MyAgent(BufferMemory, Agent):
            pass

        agent = MyAgent(model="gpt-4o", system_prompt="You are helpful.")

    **方式 2 – 自定义 LLM + Memory** ::

        class MyAgent(JSONModeLLM, RAGMemory, Agent):
            pass

    **方式 3 – 动态组合** ::

        AgentCls = Agent.compose(BufferMemory)
        agent = AgentCls(model="gpt-4o", system_prompt="You are helpful.")

    所有关键字参数通过协作式 ``__init__`` 链传递，
    每个 Mixin 自动获取它所声明的参数。
    """

    def __init__(self, agent_tag: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent_tag = agent_tag
        self._last_turn_trace: Optional[TurnTrace] = None

    # ------------------------------------------------------------------
    # 核心对话循环
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """执行一轮对话。

        处理流程::

            1. memory._init_turn_trace()         → 初始化追踪
            2. memory.get_messages(user_input)    → 构建上下文
            3. llm.generate(messages)             → LLM 生成回复
            4. memory.add_response(response)      → 持久化回复
            5. memory.get_last_turn_trace()       → 采集追踪
            6. 返回回复文本
        """
        self._init_turn_trace()
        messages = self.get_messages(user_input)
        response = self.generate(messages)
        self.add_response(response)
        self._last_turn_trace = self.get_last_turn_trace()
        return response

    @property
    def last_turn_trace(self) -> Optional[TurnTrace]:
        """最近一次 :meth:`chat` 调用产生的 message id 追踪记录。

        Runner 在每轮 chat 后读取此属性，写入 TurnResult。
        """
        return self._last_turn_trace

    @property
    def memory_library_id(self) -> str:
        """当前 memory backend 的库标识（由 MemoryMixin 提供）。"""
        return self.get_memory_library_id()

    # ------------------------------------------------------------------
    # 标识符
    # ------------------------------------------------------------------

    @property
    def identifier(self) -> str:
        """可读的唯一标识符。

        格式: ``Agent(<tag>)[<llm_id>|<memory_id>]``

        示例: ``Agent(eval)[openai/gpt-4o:creative|BufferMemory:default]``
        """
        tag = f"({self._agent_tag})" if self._agent_tag else ""
        return f"Agent{tag}[{self.llm_identifier}|{self.memory_identifier}]"

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def compose(
        cls,
        memory_cls: Type[MemoryMixin],
        llm_cls: Optional[Type[LLMMixin]] = None,
        *,
        name: Optional[str] = None,
    ) -> Type[Agent]:
        """动态创建一个具体的 ``Agent`` 子类。

        Parameters
        ----------
        memory_cls:
            一个具体的 :class:`MemoryMixin` 子类。
        llm_cls:
            可选的 :class:`LLMMixin` 子类。
            为 *None* 时使用默认的 ``LLMMixin``（直接调用 litellm）。
        name:
            生成类型的类名。为 *None* 时自动生成。

        Returns
        -------
        Type[Agent]
            可直接实例化的新类。

        示例::

            AgentCls = Agent.compose(BufferMemory, JSONModeLLM)
            agent = AgentCls(model="gpt-4o", system_prompt="Hi")
        """
        if name is None:
            llm_part = llm_cls.__name__ if llm_cls else "LLM"
            name = f"{llm_part}_{memory_cls.__name__}_Agent"

        bases: list[type] = []
        if llm_cls is not None:
            bases.append(llm_cls)
        bases.append(memory_cls)
        bases.append(cls)

        return type(name, tuple(bases), {})

    def __repr__(self) -> str:
        return f"<{self.identifier}>"
