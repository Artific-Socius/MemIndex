"""快速上手：三种 Memory 组合的 Agent 示例

用法:
    cd UniversalBenchmark
    python examples/chat_agents.py [buffer|memecho|mem0]

不传参数默认运行 buffer 示例。
所有环境变量从项目根目录的 .env 读取。
"""

from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from agent.memory import BufferMemory, MemechoMemory, Mem0Memory

MODEL = "openrouter/google/gemini-2.5-flash-lite"
SYSTEM_PROMPT = "你是一个简洁友好的助手，用中文回答。"


# ------------------------------------------------------------------
# 1. 基线 Agent —— BufferMemory（纯本地，无需 API Key）
# ------------------------------------------------------------------

def make_buffer_agent() -> Agent:
    AgentCls = Agent.compose(BufferMemory)
    return AgentCls(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_turns=10,
    )


# ------------------------------------------------------------------
# 2. Memecho Agent —— MemechoMemory（需要 MEMECHO_API_KEY）
# ------------------------------------------------------------------

def make_memecho_agent() -> Agent:
    AgentCls = Agent.compose(MemechoMemory)
    return AgentCls(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )


# ------------------------------------------------------------------
# 3. Mem0 Agent —— Mem0Memory（需要 MEM0_API_KEY）
# ------------------------------------------------------------------

def make_mem0_agent() -> Agent:
    AgentCls = Agent.compose(Mem0Memory)
    return AgentCls(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        recent_turns=5,
    )


# ------------------------------------------------------------------
# 交互式对话循环
# ------------------------------------------------------------------

AGENTS = {
    "buffer": ("BufferMemory（本地全量历史）", make_buffer_agent),
    "memecho": ("MemechoMemory（Memecho 云 API）", make_memecho_agent),
    "mem0": ("Mem0Memory（Mem0 云 API）", make_mem0_agent),
}


def chat_loop(agent: Agent) -> None:
    print(f"\n  Agent: {agent.identifier}")
    print("  输入 q 退出\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input or user_input.lower() == "q":
            break
        response = agent.chat(user_input)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    import sys

    choice = sys.argv[1] if len(sys.argv) > 1 else "buffer"

    if choice not in AGENTS:
        print(f"可选: {', '.join(AGENTS)}")
        sys.exit(1)

    label, factory = AGENTS[choice]
    print(f"\n>>> 启动 {label}")
    agent = factory()
    chat_loop(agent)
