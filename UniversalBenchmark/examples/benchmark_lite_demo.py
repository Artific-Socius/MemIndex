"""Benchmark Lite 编程式使用示例。

展示如何在 Python 代码中直接使用 Benchmark Lite 框架的三种模式：
1. 脚本化场景 + 逐回合评估
2. 预置历史 + 问答评估
3. 交互式场景 + 事后评估

用法::

    cd UniversalBenchmark
    python examples/benchmark_lite_demo.py
"""

from dotenv import load_dotenv

load_dotenv()

from agent import Agent  # noqa: E402
from agent.memory import BufferMemory  # noqa: E402
from benchmark_lite import Runner, format_report, to_json  # noqa: E402
from benchmark_lite.examples import (  # noqa: E402
    AdaptiveMemoryProbe,
    SimpleMemoryQA,
)

MODEL = "openrouter/google/gemini-2.5-flash-lite"
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always respond in the same language as the user."
)


def make_agent() -> Agent:
    AgentCls = Agent.compose(BufferMemory)
    return AgentCls(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_turns=20,
    )


def run_simple_qa() -> None:
    """运行脚本化 + 预置历史 Benchmark。"""
    print("\n>>> 运行 SimpleMemoryQA（脚本化 + 预置历史）\n")
    agent = make_agent()
    benchmark = SimpleMemoryQA()
    result = Runner(verbose=True).run(agent, benchmark)
    print(format_report(result, verbose=True))


def run_adaptive_probe() -> None:
    """运行交互式 Benchmark。"""
    print("\n>>> 运行 AdaptiveMemoryProbe（交互式）\n")
    agent = make_agent()
    benchmark = AdaptiveMemoryProbe()
    result = Runner(verbose=True).run(agent, benchmark)
    print(format_report(result, verbose=True))

    json_str = to_json(result)
    with open("adaptive_probe_result.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    print("JSON 结果已保存到 adaptive_probe_result.json")


if __name__ == "__main__":
    import sys

    choice = sys.argv[1] if len(sys.argv) > 1 else "simple"

    if choice == "simple":
        run_simple_qa()
    elif choice == "adaptive":
        run_adaptive_probe()
    elif choice == "all":
        run_simple_qa()
        run_adaptive_probe()
    else:
        print("可选: simple | adaptive | all")
        sys.exit(1)
