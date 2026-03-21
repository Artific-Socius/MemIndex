"""Benchmark Lite 命令行入口。

通过命令行参数指定 Benchmark 实现和 Agent 配置，
一键运行评估并输出报告。

用法::

    cd UniversalBenchmark

    # 使用内置示例 Benchmark
    python run_benchmark.py \\
        --benchmark benchmark_lite.examples.SimpleMemoryQA \\
        --memory buffer \\
        --model openrouter/google/gemini-2.5-flash-lite

    # 使用自定义 Benchmark（只要在 Python 路径上可导入即可）
    python run_benchmark.py \\
        --benchmark my_benchmarks.custom.MyBenchmark \\
        --memory memecho \\
        --model openai/gpt-4o \\
        --verbose \\
        --output results.json
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Type

from dotenv import load_dotenv

load_dotenv()

from agent import Agent  # noqa: E402
from agent.memory import BufferMemory, Mem0Memory, MemechoMemory  # noqa: E402
from agent.memory.base import MemoryMixin  # noqa: E402
from benchmark_lite import BenchmarkLite, Runner, format_report, to_json  # noqa: E402

MEMORY_TYPES: dict[str, Type[MemoryMixin]] = {
    "buffer": BufferMemory,
    "mem0": Mem0Memory,
    "memecho": MemechoMemory,
}


def load_class(dotted_path: str) -> type:
    """通过 Python 点分路径动态加载类。

    支持格式：
    - ``package.module.ClassName``
    - ``package.ClassName`` （ClassName 在 package.__init__ 中导出）
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"无法解析路径 '{dotted_path}'，"
            f"请使用 'module.ClassName' 格式"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"模块 '{module_path}' 中未找到 '{class_name}'"
        )
    if not isinstance(cls, type):
        raise TypeError(
            f"'{dotted_path}' 不是一个类"
        )
    return cls


def build_agent(
    memory_name: str,
    model: str,
    system_prompt: str,
) -> Agent:
    """根据参数创建 Agent 实例。"""
    if memory_name not in MEMORY_TYPES:
        raise ValueError(
            f"不支持的 Memory 类型 '{memory_name}'，"
            f"可选: {list(MEMORY_TYPES.keys())}"
        )
    memory_cls = MEMORY_TYPES[memory_name]
    agent_cls = Agent.compose(memory_cls)
    return agent_cls(model=model, system_prompt=system_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Lite — 评估 Agent 记忆能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help=(
            "BenchmarkLite 子类的点分路径，"
            "如 benchmark_lite.examples.SimpleMemoryQA"
        ),
    )
    parser.add_argument(
        "--memory",
        default="buffer",
        choices=list(MEMORY_TYPES.keys()),
        help="Agent 使用的 Memory 类型（默认: buffer）",
    )
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-2.5-flash-lite",
        help="LLM 模型名称（默认: openrouter/google/gemini-2.5-flash-lite）",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful assistant. "
            "Always respond in the same language as the user."
        ),
        help="Agent 的系统提示词",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="在报告中显示每个回合的详细信息",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="将结果保存到指定文件（JSON 格式）",
    )

    args = parser.parse_args()

    # ── 加载 Benchmark ──────────────────────────────────────
    benchmark_cls = load_class(args.benchmark)
    if not (isinstance(benchmark_cls, type)
            and issubclass(benchmark_cls, BenchmarkLite)):
        print(
            f"错误: '{args.benchmark}' 不是 BenchmarkLite 的子类",
            file=sys.stderr,
        )
        sys.exit(1)

    benchmark: BenchmarkLite = benchmark_cls()

    # ── 创建 Agent ──────────────────────────────────────────
    agent = build_agent(args.memory, args.model, args.system_prompt)

    # ── 运行 ────────────────────────────────────────────────
    runner = Runner(verbose=True)
    result = runner.run(agent, benchmark)

    # ── 输出报告 ────────────────────────────────────────────
    report = format_report(result, verbose=args.verbose)
    print(report)

    # ── 保存 JSON ───────────────────────────────────────────
    if args.output:
        json_str = to_json(result)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
