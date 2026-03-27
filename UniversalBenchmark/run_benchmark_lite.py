"""Benchmark Lite 命令行入口。

通过命令行参数指定 Benchmark 实现和 Agent 配置，
一键运行评估并输出报告。

支持两种 Benchmark 类型：

1. **BenchmarkLite 子类** — 包含完整执行逻辑的 Benchmark。
2. **benchmark.interfaces.Benchmark 子类** — 纯数据层 Benchmark，
   自动通过 ``UniversalAdapter`` 适配为可执行的 BenchmarkLite。

用法::

    cd UniversalBenchmark

    # 使用 BenchmarkLite 子类（包含执行逻辑）
    python run_benchmark_lite.py \\
        --benchmark benchmark_lite.examples.SimpleMemoryQA \\
        --memory buffer \\
        --model openrouter/google/gemini-2.5-flash-lite

    # 使用数据层 Benchmark（自动适配）
    python run_benchmark_lite.py \\
        --benchmark benchmark.data.providers.evermind_ai.evermembench_static.EverMemBenchStaticBenchmark \\
        --memory buffer \\
        --model openrouter/google/gemini-2.5-flash \\
        --eval-model openrouter/google/gemini-2.5-flash \\
        --scene-ids 0

    # 自定义 Benchmark（只要在 Python 路径上可导入即可）
    python run_benchmark_lite.py \\
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

from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True)

from dotenv import load_dotenv

load_dotenv()

from agent import Agent  # noqa: E402
from agent.env_config import get_memory_config, load_env_config  # noqa: E402
from agent.memory import BufferMemory, Mem0Memory, MemechoMemory  # noqa: E402
from agent.memory.base import MemoryMixin  # noqa: E402
from benchmark.interfaces import Benchmark as DataBenchmark  # noqa: E402
from benchmark_lite import (  # noqa: E402
    BenchmarkLite,
    RunConfig,
    Runner,
    UniversalAdapter,
    format_report,
    to_json,
)

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
    max_turns: int | None = None,
    read_only: bool = False,
    memory_lib_id: str | None = None,
) -> Agent:
    """根据参数创建 Agent 实例。

    自动从 ``env_config.yaml`` 中读取对应 Memory 类型的配置参数，
    并作为额外的构造参数传递给 Agent。
    ``max_turns`` 会覆盖配置文件中的 ``max_turns``。
    ``read_only`` 和 ``memory_lib_id`` 仅对 Memecho 生效。
    """
    if memory_name not in MEMORY_TYPES:
        raise ValueError(
            f"不支持的 Memory 类型 '{memory_name}'，"
            f"可选: {list(MEMORY_TYPES.keys())}"
        )
    memory_cls = MEMORY_TYPES[memory_name]
    env_kwargs = get_memory_config(memory_name)
    if max_turns is not None:
        env_kwargs["max_turns"] = max_turns
    if read_only:
        env_kwargs["read_only"] = True
    if memory_lib_id:
        env_kwargs["memory_lib_id"] = memory_lib_id
    agent_cls = Agent.compose(memory_cls)
    return agent_cls(model=model, system_prompt=system_prompt, **env_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Lite — 评估 Agent 记忆能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help=(
            "Benchmark 类的点分路径。支持两种类型：\n"
            "  1) BenchmarkLite 子类（包含完整执行逻辑）\n"
            "  2) benchmark.interfaces.Benchmark 子类（纯数据，自动适配）"
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
        "--eval-model",
        default="openrouter/google/gemini-2.5-flash",
        help=(
            "用于评估的 LLM 模型（仅在使用 UniversalAdapter 时生效，"
            "默认: openrouter/google/gemini-2.5-flash）"
        ),
    )
    parser.add_argument(
        "--scene-ids",
        default=None,
        nargs="+",
        help=(
            "指定要运行的 Scene ID 列表（仅在使用 UniversalAdapter 时生效，"
            "不指定则运行所有 Scene）"
        ),
    )
    parser.add_argument(
        "--max-bg-chars",
        type=int,
        default=None,
        help=(
            "背景文本最大字符数（仅 UniversalAdapter 生效）。"
            "截断过长的 background_text 以控制上下文大小"
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "每个 Scene 最多评估的问题数（仅 UniversalAdapter 生效）。"
            "用于 debug/快速验证时限制问题数量"
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "覆盖 BufferMemory 的 max_turns 参数，"
            "启用滑动窗口以控制上下文大小"
        ),
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Memecho 查询时不持久化用户消息（read_only=True）。"
            "用于评估场景，避免评估问题污染已导入的语料库"
        ),
    )
    parser.add_argument(
        "--memory-lib-id",
        default=None,
        help=(
            "手动指定 Memecho 记忆库 ID，跳过语料导入。"
            "适用于复用已导入语料的记忆库"
        ),
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

    # ── 加载环境配置 ────────────────────────────────────────
    load_env_config()
    mem_cfg = get_memory_config(args.memory)
    if mem_cfg:
        print(f"[env_config] 已加载 {args.memory} 的环境配置: {mem_cfg}")

    # ── 加载 Benchmark ──────────────────────────────────────
    benchmark_cls = load_class(args.benchmark)
    benchmark: BenchmarkLite

    if isinstance(benchmark_cls, type) and issubclass(
        benchmark_cls, BenchmarkLite,
    ):
        benchmark = benchmark_cls()
        print(f"[benchmark] BenchmarkLite 直接加载: {benchmark.name}")

    elif isinstance(benchmark_cls, type) and issubclass(
        benchmark_cls, DataBenchmark,
    ):
        data_benchmark: DataBenchmark = benchmark_cls()
        benchmark = UniversalAdapter(
            data_benchmark,
            eval_model=args.eval_model,
            scene_ids=args.scene_ids,
            max_bg_chars=args.max_bg_chars,
            max_questions=args.max_questions,
        )
        print(
            f"[benchmark] 数据层 Benchmark 已通过 UniversalAdapter 适配: "
            f"{benchmark.name}"
        )
    else:
        print(
            f"错误: '{args.benchmark}' 既不是 BenchmarkLite 的子类，"
            f"也不是 benchmark.interfaces.Benchmark 的子类",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 创建 Agent ──────────────────────────────────────────
    agent = build_agent(
        args.memory, args.model, args.system_prompt,
        max_turns=args.max_turns,
        read_only=args.read_only,
        memory_lib_id=args.memory_lib_id,
    )

    # ── 运行 ────────────────────────────────────────────────
    runner = Runner(verbose=True)
    result = runner.run(agent, benchmark)

    # ── 注入运行配置快照 ──────────────────────────────────
    extra_cfg: dict = {}
    if args.scene_ids is not None:
        extra_cfg["scene_ids"] = args.scene_ids
    if args.max_bg_chars is not None:
        extra_cfg["max_bg_chars"] = args.max_bg_chars
    if args.max_questions is not None:
        extra_cfg["max_questions"] = args.max_questions
    if args.max_turns is not None:
        extra_cfg["max_turns"] = args.max_turns
    if args.read_only:
        extra_cfg["read_only"] = True
    if args.memory_lib_id:
        extra_cfg["memory_lib_id"] = args.memory_lib_id
    extra_cfg["benchmark_path"] = args.benchmark

    result.run_config = RunConfig(
        memory_type=args.memory,
        model=args.model,
        eval_model=args.eval_model,
        system_prompt=args.system_prompt,
        extra=extra_cfg,
    )

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
