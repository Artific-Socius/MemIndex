"""
MemIndex - 长期记忆基准测试框架

主入口文件，用于运行单个基准测试任务。

执行流程:
    1. 解析命令行参数和配置文件
    2. 初始化 LLM 控制器和 Agent
    3. 加载测试数据集，创建执行器 (Actuator)
    4. 创建运行器 (Runner) 并执行测试
    5. 生成并保存测试报告
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from loguru import logger
from rich.panel import Panel
from rich.table import Table

from config import Config, ConfigManager, RunningConfig, RunningConfigManager, merge_config_with_args
from utils import LLMController, load_dataset, setup_logging, get_console
from core import Actuator, Runner, Report
from prompts import get_prompt_manager

# 获取 MemIndex 模块的根目录，用于解析相对路径
MEMINDEX_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """
    解析路径，将相对路径转换为相对于 MemIndex 模块的绝对路径
    
    这个函数确保无论从哪个目录运行脚本，相对路径都能正确解析。
    
    Args:
        path: 原始路径（可以是相对路径或绝对路径）
        
    Returns:
        解析后的绝对路径
    """
    if os.path.isabs(path):
        return path
    # 相对路径相对于 MemIndex 模块目录
    return os.path.normpath(os.path.join(MEMINDEX_ROOT, path))


def print_config_table(running_config: RunningConfig, benchmark_config_path: str, report_dir_path: str) -> None:
    """
    打印当前运行配置的表格
    
    以美观的表格形式展示所有关键配置参数，方便用户确认配置是否正确。
    
    Args:
        running_config: 运行配置对象
        benchmark_config_path: 基准测试配置文件路径
        report_dir_path: 报告输出目录路径
    """
    console = get_console()
    
    # 创建配置表格
    table = Table(title="Running Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")
    
    # 添加各项配置
    table.add_row("Chat Model", running_config.chat_model)           # 被测试的对话模型
    table.add_row("Eval Model", running_config.eval_model)           # 用于评估的模型
    table.add_row("Memory Provider", running_config.memory_provider) # Agent 类型
    table.add_row("Context Window", str(running_config.context_window))  # 上下文窗口大小
    table.add_row("Chat Prompt", running_config.chat_prompt or "(default)")
    table.add_row("Eval Prompt", running_config.eval_prompt or "(default)")
    table.add_row("Eval Mode", running_config.eval_mode)             # 评估模式: binary/score
    table.add_row("Benchmark Config", benchmark_config_path)
    table.add_row("Report Dir", report_dir_path)
    
    console.print(Panel(table, border_style="blue"))


def print_results_table(actuators: dict, runner: Runner) -> None:
    """
    打印测试结果表格
    
    展示每个执行器的 token 消耗情况，帮助分析记忆距离和测试覆盖范围。
    
    Args:
        actuators: 执行器字典 {名称: Actuator实例}
        runner: 运行器实例，包含测试日志
    """
    console = get_console()
    
    table = Table(title="Benchmark Results", show_header=True, header_style="bold green")
    table.add_column("Actuator", style="cyan")
    table.add_column("Token Start", justify="right")  # 该执行器开始时的总 token 数
    table.add_column("Token End", justify="right")    # 该执行器结束时的总 token 数
    table.add_column("Total Tokens", justify="right", style="bold")  # 该执行器消耗的 token
    
    # 遍历每个执行器，从 runner 的日志中获取 token 统计
    for index, (name, actuator) in enumerate(actuators.items()):
        token_start = runner.sub_test_log[index][0]
        token_end = runner.sub_test_log[index][1]
        total_tokens = token_end - token_start
        table.add_row(
            name,
            str(token_start),
            str(token_end),
            str(total_tokens),
        )
    
    console.print(Panel(table, border_style="green"))


async def main(args) -> None:
    """
    主函数 - 执行单个基准测试任务
    
    这是整个测试流程的核心函数，负责：
    1. 加载和合并配置
    2. 初始化各组件（LLM控制器、Agent、执行器）
    3. 运行测试
    4. 生成报告
    
    Args:
        args: 命令行参数对象
    """
    console = get_console()
    
    # ========== 第一步：加载配置 ==========
    # 解析配置文件路径
    config_path = resolve_path(args.config)
    running_config_path = resolve_path(args.running_config)
    
    # 加载系统配置 (LLM providers、重试次数等)
    config_manager = ConfigManager[Config](config_path, Config)
    config = config_manager.get_config()
    
    # 加载运行配置 (模型、Agent类型、数据集等)
    running_config_manager = RunningConfigManager(running_config_path, RunningConfig)
    running_config = running_config_manager.get_config()
    
    # 命令行参数覆盖配置文件参数（优先级：命令行 > 配置文件）
    running_config = merge_config_with_args(running_config, args)
    
    # 解析运行配置中的路径
    benchmark_config_path = resolve_path(running_config.benchmark_config)
    report_dir_path = resolve_path(running_config.report_dir)
    
    # 初始化提示词管理器
    prompt_manager = get_prompt_manager(resolve_path("prompts/prompts.yaml"))
    
    # 打印配置信息供用户确认
    print_config_table(running_config, benchmark_config_path, report_dir_path)
    
    # 验证配置文件存在
    if not os.path.exists(benchmark_config_path):
        logger.error(f"Benchmark config file not found: {benchmark_config_path}")
        exit(1)
    
    # 确保报告目录存在
    if not os.path.exists(report_dir_path) or not os.path.isdir(report_dir_path):
        os.makedirs(report_dir_path)
        logger.info(f"Created report directory: {report_dir_path}")
    
    # 生成唯一的基准测试名称，用于报告文件命名
    benchmark_name = (
        f"{running_config.benchmark_config.split('/')[-1].replace('.', '_')}-"
        f"{running_config.memory_provider.replace('/', '_')}-"
        f"{running_config.chat_model.replace('/', '_')}-"
        f"{running_config.eval_mode}"
    )
    
    logger.debug(f"Benchmark name: {benchmark_name}")
    
    # ========== 第二步：初始化组件 ==========
    # 初始化 LLM 控制器 (使用 LiteLLM 统一调用各种 LLM API)
    logger.info("Initializing LLM Controller (LiteLLM)...")
    llm_controller = LLMController(
        env_file=resolve_path(".env"),
        retry_times=config.llm_config.llm_retry_times,
    )
    await llm_controller._init_provider()  # 异步初始化，加载 API Key 等
    
    # 根据配置加载对应的 Agent（被测试的对象）
    logger.info("Loading Agent...")
    agent = load_agent(
        running_config, 
        llm_controller, 
        prompt_manager=prompt_manager,
        chat_prompt_key=running_config.chat_prompt,
    )
    
    # ========== 第三步：创建执行器 ==========
    logger.info("Loading Actuators...")
    # 加载测试数据集（包含多个测试序列）
    benchmark_dataset = load_dataset(benchmark_config_path)
    
    # 为每个测试序列创建一个执行器
    # 例如：color.json -> 颜色记忆测试, joke.json -> 笑话记忆测试
    actuators = {}
    for name, sequence in benchmark_dataset.data.items():
        actuator = Actuator(
            data=sequence.items,              # 测试数据项列表
            llm_controller=llm_controller,
            agent=agent,
            eval_model=running_config.eval_model,  # 用于评分的模型
            prompt_manager=prompt_manager,
            eval_prompt_key=running_config.eval_prompt,
            eval_mode=running_config.eval_mode,    # binary 或 score
        )
        actuator.name = name  # 设置执行器名称，用于日志和报告
        actuators[name] = actuator
    
    logger.debug(f"Loaded {len(actuators)} actuators")
    logger.debug(f"Evaluation model: {running_config.eval_model}")
    
    # ========== 第四步：创建运行器并执行测试 ==========
    # Runner 负责协调多个执行器的交替运行，并管理记忆距离
    runner = Runner(
        actuators=list(actuators.values()),
        nonsense=benchmark_dataset.nonsense_list,      # 废话列表（用于填充记忆距离）
        head_prompts=benchmark_dataset.head_prompts,   # 开场提示
        agent=agent,
        memory_distance=benchmark_dataset.memory_distance,  # 记忆距离（tokens）
        eval_model=running_config.eval_model,
        show_progress=True,  # 显示进度条
    )
    
    # 开始运行测试
    time_start = time.time()
    console.print()
    logger.info("Starting benchmark run...")
    console.print()
    
    await runner.run()  # 核心：执行完整的测试流程
    
    time_end = time.time()
    elapsed = time_end - time_start
    
    console.print()
    logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
    
    # 打印结果表格
    print_results_table(actuators, runner)
    
    # ========== 第五步：生成并保存报告 ==========
    logger.info("Generating report...")
    try:
        report = Report(
            report_path=report_dir_path,
            config_path=benchmark_config_path,
            time_start=time_start,
            time_end=time_end,
            runner=runner,
            actuator_names=list(actuators.keys()),
            agent=running_config.memory_provider,
            benchmark_name=benchmark_name,
            full_tokens=runner.current_tokens,
            model=running_config.chat_model,
            extra_metadata=agent.extra_metadata,
            eval_mode=running_config.eval_mode,
            chat_prompt=running_config.chat_prompt,
            eval_prompt=running_config.eval_prompt,
        )
        report.save()
        logger.info(f"Report saved to: {report_dir_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")


def load_agent(
    running_config: RunningConfig, 
    llm_controller: LLMController,
    prompt_manager=None,
    chat_prompt_key: str = None,
):
    """
    根据配置加载对应的 Agent（被测试对象）
    
    Agent 是整个 benchmark 的核心被测对象，不同类型的 Agent 代表不同的记忆方案：
    - llm: 纯 LLM，仅依赖上下文窗口，无外部记忆
    - mem0: 使用 Mem0 Cloud 的记忆增强 Agent
    - mem0_graph: 使用 Mem0 Graph 的记忆增强 Agent
    - memecho: 使用 Memecho 服务的记忆增强 Agent
    
    Args:
        running_config: 运行配置
        llm_controller: LLM 控制器
        prompt_manager: 提示词管理器
        chat_prompt_key: Chat 提示词的 key
        
    Returns:
        Agent 实例（继承自 BaseAgent）
    """
    agent_type = running_config.memory_provider
    
    if agent_type == "llm":
        # 纯 LLM Agent：不使用任何外部记忆系统
        # 完全依赖模型的上下文窗口进行"记忆"
        from components.agents import LLMAgent
        agent = LLMAgent(
            llm_controller=llm_controller,
            model=running_config.chat_model,
            context_window=running_config.context_window,
        )
        logger.info(f"Loaded LLM Agent")
        logger.debug(f"Target Model: {running_config.chat_model}")
    
    elif agent_type == "memecho":
        # Memecho Agent：使用 Memecho 记忆服务
        from components.agents import MemechoAgent
        agent = MemechoAgent(
            llm_controller=llm_controller,
            model=running_config.chat_model,
            context_window=running_config.context_window,
        )
        logger.info(f"Loaded Memecho Agent")
        logger.debug(f"Target Model: {running_config.chat_model}")
    
    elif agent_type == "example":
        # 示例 Agent：用于开发和测试
        from components.agents import ExampleAgent
        agent = ExampleAgent()
        logger.info(f"Loaded Example Agent")
    
    elif agent_type == "mem0":
        # Mem0 Agent：使用 Mem0 Cloud API 进行记忆管理
        from components.agents import Mem0Agent
        agent = Mem0Agent(
            llm_controller=llm_controller,
            model=running_config.chat_model,
            context_window=running_config.context_window,
            prompt_manager=prompt_manager,
            chat_prompt_key=chat_prompt_key,
        )
        logger.info(f"Loaded Mem0 Agent")
    
    elif agent_type == "mem0_graph":
        # Mem0 Graph Agent：使用 Mem0 的图记忆功能
        from components.agents import Mem0GraphAgent
        agent = Mem0GraphAgent(
            llm_controller=llm_controller,
            model=running_config.chat_model,
            context_window=running_config.context_window,
            prompt_manager=prompt_manager,
            chat_prompt_key=chat_prompt_key,
        )
        logger.info(f"Loaded Mem0 Graph Agent")
    
    else:
        raise NotImplementedError(f"Unknown agent type: {agent_type}")
    
    return agent


def parse_args():
    """
    解析命令行参数
    
    支持两类参数：
    1. 配置文件路径参数：指定配置文件位置
    2. 覆盖参数：直接在命令行覆盖配置文件中的设置
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Run Benchmark.")
    
    # ========== 配置文件路径参数 ==========
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="System Config (LLM providers), relative to MemIndex module"
    )
    
    parser.add_argument(
        "--running_config",
        type=str,
        default="running_config.yaml",
        help="Running Config (experiment variables), relative to MemIndex module"
    )
    
    # ========== 覆盖参数（优先级高于配置文件）==========
    parser.add_argument(
        "--benchmark_config",
        type=str,
        default="./data/config/2k.json",
        help="Benchmark Config (overrides running_config)"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="./data/reports",
        help="Report Directory (overrides running_config)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Memory Provider/Agent type: llm, memecho, example, mem0, mem0_graph (overrides running_config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Chat model for conversation (overrides running_config)"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="volcano/deepseek-v3-250324",
        help="Evaluation model (overrides running_config)"
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=16384,
        help="Context window size in tokens (overrides running_config)"
    )
    parser.add_argument(
        "--chat_prompt",
        type=str,
        default=None,
        help="Chat prompt key from prompts.yaml (overrides running_config)"
    )
    parser.add_argument(
        "--eval_prompt",
        type=str,
        default=None,
        help="Eval prompt key from prompts.yaml (overrides running_config)"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="binary",
        choices=["binary", "score"],
        help="Evaluation mode: binary (correct/incorrect) or score (0-1 continuous)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    return parser.parse_args()


def run():
    """
    程序运行入口
    
    负责：
    1. 解析命令行参数
    2. 初始化日志系统
    3. 显示启动信息
    4. 运行主异步函数
    """
    args = parse_args()
    
    # 初始化日志系统
    setup_logging(level=args.log_level)
    
    # 显示启动 Banner
    console = get_console()
    console.print()
    console.print(Panel(
        "[bold cyan]MemIndex[/bold cyan] - Long-term Memory Benchmark Framework",
        border_style="cyan",
    ))
    console.print()
    
    # 运行异步主函数
    asyncio.run(main(args))


if __name__ == "__main__":
    run()
