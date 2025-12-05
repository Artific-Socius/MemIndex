"""
MemIndex Batch Runner - 批量运行入口

支持多个配置并行运行基准测试。
所有任务配置在一个 batch_config.yaml 文件中。

与 main.py 的区别:
    - main.py: 单任务运行，适合调试和单次测试
    - batch_main.py: 批量并行运行，适合大规模对比实验

核心特性:
    - 并行执行：通过 max_parallel 控制同时运行的任务数
    - 任务隔离：每个任务有独立的 LLM 控制器和费用追踪
    - 实时显示：使用 Rich 显示多任务进度
    - 优雅中断：支持 Ctrl+C 中断，已完成任务会保存
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from rich.panel import Panel

from config import Config, ConfigManager
from config.batch_config import BatchConfig, BatchConfigManager, TaskConfig
from utils import load_dataset, setup_logging, get_console
from utils.litellm_controller import LiteLLMController, CostTracker
from utils.task_display import MultiTaskDisplay, TaskStatus
from core import Actuator, Runner, Report
from prompts import get_prompt_manager

# 获取 MemIndex 模块的根目录
MEMINDEX_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """
    解析路径，将相对路径转换为绝对路径
    
    Args:
        path: 原始路径
        
    Returns:
        解析后的绝对路径
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(MEMINDEX_ROOT, path))


class BatchRunner:
    """
    批量运行器
    
    管理多个任务的并行执行，提供：
    - 信号量控制并行数
    - 独立的费用追踪
    - 实时进度显示
    - 错误处理和中断支持
    
    工作流程:
        1. 注册所有任务到显示器
        2. 通过信号量控制并行数
        3. 为每个任务创建独立的 LLM 控制器
        4. 执行任务并实时更新进度
        5. 生成报告并保存结果
    """
    
    def __init__(
        self,
        batch_config: BatchConfig,
        system_config: Config,
    ):
        """
        初始化批量运行器
        
        Args:
            batch_config: 批量运行配置（包含所有任务定义）
            system_config: 系统配置（LLM providers 等）
        """
        self.batch_config = batch_config
        self.system_config = system_config
        self.display = MultiTaskDisplay(get_console())  # 多任务进度显示器
        self.results: Dict[str, Any] = {}               # 存储各任务的结果
        self._semaphore: asyncio.Semaphore = None       # 控制并行数的信号量
        
        # 初始化 PromptManager（所有任务共享）
        self.prompt_manager = get_prompt_manager(resolve_path("prompts/prompts.yaml"))
    
    async def run(self) -> Dict[str, Any]:
        """
        运行所有任务
        
        这是批量运行的主入口，负责：
        1. 创建并行控制信号量
        2. 注册所有任务
        3. 并发执行任务
        4. 处理中断和错误
        
        Returns:
            任务结果字典 {task_id: result}
        """
        tasks = self.batch_config.tasks
        
        if not tasks:
            logger.warning("No tasks found in config")
            return {}
        
        # 创建信号量控制并行数，限制同时运行的任务数量
        self._semaphore = asyncio.Semaphore(self.batch_config.max_parallel)
        self._interrupted = False    # 是否被中断
        self._async_tasks: list[asyncio.Task] = []  # 异步任务列表
        
        # 注册所有任务到显示器（显示任务列表和预估步数）
        for i, task_config in enumerate(tasks):
            task_id = f"task_{i}"
            total_steps = self._estimate_total_steps(task_config)
            self.display.register_task(
                task_id, 
                task_config.name, 
                total_steps,
                chat_model=task_config.chat_model,
            )
        
        # 启动进度显示
        self.display.start()
        
        try:
            # 为每个任务创建异步任务
            for i, task_config in enumerate(tasks):
                task_id = f"task_{i}"
                async_task = asyncio.create_task(
                    self._run_single_task(task_id, task_config)
                )
                self._async_tasks.append(async_task)
                
                # 任务间延迟（避免同时启动造成的资源竞争）
                if self.batch_config.task_delay > 0 and i < len(tasks) - 1:
                    await asyncio.sleep(self.batch_config.task_delay)
            
            # 等待所有任务完成（return_exceptions=True 确保一个任务失败不影响其他任务）
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
        
        except asyncio.CancelledError:
            # 处理 Ctrl+C 中断
            self._interrupted = True
            # 取消所有正在运行的任务
            for task in self._async_tasks:
                if not task.done():
                    task.cancel()
            # 等待所有任务完成取消
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
            # 更新运行中任务的状态为取消
            for task_state in self.display.tasks.values():
                if task_state.status == TaskStatus.RUNNING:
                    task_state.status = TaskStatus.CANCELLED
                    task_state.end_time = time.time()
        
        finally:
            # 停止进度显示
            self.display.stop()
            
            # 打印最终摘要（包含所有任务的结果统计）
            self.display.print_final_summary(interrupted=self._interrupted)
        
        return self.results
    
    def _estimate_total_steps(self, task_config: TaskConfig) -> int:
        """
        估算任务的总步数
        
        通过预加载数据集来计算总步数，用于进度条显示。
        
        Args:
            task_config: 任务配置
            
        Returns:
            预估的总步数
        """
        try:
            benchmark_config_path = resolve_path(task_config.benchmark_config)
            if os.path.exists(benchmark_config_path):
                dataset = load_dataset(benchmark_config_path)
                # 总步数 = 所有测试序列的项数之和
                return sum(len(seq.items) for seq in dataset.data.values())
        except Exception:
            pass
        return 100  # 默认估算值
    
    async def _run_single_task(self, task_id: str, task_config: TaskConfig) -> Dict[str, Any]:
        """
        运行单个任务
        
        这是每个并行任务的入口点，负责：
        1. 获取信号量（控制并行数）
        2. 创建独立的费用追踪器
        3. 执行任务
        4. 更新状态和处理错误
        
        Args:
            task_id: 任务唯一标识
            task_config: 任务配置
            
        Returns:
            任务结果字典
        """
        # 通过信号量控制并行数（async with 自动获取和释放）
        async with self._semaphore:
            self.display.update_task(task_id, status=TaskStatus.RUNNING)
            
            # 为每个任务创建独立的费用追踪器（避免多任务间的费用混淆）
            task_cost_tracker = CostTracker()
            
            try:
                # 执行任务核心逻辑
                result = await self._execute_task(task_id, task_config, task_cost_tracker)
                self.results[task_id] = result
                
                # 更新最终状态为完成
                self.display.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    cost=task_cost_tracker.total_cost,
                )
                
                return result
            
            except Exception as e:
                # 详细错误日志
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                logger.error(f"Task {task_config.name} failed: {error_msg}")
                logger.error(f"Task config: model={task_config.chat_model}, provider={task_config.memory_provider}, eval_mode={task_config.eval_mode}")
                logger.error(f"Full traceback:\n{error_traceback}")
                
                self.display.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                )
                
                # 根据配置决定是否继续执行其他任务
                if not self.batch_config.continue_on_error:
                    raise
                
                return {"error": str(e)}
    
    async def _execute_task(
        self,
        task_id: str,
        task_config: TaskConfig,
        cost_tracker: CostTracker,
    ) -> Dict[str, Any]:
        """
        执行单个任务的核心逻辑
        
        这是任务执行的实际实现，与 main.py 中的 main() 函数类似，
        但增加了进度回调和静默模式支持。
        
        Args:
            task_id: 任务标识
            task_config: 任务配置
            cost_tracker: 费用追踪器
            
        Returns:
            任务结果（包含耗时、token、费用等）
        """
        # 将 TaskConfig 转换为 RunningConfig（统一配置格式）
        running_config = task_config.to_running_config()
        
        # 解析路径
        benchmark_config_path = resolve_path(running_config.benchmark_config)
        report_dir_path = resolve_path(running_config.report_dir)
        
        # 更新显示
        self.display.update_task(task_id, log_message=f"Model: {running_config.chat_model}")
        
        # ========== 创建任务独立的 LLM 控制器 ==========
        llm_controller = LiteLLMController(
            env_file=resolve_path(".env"),
            retry_times=self.system_config.llm_config.llm_retry_times,
            track_cost=True,
        )
        # 使用任务专属的费用追踪器（关键：确保费用统计独立）
        llm_controller.cost_tracker = cost_tracker
        await llm_controller._init_provider()
        
        # ========== 加载 Agent ==========
        self.display.update_task(task_id, log_message="Loading Agent...")
        agent = self._load_agent(
            running_config, 
            llm_controller,
            chat_prompt_key=running_config.chat_prompt,
        )
        
        # ========== 加载数据集 ==========
        self.display.update_task(task_id, log_message="Loading dataset...")
        benchmark_dataset = load_dataset(benchmark_config_path)
        
        # ========== 创建执行器 ==========
        actuators = {}
        for name, sequence in benchmark_dataset.data.items():
            actuator = Actuator(
                data=sequence.items,
                llm_controller=llm_controller,
                agent=agent,
                eval_model=running_config.eval_model,
                prompt_manager=self.prompt_manager,
                eval_prompt_key=running_config.eval_prompt,
                eval_mode=running_config.eval_mode,
            )
            actuator.name = name
            actuators[name] = actuator
        
        # ========== 创建进度回调函数 ==========
        # 这个回调函数会在每个步骤后被调用，用于更新显示
        def progress_callback(
            current_step: int, 
            total_steps: int, 
            tokens: int, 
            cost: float,
            actuator_name: str = "",
            actuator_step: int = 0,
            actuator_total: int = 0,
            last_action: str = "",
        ):
            # 显示当前操作的预览
            if last_action:
                action_preview = last_action[:35] + "..." if len(last_action) > 35 else last_action
                self.display.update_task(task_id, log_message=f"💬 {action_preview}")
            
            # 更新进度信息（包含延迟统计）
            self.display.update_task(
                task_id,
                current_step=current_step,
                current_tokens=tokens,
                cost=cost_tracker.total_cost,
                current_actuator=actuator_name,
                current_actuator_step=actuator_step,
                total_actuator_steps=actuator_total,
                # 延迟统计（从 agent 获取）
                has_memory_backend=agent.has_memory_backend,
                avg_memory_latency=agent.avg_memory_latency,
                avg_chat_latency=agent.avg_chat_latency,
                last_memory_latency=agent.last_memory_latency,
                last_chat_latency=agent.last_chat_latency,
            )
            self.display.refresh()
        
        # ========== 创建 Runner（静默模式，不显示自己的进度条）==========
        runner = Runner(
            actuators=list(actuators.values()),
            nonsense=benchmark_dataset.nonsense_list,
            head_prompts=benchmark_dataset.head_prompts,
            agent=agent,
            memory_distance=benchmark_dataset.memory_distance,
            eval_model=running_config.eval_model,
            show_progress=False,           # 不显示 Runner 自己的进度条
            progress_callback=progress_callback,  # 使用回调更新批量运行器的显示
            silent=True,                   # 静默模式
        )
        
        # 更新总步数（使用实际计算的值替代估算值）
        total_steps = sum(len(a.data) for a in actuators.values())
        self.display.tasks[task_id].total_steps = total_steps
        
        # ========== 运行测试 ==========
        self.display.update_task(task_id, log_message="Running benchmark...")
        time_start = time.time()
        await runner.run()  # 执行测试
        time_end = time.time()
        
        # 确保报告目录存在
        if not os.path.exists(report_dir_path):
            os.makedirs(report_dir_path)
        
        # ========== 生成报告 ==========
        self.display.update_task(task_id, log_message="Generating report...")
        
        benchmark_name = (
            f"{running_config.benchmark_config.split('/')[-1].replace('.', '_')}-"
            f"{running_config.memory_provider.replace('/', '_')}-"
            f"{running_config.chat_model.replace('/', '_')}-"
            f"{running_config.eval_mode}"
        )
        
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
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # 更新最终状态
        self.display.update_task(
            task_id,
            current_step=total_steps,
            current_tokens=runner.current_tokens,
            cost=cost_tracker.total_cost,
            log_message="Complete!",
        )
        
        # 返回任务结果
        return {
            "name": task_config.name,
            "elapsed_time": time_end - time_start,
            "tokens": runner.current_tokens,
            "cost": cost_tracker.total_cost,
            "report_path": report_dir_path,
        }
    
    def _load_agent(
        self, 
        running_config, 
        llm_controller: LiteLLMController,
        chat_prompt_key: str = None,
    ):
        """
        加载 Agent
        
        根据配置创建对应类型的 Agent 实例。
        
        Args:
            running_config: 运行配置
            llm_controller: LLM 控制器
            chat_prompt_key: Chat 提示词 key
            
        Returns:
            Agent 实例
        """
        agent_type = running_config.memory_provider
        
        if agent_type == "llm":
            from components.agents import LLMAgent
            return LLMAgent(
                llm_controller=llm_controller,
                model=running_config.chat_model,
                context_window=running_config.context_window,
            )
        elif agent_type == "memecho":
            from components.agents import MemechoAgent
            return MemechoAgent(
                llm_controller=llm_controller,
                model=running_config.chat_model,
                context_window=running_config.context_window,
            )
        elif agent_type == "example":
            from components.agents import ExampleAgent
            return ExampleAgent()
        elif agent_type == "mem0":
            from components.agents import Mem0Agent
            return Mem0Agent(
                llm_controller=llm_controller,
                model=running_config.chat_model,
                context_window=running_config.context_window,
                prompt_manager=self.prompt_manager,
                chat_prompt_key=chat_prompt_key,
            )
        elif agent_type == "mem0_graph":
            from components.agents import Mem0GraphAgent
            return Mem0GraphAgent(
                llm_controller=llm_controller,
                model=running_config.chat_model,
                context_window=running_config.context_window,
                prompt_manager=self.prompt_manager,
                chat_prompt_key=chat_prompt_key,
            )
        else:
            raise NotImplementedError(f"Unknown agent type: {agent_type}")


async def main(args) -> None:
    """
    批量运行主函数
    
    负责加载配置、创建 BatchRunner 并启动执行。
    
    Args:
        args: 命令行参数
    """
    console = get_console()
    
    # 解析配置文件路径
    config_path = resolve_path(args.config)
    batch_config_path = resolve_path(args.batch_config)
    
    # 加载系统配置
    config_manager = ConfigManager[Config](config_path, Config)
    system_config = config_manager.get_config()
    
    # 加载批量任务配置
    batch_manager = BatchConfigManager(batch_config_path)
    batch_config = batch_manager.load_config()
    
    # 命令行参数覆盖配置
    if args.max_parallel:
        batch_config.max_parallel = args.max_parallel
    
    if not batch_config.tasks:
        logger.error(f"No tasks found in {batch_config_path}")
        return
    
    # 显示任务列表概览
    logger.info(f"Found {len(batch_config.tasks)} tasks:")
    for task in batch_config.tasks:
        logger.info(f"  - {task.name}: {task.chat_model}")
    
    console.print()
    
    # 创建并运行批量运行器
    batch_runner = BatchRunner(
        batch_config=batch_config,
        system_config=system_config,
    )
    
    await batch_runner.run()


def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Run Batch Benchmark.")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="System Config (LLM providers)"
    )
    
    parser.add_argument(
        "--batch_config",
        type=str,
        default="batch_config.yaml",
        help="Batch config file (contains all task configs)"
    )
    
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum number of parallel tasks (overrides config)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tasks",
        help="List all tasks without executing (check configuration)"
    )
    
    return parser.parse_args()


def list_tasks(args) -> None:
    """
    列出所有任务配置（不执行）
    
    用于检查配置是否正确，显示所有任务的详细信息。
    使用 --list 参数触发。
    
    Args:
        args: 命令行参数
    """
    from rich.table import Table
    
    console = get_console()
    batch_config_path = resolve_path(args.batch_config)
    
    try:
        batch_manager = BatchConfigManager(batch_config_path)
        batch_config, all_tasks = batch_manager.load_all_tasks()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    except Exception as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        return
    
    # 显示全局配置
    console.print(Panel(
        f"[cyan]Max Parallel:[/cyan] {batch_config.max_parallel}\n"
        f"[cyan]Continue on Error:[/cyan] {batch_config.continue_on_error}\n"
        f"[cyan]Task Delay:[/cyan] {batch_config.task_delay}s\n"
        f"[cyan]Default Chat Prompt:[/cyan] {batch_config.default_chat_prompt or '(default)'}\n"
        f"[cyan]Default Eval Prompt:[/cyan] {batch_config.default_eval_prompt or '(default)'}\n"
        f"[cyan]Default Eval Mode:[/cyan] {batch_config.default_eval_mode}",
        title="[bold blue]Global Settings[/bold blue]",
        border_style="blue",
    ))
    console.print()
    
    # 创建任务表格
    table = Table(title="Task Configurations", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Name", style="cyan", min_width=20)
    table.add_column("Agent", style="yellow", width=10)
    table.add_column("Model", style="green", min_width=25)
    table.add_column("Eval Mode", style="blue", width=8)
    table.add_column("Context", justify="right", width=8)
    table.add_column("Prompts", style="dim", width=15)
    table.add_column("Dataset", style="dim")
    
    enabled_count = 0
    for i, task in enumerate(all_tasks):
        # 显示启用/禁用状态
        if task.enabled:
            status = "[green]✓ ON[/green]"
            enabled_count += 1
        else:
            status = "[dim]✗ OFF[/dim]"
        
        # 简化显示
        model_short = task.chat_model.split("/")[-1] if "/" in task.chat_model else task.chat_model
        dataset_short = task.benchmark_config.split("/")[-1]
        
        # Prompt 显示
        chat_p = task.chat_prompt or "-"
        eval_p = task.eval_prompt or "-"
        prompts_str = f"C:{chat_p}/E:{eval_p}"
        
        # Eval Mode 显示（用图标区分）
        eval_mode_display = "🎯" if task.eval_mode == "binary" else "📊"
        
        table.add_row(
            str(i + 1),
            status,
            task.name,
            task.memory_provider,
            model_short,
            eval_mode_display,
            str(task.context_window),
            prompts_str,
            dataset_short,
        )
    
    console.print(table)
    console.print()
    
    # 摘要统计
    disabled_count = len(all_tasks) - enabled_count
    summary = f"[bold]Total:[/bold] {len(all_tasks)} tasks"
    if enabled_count > 0:
        summary += f"  [green]✓ {enabled_count} enabled[/green]"
    if disabled_count > 0:
        summary += f"  [dim]✗ {disabled_count} disabled[/dim]"
    
    console.print(summary)
    
    if enabled_count == 0:
        console.print("[yellow]⚠ No tasks enabled. Nothing will run.[/yellow]")


def run():
    """
    程序运行入口
    
    负责初始化、分发到不同模式（list/run）、处理中断。
    """
    args = parse_args()
    
    # 初始化日志
    setup_logging(level=args.log_level)
    
    # 显示启动 Banner
    console = get_console()
    console.print()
    console.print(Panel(
        "[bold cyan]MemIndex Batch Runner[/bold cyan] - Parallel Benchmark Execution",
        border_style="cyan",
    ))
    console.print()
    
    # 如果是 --list 模式，只显示任务列表不执行
    if args.list_tasks:
        list_tasks(args)
        return
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        # 优雅处理 Ctrl+C 中断
        console.print()
        console.print(Panel(
            "[bold yellow]⚠ Batch run interrupted by user (Ctrl+C)[/bold yellow]\n"
            "[dim]Tasks in progress have been stopped. Completed tasks are saved.[/dim]",
            title="[yellow]Interrupted[/yellow]",
            border_style="yellow",
        ))


if __name__ == "__main__":
    run()
