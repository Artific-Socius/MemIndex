"""
Runner - 基准测试运行器

负责协调多个执行器(Actuator)的运行，是整个测试流程的核心调度器。

核心职责:
    1. 交替执行多个测试序列（如颜色测试、笑话测试交替进行）
    2. 管理"记忆距离"：在信息植入和提问之间插入废话对话
    3. 处理"冻结区"：控制各序列的执行时机
    4. 管理懒评估任务
    5. 追踪进度和统计信息

关键概念:
    - 记忆距离(memory_distance): 信息植入到提问之间需要的 token 数量
    - 冻结区(frozen_area): 暂时挂起的执行器，等待记忆距离达标
    - 废话填充(nonsense): 用于填充记忆距离的无关对话
    - 懒评估(lazy_score): 延迟到后续对话后再进行的评估
"""

from __future__ import annotations

import os
import random
import time
from enum import Enum
from typing import TYPE_CHECKING, Callable, Coroutine, Any

import tiktoken
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    ProgressColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.console import Group
from rich.text import Text

from core.actuator import Actuator, FakeActuator
from utils.nonsense_generator import filler_no_response_tokens_trivia
from utils.logging import get_console
from utils.litellm_controller import get_cost_tracker

if TYPE_CHECKING:
    from components.agents.base_agent import BaseAgent

# 获取 MemIndex 模块的根目录（用于定位数据文件）
_MEMINDEX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TokensColumn(ProgressColumn):
    """
    自定义进度条列：显示当前 Token 数量
    
    用于在 Rich 进度条中实时显示累计的 token 消耗。
    """
    
    def __init__(self, runner: "Runner"):
        super().__init__()
        self.runner = runner
    
    def render(self, task) -> Text:
        tokens = self.runner.current_tokens
        return Text(f"[{tokens:,} tokens]", style="cyan")


class CostColumn(ProgressColumn):
    """
    自定义进度条列：显示当前费用
    
    用于在 Rich 进度条中实时显示 API 调用费用。
    """
    
    def render(self, task) -> Text:
        tracker = get_cost_tracker()
        cost = tracker.total_cost
        if cost > 0:
            return Text(f"[${cost:.4f}]", style="yellow")
        return Text("[--]", style="dim")


class MemoryDistanceLevel(Enum):
    """
    记忆距离级别枚举
    
    定义如何计算和应用记忆距离：
    - TOTAL: 所有序列共享一个记忆距离（总计）
    - EACH_FIRST: 每个序列的第一个信息点单独计算记忆距离（推荐）
    - EACH_ALL: 每个序列的所有信息点都单独计算记忆距离
    """
    TOTAL = "total"
    EACH_FIRST = "each_first"
    EACH_ALL = "each_all"


class Runner:
    """
    基准测试运行器
    
    协调多个执行器的运行，管理废话填充和记忆距离。
    
    工作原理:
        1. 初始化时接收多个 Actuator（每个对应一个测试序列）
        2. 发送开场提示，告诉 Agent 这是记忆测试
        3. 主循环中交替执行各个 Actuator 的步骤
        4. 在需要时插入废话对话来填充记忆距离
        5. 管理冻结区，确保记忆距离达标后再继续执行
        6. 处理懒评估任务
    
    关键数据结构:
        - queue: 待执行的执行器队列
        - frozen_area: 冻结区，存储暂停的执行器和冻结时的 token 数
        - mark_queue: 标记队列，优先执行的评分步骤
        - lazy_score_task: 懒评估任务列表
    """
    
    def __init__(
        self,
        actuators: list[Actuator],
        nonsense: list[str],
        head_prompts: list[str],
        agent: "BaseAgent",
        memory_distance: int = 2048,
        memory_distance_level: MemoryDistanceLevel = MemoryDistanceLevel.EACH_FIRST,
        eval_model: str = "volcano/deepseek-v3-250324",
        show_progress: bool = True,
        progress_callback: callable = None,
        silent: bool = False,
    ):
        """
        初始化运行器
        
        Args:
            actuators: 执行器列表，每个执行器对应一个测试序列
            nonsense: 废话列表（如果为空会自动生成）
            head_prompts: 开头提示列表（告诉模型这是记忆测试）
            agent: 被测试的 Agent 实例
            memory_distance: 记忆距离，单位为 tokens
            memory_distance_level: 记忆距离计算级别
            eval_model: 用于评估的模型
            show_progress: 是否显示进度条
            progress_callback: 进度回调函数，用于外部进度更新
            silent: 静默模式，不输出任何内容
        """
        self.actuators = actuators
        self.nonsense = nonsense
        self.head_prompts = head_prompts
        self.nonsense_visited = []                # 已使用的废话（避免重复）
        self.memory_distance = memory_distance    # 目标记忆距离
        self.token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")  # token 计数器
        self.conversation_history = []            # 全局对话历史
        self.memory_distance_level = memory_distance_level
        self.agent = agent
        self.running_history = []                 # 运行历史，记录执行顺序
        self.target_nonsense = 20                 # 目标废话数量
        self.conversation_index = 0               # 当前对话轮次
        self.nonsense_possibility = 0.5           # 随机插入废话的概率
        self.eval_model = eval_model
        self.sub_test_log = []                    # 各执行器的 token 统计
        self.show_progress = show_progress and not silent
        self.progress_callback = progress_callback
        self.silent = silent
        self._start_time = 0
        
        # 统计信息
        self._stats = {
            "total_steps": 0,       # 总步数
            "nonsense_steps": 0,    # 废话步数
            "actuator_steps": 0,    # 执行器步数
            "lazy_scores": 0,       # 懒评估次数
            "llm_calls": 0,         # LLM 调用次数
        }
    
    async def run(self) -> None:
        """
        运行基准测试（主入口）
        
        这是 Runner 的主入口方法，负责：
        1. 重置统计信息
        2. 验证执行器配置
        3. 根据记忆距离级别选择运行策略
        """
        self._start_time = time.time()
        
        # 重置费用追踪器
        cost_tracker = get_cost_tracker()
        cost_tracker.reset()
        
        # 验证：确保每个执行器至少有一个评分点
        for actuator in self.actuators:
            if actuator.first_mark == -1:
                raise Exception("First mark cannot be -1")
        
        # 根据记忆距离级别选择运行策略
        if self.memory_distance_level in [
            MemoryDistanceLevel.EACH_FIRST, 
            MemoryDistanceLevel.EACH_ALL
        ]:
            await self._run_each()
        else:
            raise NotImplementedError
    
    @property
    def all_text(self) -> str:
        """
        获取所有对话文本
        
        将对话历史拼接成文本，用于计算 token 数量。
        
        Returns:
            拼接后的对话文本
        """
        texts = []
        for item in self.conversation_history:
            texts.append(f"{item['role']}:{item['content']}")
        return "\n".join(texts)
    
    @property
    def current_tokens(self) -> int:
        """
        计算当前总 token 数量
        
        包括全局对话历史和所有执行器的对话历史。
        
        Returns:
            当前累计的 token 数量
        """
        # 全局对话历史的 token
        tokens = len(self.token_encoder.encode(self.all_text))
        # 加上各执行器的 token
        for item in self.actuators:
            tokens += item.current_tokens
        return tokens
    
    def _create_progress(self, total: int) -> Progress:
        """
        创建 Rich 进度条
        
        配置一个包含多列信息的进度条：
        - 步骤进度
        - Token 数量
        - 费用
        - 耗时
        
        Args:
            total: 总步数
            
        Returns:
            配置好的 Progress 实例
        """
        return Progress(
            SpinnerColumn(),                              # 旋转动画
            TextColumn("[bold blue]{task.description}"),  # 任务描述
            BarColumn(bar_width=30),                      # 进度条
            TaskProgressColumn(),                         # 百分比
            MofNCompleteColumn(),                         # M/N 格式
            TextColumn("•"),
            TokensColumn(self),                           # 自定义: Token 数量
            TextColumn("•"),
            CostColumn(),                                 # 自定义: 费用
            TextColumn("•"),
            TimeElapsedColumn(),                          # 已用时间
            TextColumn("→"),
            TimeRemainingColumn(),                        # 剩余时间
            console=get_console(),
            transient=False,
            refresh_per_second=4,
        )
    
    def _create_stats_table(self) -> Table:
        """
        创建统计信息表格
        
        在测试完成后显示详细的统计信息。
        
        Returns:
            配置好的 Table 实例
        """
        cost_tracker = get_cost_tracker()
        elapsed = time.time() - self._start_time
        
        table = Table(title="Benchmark Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        # 步骤统计
        table.add_row("Total Steps", str(self._stats["total_steps"]))
        table.add_row("Actuator Steps", str(self._stats["actuator_steps"]))
        table.add_row("Nonsense Steps", str(self._stats["nonsense_steps"]))
        table.add_row("Lazy Scores", str(self._stats["lazy_scores"]))
        table.add_row("─" * 15, "─" * 10)
        
        # Token 统计
        table.add_row("Current Tokens", f"{self.current_tokens:,}")
        table.add_row("Conversation Index", str(self.conversation_index))
        table.add_row("─" * 15, "─" * 10)
        
        # LLM 调用统计
        table.add_row("LLM Calls", str(cost_tracker.call_count))
        table.add_row("Total Input Tokens", f"{cost_tracker.total_input_tokens:,}")
        table.add_row("Total Output Tokens", f"{cost_tracker.total_output_tokens:,}")
        table.add_row("Total Cost", f"${cost_tracker.total_cost:.4f}")
        table.add_row("─" * 15, "─" * 10)
        
        # 时间统计
        table.add_row("Elapsed Time", f"{elapsed:.2f}s")
        
        return table
    
    def _create_model_stats_table(self) -> Table | None:
        """
        创建模型使用统计表格
        
        显示各个模型的调用次数、token 消耗和费用。
        
        Returns:
            Table 实例，如果没有数据返回 None
        """
        cost_tracker = get_cost_tracker()
        
        if not cost_tracker.model_stats:
            return None
        
        table = Table(title="Model Usage", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="dim")
        table.add_column("Calls", justify="right")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cost", justify="right", style="yellow")
        
        for model, stats in cost_tracker.model_stats.items():
            # 缩短模型名称显示
            display_name = model.split("/")[-1] if "/" in model else model
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            
            table.add_row(
                display_name,
                str(stats["calls"]),
                f"{stats['input_tokens']:,}",
                f"{stats['output_tokens']:,}",
                f"${stats['cost']:.4f}",
            )
        
        return table
    
    async def _run_each(self) -> None:
        """
        按"每个执行器"模式运行
        
        这是主要的运行模式，实现：
        1. 发送开场提示
        2. 创建执行队列
        3. 启动主循环
        4. 显示完成统计
        """
        # 初始化各种队列
        queue: list[Actuator | FakeActuator] = [x for x in self.actuators]  # 待执行队列
        frozen_area: list[tuple[Actuator, int]] = []  # 冻结区：(执行器, 冻结时的token数)
        mark_queue: list[Actuator] = []               # 标记队列：优先执行的评分步骤
        lazy_score_task: list[tuple[Actuator, int, int, int]] = []  # 懒评估任务
        self.conversation_index = 0
        self.sub_test_log = [(0, 0) for _ in range(len(self.actuators))]  # 各执行器的token统计
        
        # 发送开场提示（告诉模型这是记忆测试）
        self.conversation_index += await self._send_head_prompts()
        for _ in range(len(self.head_prompts)):
            self.running_history.append(("nonsense", None, len(self.conversation_history) - 2))
        
        # 计算总步数（用于进度条）
        total_steps = sum([len(x.data) for x in self.actuators])
        console = get_console()
        
        if self.show_progress:
            # 使用 Rich 进度条显示
            progress = self._create_progress(total_steps)
            task_id = progress.add_task(
                "[cyan]Running Benchmark...", 
                total=total_steps
            )
            
            with progress:
                await self._run_loop(
                    queue, frozen_area, mark_queue, lazy_score_task,
                    progress=progress, task_id=task_id
                )
            
            # 显示完成统计
            console.print()
            stats_table = self._create_stats_table()
            model_table = self._create_model_stats_table()
            
            if model_table:
                # 并排显示两个表格
                layout = Table.grid(padding=1)
                layout.add_column()
                layout.add_column()
                layout.add_row(stats_table, model_table)
                
                console.print(Panel(
                    layout,
                    title="[bold green]✓ Benchmark Complete",
                    border_style="green",
                ))
            else:
                console.print(Panel(
                    stats_table,
                    title="[bold green]✓ Benchmark Complete",
                    border_style="green",
                ))
        else:
            # 静默模式：不显示进度条
            await self._run_loop(
                queue, frozen_area, mark_queue, lazy_score_task,
                progress=None, task_id=None
            )
    
    async def _run_loop(
        self,
        queue: list,
        frozen_area: list,
        mark_queue: list,
        lazy_score_task: list,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> None:
        """
        主运行循环
        
        这是整个测试的核心循环，实现复杂的调度逻辑：
        
        执行流程:
        1. 从队列取出一个执行器（优先从 mark_queue 取）
        2. 执行一步（发送消息、可能评分）
        3. 根据执行结果更新冻结区
        4. 处理懒评估任务
        5. 更新队列（可能插入废话）
        6. 重复直到所有执行器完成
        
        Args:
            queue: 待执行队列
            frozen_area: 冻结区
            mark_queue: 标记队列（优先执行）
            lazy_score_task: 懒评估任务列表
            progress: Rich 进度条实例
            task_id: 进度条任务 ID
        """
        while queue or mark_queue:
            # ========== 1. 取出执行器 ==========
            # 优先从 mark_queue 取（评分步骤优先）
            actuator = queue.pop(0) if len(mark_queue) == 0 else mark_queue.pop(0)
            
            # 如果是真正的执行器（非废话），记录全局 token 和索引
            if isinstance(actuator, Actuator):
                actuator.global_tokens.append(self.current_tokens)
                actuator.global_index.append(self.conversation_index)
                # 如果是该执行器的第一步，记录起始 token
                if len(actuator.global_tokens) == 1:
                    self.sub_test_log[self.actuators.index(actuator)] = (self.current_tokens, 0)
            
            # ========== 2. 执行一步 ==========
            response, activate, lazy_score, inner_index = await actuator.step()
            self._stats["total_steps"] += 1
            
            if isinstance(actuator, Actuator):
                # 真正的执行器步骤
                self._stats["actuator_steps"] += 1
                
                # 如果还有后续步骤，放入冻结区等待记忆距离达标
                if actuator.has_next:
                    frozen_area.append((actuator, self.current_tokens))
                else:
                    # 执行器完成，记录结束 token
                    idx = self.actuators.index(actuator)
                    self.sub_test_log[idx] = (self.sub_test_log[idx][0], self.current_tokens)
                
                # 记录运行历史
                self.running_history.append(
                    ("actuator", self.actuators.index(actuator), actuator.index)
                )
                
                # 处理懒评估任务
                if lazy_score:
                    self._stats["lazy_scores"] += 1
                    lazy_score_task.append((
                        actuator, 
                        inner_index, 
                        self.conversation_index,
                        actuator.data[inner_index].score.lazy_count - 1
                    ))
                
                # 更新进度条描述
                if progress and task_id is not None:
                    actuator_idx = self.actuators.index(actuator)
                    cost_tracker = get_cost_tracker()
                    progress.update(
                        task_id, 
                        description=f"[cyan]Act {actuator_idx + 1}/{len(self.actuators)} Step {actuator.index}/{len(actuator.data)}"
                    )
                
                # 调用外部进度回调
                if self.progress_callback:
                    total_steps = sum(len(a.data) for a in self.actuators)
                    cost_tracker = get_cost_tracker()
                    actuator_idx = self.actuators.index(actuator)
                    actuator_name = getattr(actuator, 'name', None) or f"Actuator {actuator_idx + 1}"
                    
                    self.progress_callback(
                        current_step=self._stats["total_steps"],
                        total_steps=total_steps,
                        tokens=self.current_tokens,
                        cost=cost_tracker.total_cost,
                        actuator_name=actuator_name,
                        actuator_step=actuator.index,
                        actuator_total=len(actuator.data),
                        last_action=actuator.data[inner_index].ask[:30] if inner_index < len(actuator.data) else "",
                    )
            else:
                # 废话步骤
                self._stats["nonsense_steps"] += 1
                self.running_history.append(
                    ("nonsense", None, len(self.conversation_history) - 2)
                )
                
                # 废话步骤也增加总进度
                if progress and task_id is not None:
                    progress.update(task_id, total=progress.tasks[task_id].total + 1)
            
            # ========== 3. 处理冻结区 ==========
            # 检查冻结区中的执行器是否可以解冻（记忆距离达标）
            remove_frozen_indexes = []
            for idx, (item, frozen_tokens) in enumerate(frozen_area):
                actuator_idx = self.actuators.index(item)
                start_tokens = (
                    item.global_tokens[0] 
                    if item.global_tokens 
                    else self.sub_test_log[actuator_idx][0]
                )
                
                # 计算进度比例，动态调整需要的记忆距离
                progress_ratio = (item.index + 1) / len(item.data)
                # 如果当前 token 数 - 起始 token 数 >= 记忆距离 * 进度比例，则解冻
                if (
                    self.current_tokens - start_tokens >= (self.memory_distance * progress_ratio)
                    or not item.has_next  # 或者执行器已完成
                ):
                    if item.has_next:
                        mark_queue.append(item)  # 放入优先队列
                    remove_frozen_indexes.append(idx)
            
            # 从冻结区移除已解冻的执行器
            frozen_area = [
                frozen_area[i] 
                for i in range(len(frozen_area)) 
                if i not in remove_frozen_indexes
            ]
            
            # ========== 4. 处理懒评估 ==========
            # 检查是否有懒评估任务达到触发条件
            for _actuator, inner_idx, outside_idx, lazy_count in lazy_score_task:
                if self.conversation_index - outside_idx == lazy_count:
                    await _actuator.execute_lazy_score(inner_idx, response)
            
            # 移除已执行的懒评估任务
            lazy_score_task = [
                item 
                for item in lazy_score_task 
                if self.conversation_index - item[2] != item[3]
            ]
            
            # ========== 5. 更新队列 ==========
            # 如果队列为空，重新填充未完成的执行器
            if len(queue) == 0:
                queue = [
                    x for x in self.actuators 
                    if x.has_next 
                    and not x.is_finished 
                    and x not in mark_queue 
                    and x not in [a[0] for a in frozen_area]
                ]
            
            # 检查是否所有任务都完成
            if all(not isinstance(x, Actuator) for x in queue) and len(mark_queue) == 0 and len(frozen_area) == 0:
                break
            
            # ========== 6. 插入废话 ==========
            # 如果队列为空但冻结区有执行器，需要插入废话来填充记忆距离
            if len(queue) == 0 and len(mark_queue) == 0 and len(frozen_area) > 0:
                queue.append(FakeActuator(self._send_nonsense))
            else:
                # 随机插入废话（当还有较多记忆距离需要填充时）
                if (
                    all(x.index > 0 for x in self.actuators)  # 所有执行器都已开始
                    and max(self.memory_distance - self.current_tokens, 0) / self.memory_distance > 0.15  # 还有15%以上距离
                    and self.memory_distance - self.current_tokens > 0
                ):
                    if random.randint(0, 100) < self.nonsense_possibility * 100:
                        queue.insert(0, FakeActuator(self._send_nonsense))
            
            # 更新进度条
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            
            self.conversation_index += 1
    
    async def _send_head_prompts(self) -> int:
        """
        发送开场提示
        
        发送预设的开场提示，告诉 Agent 这是一个记忆测试。
        例如："I'm going to give you a long-term memory benchmark test..."
        
        Returns:
            发送的提示数量
        """
        for message in self.head_prompts:
            self.conversation_history.append({"role": "user", "content": message})
            answer = await self.agent.send_message(message)
            self.conversation_history.append({"role": "assistant", "content": answer})
        return len(self.head_prompts)
    
    async def _send_nonsense(self) -> tuple[str, bool, bool, int]:
        """
        发送废话
        
        发送一条无关的对话来填充记忆距离。
        废话内容从 nonsense.json 自动生成。
        
        Returns:
            (响应, True, False, -1) - 与 Actuator.step() 返回格式一致
        """
        nonsense = await self._auto_generate_nonsense()
        self.nonsense_visited.append(nonsense)
        self.conversation_history.append({"role": "user", "content": nonsense})
        answer = await self.agent.send_message(nonsense)
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer, True, False, -1
    
    async def _auto_generate_nonsense(self) -> str:
        """
        自动生成废话
        
        根据记忆距离自动生成适当长度的无关对话内容。
        使用 utils/nonsense_generator.py 生成随机知识问答。
        
        Returns:
            生成的废话文本
        """
        min_token = 20.0
        # 废话长度为记忆距离的5%，至少20个token
        tokens = self.memory_distance * 0.05
        tokens = max(min_token, tokens)
        
        # 从 nonsense.json 加载随机问答
        nonsense_path = os.path.join(_MEMINDEX_ROOT, "data", "nonsense.json")
        message, _ = filler_no_response_tokens_trivia(
            int(tokens), 
            10240, 
            token_len_function=lambda x: len(self.token_encoder.encode(x)),
            data_path=nonsense_path
        )
        return message
