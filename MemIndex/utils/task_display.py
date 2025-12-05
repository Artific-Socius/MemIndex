"""
TaskDisplay - 多任务显示模块

提供多任务并行运行时的 Rich 控制台显示功能。

核心功能:
    1. 实时显示多个并行任务的进度
    2. 自适应终端宽度的响应式布局
    3. 显示 Token 使用量、费用、预估时间等信息
    4. 任务状态追踪（等待/运行/完成/失败）

组件结构:
    - TaskStatus: 任务状态枚举
    - TaskState: 单个任务的状态数据
    - TaskOutputCapture: 任务输出捕获器
    - MultiTaskDisplay: 多任务显示器（主类）

显示布局:
    ┌────────────────────────────────────────┐
    │          📊 Batch Run Status           │  <- 全局信息面板
    │  Total: 10  Running: 3  Completed: 5   │
    └────────────────────────────────────────┘
    
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ 🔄 Task A    │  │ 🔄 Task B    │  │ 🔄 Task C    │  <- 运行中任务
    │ [████░░] 60% │  │ [██░░░░] 30% │  │ [█░░░░░] 15% │
    │ 5000 tokens  │  │ 2000 tokens  │  │ 1000 tokens  │
    └──────────────┘  └──────────────┘  └──────────────┘

使用方式:
    display = MultiTaskDisplay()
    display.register_task("task1", "My Task", total_steps=100)
    display.start()
    
    # 更新进度
    display.update_task("task1", current_step=50, current_tokens=5000)
    display.refresh()
    
    display.stop()
    display.print_final_summary()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from collections import deque

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeElapsedColumn,
    ProgressColumn,
    Task as ProgressTask,
)
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich.align import Align
from loguru import logger

from .litellm_controller import CostTracker


class TaskStatus(Enum):
    """
    任务状态枚举
    
    表示任务在生命周期中的不同阶段。
    
    状态流转:
        PENDING -> RUNNING -> COMPLETED
                          -> FAILED
                          -> CANCELLED
    """
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 正在执行
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"        # 执行失败
    CANCELLED = "cancelled"  # 已取消


def _format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"


def _format_cost(cost: float, prefix: str = "$") -> str:
    """
    格式化费用显示
    
    - 正常情况：$0.0500
    - 极小值（小数点后4位都是0）：$1.23e-06
    
    Args:
        cost: 费用金额
        prefix: 货币符号前缀（默认 $）
        
    Returns:
        格式化后的费用字符串
    """
    if cost == 0:
        return f"{prefix}0.0000"
    
    # 检查小数点后4位是否都是0（即值 < 0.00005）
    if abs(cost) < 0.00005:
        # 使用科学计数法
        return f"{prefix}{cost:.2e}"
    else:
        # 使用普通4位小数
        return f"{prefix}{cost:.4f}"


@dataclass
class TaskState:
    """
    单个任务的状态
    
    存储任务执行过程中的所有状态信息，包括：
    - 基本信息（ID、名称、状态）
    - 进度信息（步骤、Token、费用）
    - 时间信息（开始、结束、耗时）
    - 延迟统计（memory、chat）
    - 日志信息（最近的日志消息）
    
    Attributes:
        task_id: 任务唯一标识
        name: 任务显示名称
        status: 当前状态
        progress: 进度百分比 (0.0 - 1.0)
        total_steps: 总步骤数
        current_step: 当前步骤
        current_tokens: 已使用的 Token 数
        cost: 已产生的费用
        start_time: 开始时间戳
        end_time: 结束时间戳
        error: 错误信息
        logs: 最近的日志消息队列（最多5条）
        cost_tracker: 费用追踪器
        current_actuator: 当前执行的 Actuator 名称
        current_actuator_step: Actuator 当前步骤
        total_actuator_steps: Actuator 总步骤数
        chat_model: Chat 模型名称（用于预估费用）
        has_memory_backend: 是否有 memory 后端
        avg_memory_latency: 平均 memory 延迟（秒）
        avg_chat_latency: 平均 chat 延迟（秒）
        last_memory_latency: 最后一次 memory 延迟（秒）
        last_chat_latency: 最后一次 chat 延迟（秒）
    """
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    total_steps: int = 0
    current_step: int = 0
    current_tokens: int = 0
    cost: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    logs: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # 费用追踪器（每个任务独立）
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    
    # 当前处理的 Actuator 详细信息
    current_actuator: str = ""
    current_actuator_step: int = 0
    total_actuator_steps: int = 0
    
    # 模型信息（用于预估费用）
    chat_model: str = ""
    
    # 延迟统计
    has_memory_backend: bool = False  # 是否有 memory 后端
    avg_memory_latency: float = 0.0   # 平均 memory 延迟（秒）
    avg_chat_latency: float = 0.0     # 平均 chat 延迟（秒）
    last_memory_latency: float = 0.0  # 最后一次 memory 延迟（秒）
    last_chat_latency: float = 0.0    # 最后一次 chat 延迟（秒）
    
    def add_log(self, message: str):
        """
        添加日志消息
        
        Args:
            message: 日志消息内容
        """
        self.logs.append(message)
    
    @property
    def elapsed_time(self) -> float:
        """
        已用时间（秒）
        
        如果任务未开始返回 0，
        如果任务未结束则计算到当前时间。
        """
        if self.start_time == 0:
            return 0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time
    
    @property
    def estimated_remaining(self) -> float:
        """
        预估剩余时间（秒）
        
        基于当前进度和已用时间线性估算。
        """
        if self.current_step <= 0 or self.elapsed_time <= 0:
            return 0
        
        # 基于当前进度估算
        progress_ratio = self.current_step / self.total_steps if self.total_steps > 0 else 0
        if progress_ratio <= 0:
            return 0
        
        total_estimated = self.elapsed_time / progress_ratio
        remaining = total_estimated - self.elapsed_time
        return max(0, remaining)
    
    @property
    def speed(self) -> float:
        """
        处理速度 (steps/min)
        
        返回每分钟处理的步骤数。
        """
        if self.elapsed_time <= 0:
            return 0
        return (self.current_step / self.elapsed_time) * 60
    
    @property
    def status_icon(self) -> str:
        """
        状态对应的 emoji 图标
        
        Returns:
            状态对应的 emoji
        """
        icons = {
            TaskStatus.PENDING: "⏳",    # 等待
            TaskStatus.RUNNING: "🔄",    # 运行
            TaskStatus.COMPLETED: "✅",  # 完成
            TaskStatus.FAILED: "❌",     # 失败
            TaskStatus.CANCELLED: "🚫",  # 取消
        }
        return icons.get(self.status, "❓")


class TaskOutputCapture:
    """
    任务输出捕获器
    
    用于捕获单个任务的输出，实现任务间输出隔离。
    每个任务可以有自己的输出捕获器，互不影响。
    
    使用场景:
        在并行任务中，将每个任务的 print 输出
        重定向到对应的捕获器，避免输出混乱。
    
    Attributes:
        task_id: 任务 ID
        callback: 输出回调函数
        buffer: 输出缓冲区
    """
    
    def __init__(self, task_id: str, callback: Callable[[str, str], None] = None):
        """
        初始化输出捕获器
        
        Args:
            task_id: 任务 ID
            callback: 输出回调函数 (task_id, message) -> None
                每当有新输出时会调用此回调
        """
        self.task_id = task_id
        self.callback = callback
        self.buffer: List[str] = []
    
    def write(self, message: str):
        """
        写入消息到缓冲区
        
        会自动过滤空白消息，并触发回调。
        
        Args:
            message: 消息内容
        """
        if message.strip():
            self.buffer.append(message)
            if self.callback:
                self.callback(self.task_id, message)
    
    def get_recent(self, n: int = 5) -> List[str]:
        """
        获取最近的 n 条消息
        
        Args:
            n: 消息数量
            
        Returns:
            最近的消息列表
        """
        return self.buffer[-n:] if len(self.buffer) >= n else self.buffer


class MultiTaskDisplay:
    """
    多任务显示器
    
    使用 Rich Live 实时显示多个并行任务的状态。
    支持响应式布局，自动适应终端宽度。
    
    显示功能:
        - 全局状态面板（总任务数、运行中、已完成等）
        - 运行中任务的详细进度（进度条、Token、费用、时间）
        - 等待中任务的简要列表
        - 已完成任务的摘要
    
    使用方式:
        display = MultiTaskDisplay()
        display.register_task("task1", "Task Name", total_steps=100)
        display.start()
        
        # 在任务执行过程中更新
        display.update_task("task1", 
            status=TaskStatus.RUNNING,
            current_step=50,
            current_tokens=5000,
            cost=0.05
        )
        display.refresh()
        
        display.stop()
        display.print_final_summary()
    
    Attributes:
        console: Rich Console 实例
        tasks: 任务状态字典
        global_start_time: 全局开始时间
        global_cost: 全局累计费用
        total_tasks: 总任务数
        completed_tasks: 已完成任务数
        failed_tasks: 失败任务数
    """
    
    # 面板布局参数
    MIN_PANEL_WIDTH = 45   # 面板最小宽度
    MAX_PANEL_WIDTH = 55   # 面板最大宽度
    PANEL_PADDING = 2      # 面板之间的间距
    
    def __init__(self, console: Console = None):
        """
        初始化多任务显示器
        
        Args:
            console: Rich Console 实例（可选）
                如果不提供，会创建新的 Console
        """
        self.console = console or Console()
        self.tasks: Dict[str, TaskState] = {}
        self.global_start_time: float = 0
        self.global_cost: float = 0.0
        self.total_tasks: int = 0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self._live: Optional[Live] = None
        self._lock = asyncio.Lock()
        self._last_width: int = 0  # 用于检测终端宽度变化
        
        # 刷新节流控制
        self._last_refresh_time: float = 0.0  # 上次刷新时间
        self._min_refresh_interval: float = 0.15  # 最小刷新间隔（秒）
    
    def _calculate_estimated_cost(self) -> tuple[float, bool]:
        """
        计算预估总费用
        
        策略:
        1. 已完成任务：使用实际费用
        2. 正在运行任务：已用费用 + (已完成对话平均费用 × 剩余对话数)
        3. 未开始任务：
           - 优先使用同类模型（相同 chat_model）已完成任务的平均费用
           - 如果没有同类模型，使用所有已完成任务的平均费用
        
        Returns:
            (预估总费用, 是否有有效预估)
        """
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        running_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
        pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        
        # 如果没有已完成的任务，无法预估
        if not completed_tasks:
            return 0.0, False
        
        # 1. 计算已完成任务的实际费用
        completed_cost = sum(t.cost for t in completed_tasks)
        
        # 按模型分组计算平均费用
        model_costs: Dict[str, list[float]] = {}
        for t in completed_tasks:
            model = t.chat_model or "unknown"
            if model not in model_costs:
                model_costs[model] = []
            model_costs[model].append(t.cost)
        
        # 计算各模型的平均费用
        model_avg_costs: Dict[str, float] = {
            model: sum(costs) / len(costs) 
            for model, costs in model_costs.items()
        }
        
        # 所有已完成任务的平均费用（作为后备）
        overall_avg_cost = completed_cost / len(completed_tasks)
        
        # 2. 计算正在运行任务的预估剩余费用
        running_estimated = 0.0
        for t in running_tasks:
            if t.current_step > 0 and t.total_steps > 0:
                # 根据已完成步骤的平均费用预估剩余
                cost_per_step = t.cost / t.current_step
                remaining_steps = t.total_steps - t.current_step
                running_estimated += t.cost + (cost_per_step * remaining_steps)
            else:
                # 没有进度信息，使用同类模型或总体平均
                model = t.chat_model or "unknown"
                if model in model_avg_costs:
                    running_estimated += model_avg_costs[model]
                else:
                    running_estimated += overall_avg_cost
        
        # 3. 计算未开始任务的预估费用
        pending_estimated = 0.0
        for t in pending_tasks:
            model = t.chat_model or "unknown"
            # 优先使用同类模型的平均费用
            if model in model_avg_costs:
                pending_estimated += model_avg_costs[model]
            else:
                # 没有同类模型，使用总体平均
                pending_estimated += overall_avg_cost
        
        total_estimated = completed_cost + running_estimated + pending_estimated
        return total_estimated, True
    
    def _estimate_single_task_cost(self, task: TaskState) -> float:
        """
        计算单个任务的预估费用
        
        策略:
        - 优先使用同类模型（相同 chat_model）已完成任务的平均费用
        - 如果没有同类模型，使用所有已完成任务的平均费用
        - 如果没有已完成任务，返回 0
        
        Args:
            task: 要预估的任务
            
        Returns:
            预估费用
        """
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        
        if not completed_tasks:
            return 0.0
        
        # 按模型分组
        model_costs: Dict[str, list[float]] = {}
        for t in completed_tasks:
            model = t.chat_model or "unknown"
            if model not in model_costs:
                model_costs[model] = []
            model_costs[model].append(t.cost)
        
        # 优先使用同类模型的平均费用
        task_model = task.chat_model or "unknown"
        if task_model in model_costs:
            return sum(model_costs[task_model]) / len(model_costs[task_model])
        
        # 否则使用总体平均
        total_cost = sum(t.cost for t in completed_tasks)
        return total_cost / len(completed_tasks)
    
    def _get_layout_info(self) -> tuple[int, int]:
        """
        根据终端宽度计算最优布局
        
        自动计算每行能放多少个面板以及面板的宽度，
        以充分利用终端空间同时保持美观。
        
        算法:
            1. 计算最小宽度下能放多少面板
            2. 均匀分配剩余空间
            3. 限制面板宽度在 MIN_PANEL_WIDTH 和 MAX_PANEL_WIDTH 之间
        
        Returns:
            (每行面板数, 面板宽度)
        """
        terminal_width = self.console.width
        
        # 计算能放下多少个最小宽度的面板
        min_panels = terminal_width // (self.MIN_PANEL_WIDTH + self.PANEL_PADDING)
        
        if min_panels <= 1:
            # 只能放一个，使用最大可用宽度
            return 1, min(terminal_width - 4, self.MAX_PANEL_WIDTH)
        
        # 计算最优面板宽度（均匀分配）
        panel_width = (terminal_width - self.PANEL_PADDING * min_panels) // min_panels
        panel_width = max(self.MIN_PANEL_WIDTH, min(panel_width, self.MAX_PANEL_WIDTH))
        
        # 重新计算能放多少个
        panels_per_row = terminal_width // (panel_width + self.PANEL_PADDING)
        
        return max(1, panels_per_row), panel_width
    
    def register_task(self, task_id: str, name: str, total_steps: int = 0, chat_model: str = "") -> TaskState:
        """
        注册新任务
        
        在开始显示前，需要先注册所有任务。
        
        Args:
            task_id: 任务唯一标识
            name: 任务显示名称
            total_steps: 总步骤数（用于计算进度）
            chat_model: Chat 模型名称（用于预估费用）
            
        Returns:
            创建的任务状态对象
        """
        state = TaskState(
            task_id=task_id,
            name=name,
            total_steps=total_steps,
            chat_model=chat_model,
        )
        self.tasks[task_id] = state
        self.total_tasks += 1
        return state
    
    def update_task(
        self,
        task_id: str,
        status: TaskStatus = None,
        progress: float = None,
        current_step: int = None,
        current_tokens: int = None,
        cost: float = None,
        log_message: str = None,
        error: str = None,
        current_actuator: str = None,
        current_actuator_step: int = None,
        total_actuator_steps: int = None,
        has_memory_backend: bool = None,
        avg_memory_latency: float = None,
        avg_chat_latency: float = None,
        last_memory_latency: float = None,
        last_chat_latency: float = None,
    ):
        """
        更新任务状态
        
        所有参数都是可选的，只更新提供的字段。
        
        Args:
            task_id: 要更新的任务 ID
            status: 新状态
            progress: 进度 (0.0 - 1.0)
            current_step: 当前步骤
            current_tokens: 当前 Token 数
            cost: 当前费用
            log_message: 添加的日志消息
            error: 错误信息
            current_actuator: 当前 Actuator 名称
            current_actuator_step: Actuator 当前步骤
            total_actuator_steps: Actuator 总步骤
            has_memory_backend: 是否有 memory 后端
            avg_memory_latency: 平均 memory 延迟（秒）
            avg_chat_latency: 平均 chat 延迟（秒）
            last_memory_latency: 最后一次 memory 延迟（秒）
            last_chat_latency: 最后一次 chat 延迟（秒）
        """
        if task_id not in self.tasks:
            return
        
        state = self.tasks[task_id]
        
        if status is not None:
            old_status = state.status
            state.status = status
            
            if status == TaskStatus.RUNNING and old_status != TaskStatus.RUNNING:
                state.start_time = time.time()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                state.end_time = time.time()
                if status == TaskStatus.COMPLETED:
                    self.completed_tasks += 1
                elif status == TaskStatus.FAILED:
                    self.failed_tasks += 1
        
        if progress is not None:
            state.progress = progress
        if current_step is not None:
            state.current_step = current_step
        if current_tokens is not None:
            state.current_tokens = current_tokens
        if cost is not None:
            state.cost = cost
        if log_message is not None:
            state.add_log(log_message)
        if error is not None:
            state.error = error
        if current_actuator is not None:
            state.current_actuator = current_actuator
        if current_actuator_step is not None:
            state.current_actuator_step = current_actuator_step
        if total_actuator_steps is not None:
            state.total_actuator_steps = total_actuator_steps
        if has_memory_backend is not None:
            state.has_memory_backend = has_memory_backend
        if avg_memory_latency is not None:
            state.avg_memory_latency = avg_memory_latency
        if avg_chat_latency is not None:
            state.avg_chat_latency = avg_chat_latency
        if last_memory_latency is not None:
            state.last_memory_latency = last_memory_latency
        if last_chat_latency is not None:
            state.last_chat_latency = last_chat_latency
    
    def _create_task_panel(self, state: TaskState, show_estimate: bool = False) -> Panel:
        """
        创建单个任务的详细面板（用于运行中的任务）
        
        面板内容包括:
            - 进度条和百分比
            - 已用时间和预估剩余时间
            - Token 使用量和费用
            - 当前执行的 Actuator
            - 最近的日志消息
            - (可选) 预估费用（用于 Pending 任务）
        
        Args:
            state: 任务状态
            show_estimate: 是否显示预估费用（用于 Pending 任务）
            
        Returns:
            Rich Panel 组件
        """
        # 状态颜色
        status_colors = {
            TaskStatus.PENDING: "dim",
            TaskStatus.RUNNING: "cyan",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.CANCELLED: "yellow",
        }
        color = status_colors.get(state.status, "white")
        
        # 创建内容组件
        renderables: list[RenderableType] = []
        
        # 使用 Rich Progress 创建进度条
        if state.total_steps > 0:
            # 确保不超过 100%
            completed = min(state.current_step, state.total_steps)
            
            # 创建一个临时的 Progress 对象用于渲染
            progress = Progress(
                TextColumn("[bold]{task.percentage:>3.0f}%"),
                BarColumn(bar_width=20, style="bar.back", complete_style="bar.complete", finished_style="bar.finished"),
                TextColumn("[cyan]{task.completed}[/cyan]/[dim]{task.total}[/dim]"),
                expand=False,
                console=self.console,
            )
            # 添加任务并设置进度
            progress.add_task("", total=state.total_steps, completed=completed)
            
            # 渲染 Progress 对象
            renderables.append(progress)
        
        # 时间信息行（已用时间 + 预估剩余）
        elapsed = state.elapsed_time
        remaining = state.estimated_remaining
        speed = state.speed
        
        time_info = Text()
        time_info.append("⏱ ", style="dim")
        time_info.append(_format_time(elapsed), style="cyan")
        
        if remaining > 0 and state.status == TaskStatus.RUNNING:
            time_info.append(" → ", style="dim")
            time_info.append(f"~{_format_time(remaining)}", style="yellow")
            time_info.append(" left", style="dim")
        
        if speed > 0:
            time_info.append(f"  ({speed:.1f}/min)", style="dim")
        
        renderables.append(time_info)
        
        # Token 和费用信息
        stats_info = Text()
        stats_info.append("📊 ", style="dim")
        stats_info.append(f"{state.current_tokens:,}", style="cyan")
        stats_info.append(" tokens", style="dim")
        stats_info.append("  💰 ", style="dim")
        
        # 对于 Pending 任务，显示预估费用
        if show_estimate and state.status == TaskStatus.PENDING:
            estimated = self._estimate_single_task_cost(state)
            if estimated > 0:
                stats_info.append(_format_cost(estimated, "~"), style="yellow italic")
                stats_info.append(" (est.)", style="dim")
            else:
                stats_info.append("--", style="dim")
        else:
            stats_info.append(_format_cost(state.cost), style="yellow")
        renderables.append(stats_info)
        
        # 延迟统计信息（仅运行中显示）
        if state.status == TaskStatus.RUNNING and (state.avg_chat_latency > 0 or state.avg_memory_latency > 0):
            latency_info = Text()
            latency_info.append("⚡ ", style="dim")
            
            # Chat 延迟（始终显示）
            latency_info.append("Chat: ", style="dim")
            latency_info.append(f"{state.avg_chat_latency:.1f}s", style="magenta")
            latency_info.append(f"/{state.last_chat_latency:.1f}s", style="dim magenta")
            
            # Memory 延迟（仅有 memory 后端时显示）
            if state.has_memory_backend:
                latency_info.append("  Mem: ", style="dim")
                latency_info.append(f"{state.avg_memory_latency:.1f}s", style="blue")
                latency_info.append(f"/{state.last_memory_latency:.1f}s", style="dim blue")
            
            renderables.append(latency_info)
        
        # 分隔线
        renderables.append(Text("─" * 44, style="dim"))
        
        # 当前处理信息
        if state.current_actuator and state.status == TaskStatus.RUNNING:
            actuator_info = Text()
            actuator_info.append("▶ ", style="green")
            actuator_info.append(state.current_actuator, style="bold white")
            if state.total_actuator_steps > 0:
                actuator_info.append(f" [{state.current_actuator_step}/{state.total_actuator_steps}]", style="dim")
            renderables.append(actuator_info)
        
        # 最近日志
        log_count = 3 if not state.current_actuator else 2
        if state.logs:
            for log in list(state.logs)[-log_count:]:
                # 截断长日志
                if len(log) > 42:
                    log = log[:39] + "..."
                renderables.append(Text(f"  {log}", style="dim italic"))
        else:
            # 空占位
            for _ in range(log_count):
                renderables.append(Text("", style="dim"))
        
        # 错误信息
        if state.error:
            error_text = state.error[:44] + "..." if len(state.error) > 44 else state.error
            renderables.append(Text(f"⚠ {error_text}", style="bold red"))
        
        content = Group(*renderables)
        
        _, panel_width = self._get_layout_info()
        
        return Panel(
            content,
            title=f"{state.status_icon} [{color}]{state.name}[/{color}]",
            border_style=color,
            width=panel_width,
            height=14,  # 增加高度以容纳延迟统计行
        )
    
    def _create_global_panel(self) -> Panel:
        """
        创建全局信息面板
        
        显示批量运行的整体状态，包括:
            - 总任务数
            - 运行中/等待中/已完成/失败数量
            - 总 Token 使用量
            - 总费用（实际 + 预估）
            - 总耗时
        
        Returns:
            Rich Panel 组件
        """
        elapsed = time.time() - self.global_start_time if self.global_start_time > 0 else 0
        
        # 计算总费用
        total_cost = sum(t.cost for t in self.tasks.values())
        total_tokens = sum(t.current_tokens for t in self.tasks.values())
        
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        
        # 计算预估总费用
        estimated_cost, has_estimate = self._calculate_estimated_cost()
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row(
            "Total Tasks", str(self.total_tasks),
            "Running", str(running),
        )
        table.add_row(
            "Completed", str(self.completed_tasks),
            "Pending", str(pending),
        )
        table.add_row(
            "Failed", str(self.failed_tasks),
            "Elapsed", _format_time(elapsed),
        )
        table.add_row(
            "Total Tokens", f"{total_tokens:,}",
            "Total Cost", _format_cost(total_cost),
        )
        
        # 显示预估费用（只有在有已完成任务时才显示）
        if has_estimate and (running > 0 or pending > 0):
            table.add_row(
                "",
                "",
                "[bold yellow]Est. Total[/bold yellow]",
                f"[bold yellow]{_format_cost(estimated_cost, '~$')}[/bold yellow]",
            )
        
        return Panel(
            table,
            title="[bold blue]📊 Batch Run Status[/bold blue]",
            border_style="blue",
        )
    
    def _create_display(self) -> Group:
        """
        创建完整的显示内容
        
        组合所有面板形成最终显示:
            1. 全局状态面板
            2. 运行中任务（详细面板）
            3. 等待中任务（详细面板，最多显示一行）
            4. 已完成任务（简化面板）
        
        Returns:
            Rich Group 组件
        """
        # 全局面板
        global_panel = self._create_global_panel()
        
        # 任务面板（按状态分组）
        running_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
        pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        
        # 动态计算布局
        panels_per_row, _ = self._get_layout_info()
        
        renderables: list[RenderableType] = [global_panel]
        
        # 1. 运行中的任务
        if running_tasks:
            running_panels = [self._create_task_panel(task) for task in running_tasks]
            running_rows = self._create_panel_grid(running_panels, panels_per_row)
            renderables.extend(running_rows)
        
        # 2. 等待中的任务
        if pending_tasks:
            # 计算 Pending 任务的预估总费用
            pending_estimated = sum(self._estimate_single_task_cost(t) for t in pending_tasks)
            has_estimate = any(self._estimate_single_task_cost(t) > 0 for t in pending_tasks)
            
            if has_estimate:
                separator = Text(f"─── ⏳ Pending ({len(pending_tasks)}) | Est. {_format_cost(pending_estimated, '~$')} ───", style="dim yellow", justify="center")
            else:
                separator = Text(f"─── ⏳ Pending ({len(pending_tasks)}) ───", style="dim yellow", justify="center")
            renderables.append(separator)
            
            pending_panels = [self._create_task_panel(task, show_estimate=True) for task in pending_tasks[:panels_per_row]]
            pending_rows = self._create_panel_grid(pending_panels, panels_per_row)
            renderables.extend(pending_rows)
        
        # 3. 已完成的任务（使用简化的小面板）
        finished_tasks = completed_tasks + failed_tasks
        if finished_tasks:
            # 计算已完成任务的总费用
            finished_cost = sum(t.cost for t in finished_tasks)
            separator = Text(f"─── ✅ Completed ({len(finished_tasks)}) | {_format_cost(finished_cost)} ───", style="dim green", justify="center")
            
            finished_panels = [self._create_mini_task_panel(task) for task in finished_tasks]
            finished_rows = self._create_panel_grid(finished_panels, panels_per_row)
            renderables.extend(finished_rows)
        
        return Group(*renderables)
    
    def _create_panel_grid(self, panels: list[Panel], panels_per_row: int) -> list[Table]:
        """
        创建面板网格布局
        
        将多个面板排列成网格，每行显示指定数量的面板。
        
        Args:
            panels: 面板列表
            panels_per_row: 每行面板数
            
        Returns:
            Table 列表，每个 Table 代表一行
        """
        rows = []
        for i in range(0, len(panels), panels_per_row):
            row_panels = panels[i:i + panels_per_row]
            row_table = Table.grid(padding=1)
            for _ in row_panels:
                row_table.add_column()
            row_table.add_row(*row_panels)
            rows.append(row_table)
        return rows
    
    def _create_mini_task_panel(self, state: TaskState) -> Panel:
        """
        创建简化的任务面板（用于已完成的任务）
        
        只显示最关键的信息:
            - 步骤进度
            - Token 数
            - 费用
            - 耗时
        
        Args:
            state: 任务状态
            
        Returns:
            Rich Panel 组件（高度仅为3行）
        """
        # 状态颜色
        if state.status == TaskStatus.COMPLETED:
            color = "green"
            icon = "✅"
        else:
            color = "red"
            icon = "❌"
        
        # 简化的信息显示（使用 Table.grid 来更好地控制布局）
        info_table = Table.grid(padding=(0, 1))
        info_table.add_column(style="dim", no_wrap=True)
        info_table.add_column(style="cyan", no_wrap=True)
        info_table.add_column(style="yellow", no_wrap=True)
        info_table.add_column(style="magenta", no_wrap=True)
        
        info_table.add_row(
            f"{state.current_step}/{state.total_steps}",
            f"{state.current_tokens:,} tok",
            _format_cost(state.cost),
            _format_time(state.elapsed_time),
        )
        
        _, panel_width = self._get_layout_info()
        
        return Panel(
            info_table,
            title=f"{icon} [{color}]{state.name}[/{color}]",
            border_style=color,
            width=panel_width,
            height=3,
        )
    
    def start(self):
        """
        开始实时显示
        
        启动 Rich Live，开始在终端实时更新显示内容。
        """
        self.global_start_time = time.time()
        self._last_width = self.console.width
        self._live_start_line = self._get_cursor_line()
        self._live = Live(
            self._create_display(),
            console=self.console,
            refresh_per_second=4,  # 刷新率
            transient=False,
            vertical_overflow="crop",  # 使用裁剪而非溢出，减少闪烁
        )
        self._live.start()
    
    def _get_cursor_line(self) -> int:
        """获取当前光标行（近似值）"""
        # 这是一个近似值，用于跟踪输出位置
        return 0
    
    def _clear_live_area(self):
        """清除 Live 显示区域"""
        # 使用 ANSI 转义序列清除从当前位置到屏幕底部的内容
        # \033[J - 清除从光标到屏幕底部
        # \033[H - 移动光标到起始位置
        import sys
        
        # 先停止 Live
        if self._live:
            self._live.stop()
        
        # 清除屏幕（从当前位置向下）
        # 移动到 Live 开始的大概位置，然后清除到底部
        sys.stdout.write("\033[J")  # 清除从光标到屏幕底部
        sys.stdout.flush()
    
    def refresh(self, force: bool = False):
        """
        刷新显示内容
        
        检测终端宽度变化，必要时重新创建 Live。
        更新所有面板的显示内容。
        
        使用节流机制防止过于频繁的刷新导致闪烁。
        
        Args:
            force: 是否强制刷新（忽略节流）
        """
        if not self._live:
            return
        
        # 节流检查：避免过于频繁的刷新导致闪烁
        current_time = time.time()
        if not force and (current_time - self._last_refresh_time) < self._min_refresh_interval:
            return  # 跳过这次刷新
        self._last_refresh_time = current_time
        
        current_width = self.console.width
        
        # 检测终端宽度变化
        if current_width != self._last_width:
            self._last_width = current_width
            
            # 清除当前 Live 区域的残影
            self._clear_live_area()
            
            # 重新创建 Live
            self._live = Live(
                self._create_display(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
                vertical_overflow="crop",  # 使用裁剪而非溢出，减少闪烁
            )
            self._live.start()
        else:
            self._live.update(self._create_display())
    
    def stop(self):
        """
        停止实时显示
        
        停止 Rich Live，释放终端。
        """
        if self._live:
            self._live.stop()
            self._live = None
    
    def print_final_summary(self, interrupted: bool = False):
        """
        打印最终摘要
        
        在所有任务完成后打印汇总报告，包括:
            - 每个任务的详细信息
            - 总 Token 使用量
            - 总费用
            - 总耗时
        
        Args:
            interrupted: 是否被中断（影响显示样式）
        """
        elapsed = time.time() - self.global_start_time
        total_cost = sum(t.cost for t in self.tasks.values())
        total_tokens = sum(t.current_tokens for t in self.tasks.values())
        
        # 创建摘要表格
        table = Table(title="Batch Run Summary", show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Steps", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Time", justify="right")
        
        for task in self.tasks.values():
            status_str = f"{task.status_icon} {task.status.value}"
            table.add_row(
                task.name,
                status_str,
                f"{task.current_step}/{task.total_steps}",
                f"{task.current_tokens:,}",
                _format_cost(task.cost),
                f"{task.elapsed_time:.1f}s",
            )
        
        # 添加总计行
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{self.completed_tasks}/{self.total_tasks}[/bold]",
            "",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold]{_format_cost(total_cost)}[/bold]",
            f"[bold]{elapsed:.1f}s[/bold]",
        )
        
        self.console.print()
        
        if interrupted:
            # 中断时的摘要
            self.console.print(Panel(
                table,
                title="[bold yellow]⚠ Batch Run Interrupted[/bold yellow]",
                border_style="yellow",
            ))
        else:
            # 正常完成的摘要
            self.console.print(Panel(
                table,
                title="[bold green]✓ Batch Run Complete[/bold green]",
                border_style="green",
            ))

