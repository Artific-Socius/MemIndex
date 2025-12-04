"""
工具类集合
包含:
- StatisticsHelper: 统计工具
- ProgressManager: 进度管理 (支持Rich)
"""
from __future__ import annotations

import io
import sys
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel


class StatisticsHelper:
    """
    统计工具类
    提供描述性统计和直方图生成功能
    """
    
    @staticmethod
    def print_distribution_summary(data: list[float]) -> str:
        """
        打印并返回描述性统计摘要
        
        Args:
            data: 数据列表
            
        Returns:
            str: 统计摘要文本
        """
        if not data:
            return "No data for distribution summary."
        
        arr = np.array(data)
        
        summary = (
            f"\n--- 描述性统计 ---\n"
            f"样本数: {len(data)}\n"
            f"平均值: {np.mean(arr):.4f}\n"
            f"中位数: {np.median(arr):.4f}\n"
            f"标准差: {np.std(arr):.4f}\n"
            f"最小值: {np.min(arr):.4f}\n"
            f"最大值: {np.max(arr):.4f}\n"
            f"\n--- 四分位数 ---\n"
            f"25%分位: {np.percentile(arr, 25):.4f}\n"
            f"50%分位: {np.percentile(arr, 50):.4f}\n"
            f"75%分位: {np.percentile(arr, 75):.4f}\n"
        )
        
        print(summary)
        return summary
    
    @staticmethod
    def print_text_histogram_quantile(
        data: list[float],
        num_bins: int = 15,
        bar_char: str = "█",
        max_width: int = 100
    ) -> str:
        """
        打印基于分位数的文本直方图
        
        Args:
            data: 数据列表
            num_bins: bin数量
            bar_char: 柱状图字符
            max_width: 最大宽度
            
        Returns:
            str: 直方图文本
        """
        output_buffer = io.StringIO()
        
        def _manual_print(*args, end="\n"):
            text = " ".join(str(a) for a in args)
            sys.stdout.write(text + end)
            output_buffer.write(text + end)
        
        total_count = len(data)
        if total_count == 0:
            _manual_print("Error: 输入数据为空")
            return output_buffer.getvalue()
        
        arr = np.array(data)
        
        # 确保bins在合理范围
        num_bins = max(2, min(100, num_bins))
        
        # 计算分位数边界
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(arr, quantiles)
        
        # 处理重复边界
        unique_bin_edges = np.unique(bin_edges)
        
        if len(unique_bin_edges) <= 2:
            _manual_print("\nWarning: 数据过于集中，仅使用2个bins")
            unique_bin_edges = [arr.min(), arr.max()]
            if len(unique_bin_edges) == 1:
                _manual_print(f"Error: 所有值相同: {unique_bin_edges[0]:.4f}")
                return output_buffer.getvalue()
        
        counts, final_edges = np.histogram(arr, bins=unique_bin_edges, density=False)
        
        max_bar_count = counts.max() if counts.max() > 0 else 1
        final_num_bins = len(counts)
        
        _manual_print(f"\n--- 文本直方图 (分位数分箱) ---")
        _manual_print(f"--- (实际Bins: {final_num_bins}, 目标Bins: {num_bins}) ---")
        _manual_print("-" * (max_width + 45))
        _manual_print(f"Bin Range ({final_num_bins} Bins){'':<14} | Count | Bar")
        _manual_print("-" * (max_width + 45))
        
        for i in range(len(counts)):
            bar_length = int(counts[i] / max_bar_count * max_width)
            bar_char_local = bar_char if counts[i] > 0 else " "
            
            bin_start = final_edges[i]
            bin_end = final_edges[i + 1]
            bin_label = f"[{bin_start: .4f}, {bin_end})"
            
            print_line = f"{bin_label:<30} | {counts[i]:<5} | {bar_char_local * bar_length}"
            _manual_print(print_line)
        
        _manual_print("-" * (max_width + 45))
        
        return output_buffer.getvalue()


class ProgressManager:
    """
    基于Rich的进度管理器
    支持异步任务的进度展示
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        初始化进度管理器
        
        Args:
            total: 总任务数
            description: 描述文本
        """
        self.total = total
        self.description = description
        self.console = Console()
        
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._live: Optional[Live] = None
        
        # 统计信息
        self.completed = 0
        self.errors = 0
        self.accuracy = 0.0
        self.filter_accuracy = 0.0
        self.avg_logprobs = 0.0
    
    def __enter__(self) -> ProgressManager:
        """上下文管理器入口"""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("• [cyan]Acc: {task.fields[accuracy]:.1%}[/cyan]"),
            TextColumn("• [yellow]Err: {task.fields[errors]}[/yellow]"),
            TextColumn("• [green]AvgLP: {task.fields[avg_lp]:.2f}[/green]"),
            console=self.console,
            expand=True,
        )
        self._task_id = self._progress.add_task(
            self.description,
            total=self.total,
            accuracy=0.0,
            errors=0,
            avg_lp=0.0,
        )
        self._progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self._progress:
            self._progress.stop()
    
    def advance(
        self,
        accuracy: float = 0.0,
        errors: int = 0,
        avg_logprobs: float = 0.0,
    ) -> None:
        """
        推进进度
        
        Args:
            accuracy: 当前准确率
            errors: 错误数
            avg_logprobs: 平均logprobs
        """
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                advance=1,
                accuracy=accuracy,
                errors=errors,
                avg_lp=avg_logprobs,
            )
            self.completed += 1
            self.accuracy = accuracy
            self.errors = errors
            self.avg_logprobs = avg_logprobs
    
    def update_stats(
        self,
        accuracy: float,
        filter_accuracy: float,
        errors: int,
        avg_logprobs: float,
    ) -> None:
        """仅更新统计信息，不推进进度"""
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                accuracy=accuracy,
                errors=errors,
                avg_lp=avg_logprobs,
            )
            self.accuracy = accuracy
            self.filter_accuracy = filter_accuracy
            self.errors = errors
            self.avg_logprobs = avg_logprobs


class AsyncProgressManager:
    """
    异步进度管理器
    用于并发任务的进度追踪
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.console = Console()
        
        # 使用Rich的Live来实现实时更新
        self._completed = 0
        self._errors = 0
        self._accuracy = 0.0
        self._filter_accuracy = 0.0
        self._avg_logprobs = 0.0
        self._filter_total = 0
    
    def create_status_table(self) -> Table:
        """创建状态表格"""
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("进度", f"{self._completed}/{self.total} ({self._completed/self.total*100:.1f}%)")
        table.add_row("准确率", f"{self._accuracy:.2%}")
        table.add_row("过滤准确率", f"{self._filter_accuracy:.2%} ({self._filter_total})")
        table.add_row("错误数", f"{self._errors}")
        table.add_row("平均LogProbs", f"{self._avg_logprobs:.4f}")
        
        return table
    
    def update(
        self,
        completed: int,
        accuracy: float,
        filter_accuracy: float,
        errors: int,
        avg_logprobs: float,
        filter_total: int,
    ) -> None:
        """更新统计信息"""
        self._completed = completed
        self._accuracy = accuracy
        self._filter_accuracy = filter_accuracy
        self._errors = errors
        self._avg_logprobs = avg_logprobs
        self._filter_total = filter_total
    
    def print_status(self) -> None:
        """打印当前状态"""
        progress_bar = "█" * int(self._completed / self.total * 40) + "░" * (40 - int(self._completed / self.total * 40))
        status = (
            f"\r[{progress_bar}] {self._completed}/{self.total} "
            f"| Acc: {self._accuracy:.1%} "
            f"| Filter: {self._filter_accuracy:.1%} ({self._filter_total}) "
            f"| Err: {self._errors} "
            f"| AvgLP: {self._avg_logprobs:.2f}"
        )
        print(status, end="", flush=True)
