"""
日志和数据写入管理
包含:
- Timer: 计时器
- ExperimentLogger: 实验日志管理
- AsyncResultWriter: 异步结果写入器
- RichLogHandler: Rich日志后端
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.text import Text

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from utils.models_utils import TokenUsage

from i18n import t


class RichLogHandler:
    """
    Rich日志处理器
    将loguru日志输出到Rich控制台，保留颜色样式
    """
    
    # loguru级别到Rich样式的映射
    LEVEL_STYLES = {
        "TRACE": "dim",
        "DEBUG": "dim cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    def write(self, message: str) -> None:
        """处理loguru消息"""
        record = message.record
        level_name = record["level"].name
        
        # 构建时间部分
        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 构建级别部分
        level_style = self.LEVEL_STYLES.get(level_name, "")
        
        # 构建位置部分
        module = record["name"]
        function = record["function"]
        line = record["line"]
        
        # 获取原始消息
        msg = str(record["message"])
        
        # 构建消息
        text = Text()
        text.append(f"{time_str} ", style="dim")
        text.append("| ", style="dim")
        text.append(f"{level_name:<8}", style=level_style)
        text.append(" | ", style="dim")
        text.append(f"{module}:{function}:{line}", style="cyan dim")
        text.append(" - ", style="dim")
        
        # 检查消息是否包含Rich标记 (用于带颜色的日志)
        if msg.startswith("[") and "]" in msg:
            # 使用Rich的markup解析
            text.append_text(Text.from_markup(msg))
        else:
            text.append(msg)
        
        self.console.print(text, highlight=False)


class Timer:
    """简单的计时器类，支持多种打印方式和链式调用"""

    def __init__(self, name: str = "Timer", auto_start: bool = False, print_when_start: bool = False) -> None:
        """初始化计时器

        Args:
            name: 计时器名称
            auto_start: 是否自动开始计时，默认False
            print_when_start: 是否在开始时打印日志，默认False
        """
        self.name: str = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed: float | None = None
        self.start_cpu_time: float | None = None
        self.end_cpu_time: float | None = None
        self.cpu_elapsed: float | None = None
        self.print_when_start: bool = print_when_start
        if auto_start:
            self.start()

    def start(self, new_name: str | None = None) -> Timer:
        """开始计时

        Args:
            new_name: 可选的新名称，如果提供则更新计时器名称

        Returns:
            Timer: 返回自身以支持链式调用
        """
        if new_name is not None:
            self.name = new_name
        self.start_time = time.time()
        self.start_cpu_time = time.process_time()
        self.end_time = None
        self.elapsed = None
        self.end_cpu_time = None
        self.cpu_elapsed = None
        if self.print_when_start:
            logger.debug(f"[{self.name}] 开始计时")
        return self

    def stop(self) -> Timer:
        """单独结束计时，不打印

        Returns:
            Timer: 返回自身以支持链式调用
        """
        if self.end_time is None:
            self.end_time = time.time()
            self.end_cpu_time = time.process_time()
            self.elapsed = self.end_time - self.start_time
            self.cpu_elapsed = self.end_cpu_time - self.start_cpu_time
        return self

    def _format_time(self, elapsed: float) -> str:
        """格式化时间，自动选择合适的时间单位

        Args:
            elapsed: 耗时（秒）

        Returns:
            str: 格式化的时间字符串
        """
        # 自适应单位选择
        if elapsed < 1e-6:  # 小于1微秒
            return f"{elapsed * 1e9:.2f}ns"
        elif elapsed < 1e-3:  # 小于1毫秒
            return f"{elapsed * 1e6:.2f}μs"
        elif elapsed < 1:  # 小于1秒
            return f"{elapsed * 1000:.2f}ms"
        elif elapsed < 60:  # 小于60秒
            return f"{elapsed:.2f}s"
        elif elapsed < 3600:  # 小于1小时
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.2f}s"
        else:  # 大于等于1小时
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{hours}h {minutes}m {seconds:.2f}s"

    def get_format(self) -> str:
        """获取格式化的时间字符串，自动选择合适的时间单位

        Returns:
            str: 格式化的耗时信息，包含墙上时间和CPU时间
        """
        if self.elapsed is None:
            self.stop()

        wall_time_str = self._format_time(self.elapsed)
        cpu_time_str = self._format_time(self.cpu_elapsed)
        
        return f"[{self.name}] 耗时: {wall_time_str} (CPU: {cpu_time_str})"

    def stop_and_print(self) -> Timer:
        """结束计时并立即打印

        Returns:
            Timer: 返回自身以支持链式调用
        """
        self.stop()
        logger.debug(self.get_format())
        return self


class ExperimentLogger:
    """
    实验日志管理器
    
    负责:
    - 创建实验目录和文件
    - 管理日志文件
    - 记录实验配置和结果
    - 使用Rich后端输出控制台日志
    """
    
    def __init__(
        self,
        output_dir: str = "experiment_results_boolq",
        model_name: str = "unknown_model",
        style: str = "direct",
        use_reasoning: bool = False,
        reason_order: str = "reason-after",
        console: Optional[Console] = None,
    ):
        """
        初始化实验日志管理器
        
        Args:
            output_dir: 输出目录
            model_name: 模型名称
            style: 提示词风格
            use_reasoning: 是否使用推理
            reason_order: 推理顺序
            console: Rich Console实例
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.console = console or Console()
        
        self.exp_uuid = str(uuid.uuid4())[:8]  # 使用短UUID
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_datetime = datetime.now()
        
        # 生成更可读的文件名
        # 格式: {model}_{style}_{date}_{time}_{uuid}
        safe_model_name = self._sanitize_model_name(model_name)
        reasoning_str = "CoT" if use_reasoning else "direct"
        
        # 新格式: gemini-2.5-flash_sse_CoT_20251201_001022_e0804eb9
        base_filename = f"{safe_model_name}_{style}_{reasoning_str}_{self.timestamp}_{self.exp_uuid}"
        
        self.log_path = self.output_dir / f"{base_filename}.log"
        self.data_path = self.output_dir / f"{base_filename}.jsonl"
        
        # 保存配置信息用于元数据
        self.model_name = model_name
        self.style = style
        self.use_reasoning = use_reasoning
        self.reason_order = reason_order
        
        # 配置loguru
        self._setup_loguru()
        
        logger.info(t("实验日志初始化完成"))
        logger.info(f"{t('日志文件')}: {self.log_path}")
        logger.info(f"{t('数据文件')}: {self.data_path}")
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """
        清理模型名称，使其适合作为文件名
        google/gemini-2.5-flash -> gemini-2.5-flash
        """
        # 移除provider前缀 (如 google/, openai/, anthropic/)
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        
        # 替换不安全字符
        model_name = model_name.replace(":", "-").replace(" ", "_")
        
        return model_name
    
    def _setup_loguru(self) -> None:
        """配置loguru日志，使用Rich后端"""
        # 移除默认handler
        logger.remove()
        
        # 添加Rich控制台handler
        rich_handler = RichLogHandler(self.console)
        logger.add(
            rich_handler.write,
            format="{message}",  # 格式由RichLogHandler处理
            level="DEBUG",
            colorize=False,  # Rich处理颜色
        )
        
        # 添加文件handler
        logger.add(
            self.log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            encoding="utf-8",
        )
    
    def log_config(self, config: Any) -> None:
        """记录实验配置"""
        logger.info(f"实验配置: {config}")
    
    def log_result(self, result: dict[str, Any]) -> None:
        """记录单条结果到JSONL文件"""
        with open(self.data_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    def log_summary(self, summary: str) -> None:
        """记录实验总结"""
        logger.info(f"\n{'='*50}\n{summary}\n{'='*50}")
    
    def log_evaluation_start(self, total_items: int, concurrency: int) -> None:
        """记录评估开始"""
        logger.info(t("开始评估: 模型={model}, 风格={style}", model=self.model_name, style=self.style))
        logger.info(t("数据集: {count} 条, 并发数: {concurrency}", count=total_items, concurrency=concurrency))
        reasoning_mode = t("启用") if self.use_reasoning else t("禁用")
        logger.info(t("推理模式: {mode}, 顺序: {order}", mode=reasoning_mode, order=self.reason_order))
    
    def log_evaluation_progress(
        self,
        completed: int,
        total: int,
        correct: int,
        errors: int,
        pending_write: int,
        written: int,
    ) -> None:
        """记录评估进度 (用于DEBUG级别)"""
        accuracy = correct / completed * 100 if completed > 0 else 0
        logger.debug(
            f"进度: {completed}/{total} ({completed/total*100:.1f}%) | "
            f"正确: {correct} ({accuracy:.1f}%) | "
            f"错误: {errors} | "
            f"待写入: {pending_write} | 已写入: {written}"
        )
    
    def log_item_result(
        self,
        index: int,
        is_correct: bool,
        predicted: Optional[bool],
        expected: bool,
        latency: float,
        avg_logprobs: Optional[float] = None,
        token_usage: Optional[TokenUsage] = None,
        cost: Optional[float] = None,
    ) -> None:
        """
        记录单项评估结果 (带颜色)
        
        使用Rich markup语法来添加颜色
        """
        # 状态标记 (带颜色)
        if is_correct:
            status = "[green]✓[/green]"
        else:
            status = "[red]✗[/red]"
        
        # 预测值 (带颜色)
        pred_str = str(predicted) if predicted is not None else "[dim]None[/dim]"
        if predicted is not None:
            pred_str = f"[cyan]{predicted}[/cyan]"
        
        # 期望值
        exp_str = f"[yellow]{expected}[/yellow]"
        
        # 延迟 (带颜色)
        if latency < 1:
            latency_str = f"[green]{latency:.2f}s[/green]"
        elif latency < 5:
            latency_str = f"[yellow]{latency:.2f}s[/yellow]"
        else:
            latency_str = f"[red]{latency:.2f}s[/red]"
        
        # LogProbs
        lp_str = ""
        if avg_logprobs is not None:
            if avg_logprobs > -0.1:
                lp_str = f", LP=[green]{avg_logprobs:.3f}[/green]"
            elif avg_logprobs > -0.2:
                lp_str = f", LP=[yellow]{avg_logprobs:.3f}[/yellow]"
            else:
                lp_str = f", LP=[red]{avg_logprobs:.3f}[/red]"
        
        # Token信息
        token_str = ""
        if token_usage:
            token_str = f", [dim]tok={token_usage.total_tokens}[/dim]"
        
        # 成本信息
        cost_str = ""
        if cost is not None and cost > 0:
            cost_str = f", [cyan]${cost:.6f}[/cyan]"
        
        logger.debug(
            f"[{status}] [blue]#{index}[/blue]: pred={pred_str}, exp={exp_str}, "
            f"lat={latency_str}{lp_str}{token_str}{cost_str}"
        )


class AsyncResultWriter:
    """
    异步结果写入器
    
    使用独立的异步任务处理结果写入，避免阻塞主评估流程
    跟踪待写入和已写入计数
    """
    
    def __init__(self, data_path: Path, buffer_size: int = 100):
        """
        初始化异步写入器
        
        Args:
            data_path: 数据文件路径
            buffer_size: 缓冲区大小
        """
        self.data_path = data_path
        self.buffer_size = buffer_size
        self._queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._stopped = False
        
        # 计数器
        self._pending_count = 0  # 待写入队列中的数量
        self._written_count = 0  # 已写入文件的数量
        self._lock = asyncio.Lock()
    
    @property
    def pending_count(self) -> int:
        """待写入数量"""
        return self._pending_count
    
    @property
    def written_count(self) -> int:
        """已写入数量"""
        return self._written_count
    
    async def start(self) -> None:
        """启动写入器"""
        self._stopped = False
        self._pending_count = 0
        self._written_count = 0
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.debug(t("异步结果写入器已启动"))
    
    async def stop(self) -> None:
        """停止写入器并等待所有数据写入完成"""
        self._stopped = True
        await self._queue.put(None)  # 发送停止信号
        
        if self._writer_task:
            await self._writer_task
        
        logger.debug(t("异步结果写入器已停止, 共写入 {count} 条记录", count=self._written_count))
    
    async def write(self, result: dict[str, Any]) -> None:
        """将结果放入队列等待写入"""
        if not self._stopped:
            async with self._lock:
                self._pending_count += 1
            await self._queue.put(result)
    
    async def _writer_loop(self) -> None:
        """写入循环"""
        buffer: list[dict[str, Any]] = []
        
        while True:
            try:
                # 获取数据，最多等待0.5秒
                try:
                    result = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # 超时时刷新缓冲区
                    if buffer:
                        await self._flush_buffer(buffer)
                        buffer.clear()
                    continue
                
                # 收到停止信号
                if result is None:
                    if buffer:
                        await self._flush_buffer(buffer)
                    break
                
                buffer.append(result)
                
                # 缓冲区满时刷新
                if len(buffer) >= self.buffer_size:
                    await self._flush_buffer(buffer)
                    buffer.clear()
                    
            except Exception as e:
                logger.error(f"写入器错误: {e}")
    
    async def _flush_buffer(self, buffer: list[dict[str, Any]]) -> None:
        """刷新缓冲区到文件"""
        if not buffer:
            return
        
        # 在线程池中执行IO操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_to_file, buffer)
        
        # 更新计数
        async with self._lock:
            count = len(buffer)
            self._pending_count -= count
            self._written_count += count
    
    def _write_to_file(self, buffer: list[dict[str, Any]]) -> None:
        """同步写入文件"""
        with open(self.data_path, "a", encoding="utf-8") as f:
            for result in buffer:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

