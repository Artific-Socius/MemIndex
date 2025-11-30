from __future__ import annotations

import time
from loguru import logger


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

