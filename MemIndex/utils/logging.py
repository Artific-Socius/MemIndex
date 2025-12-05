"""
Logging - 日志配置模块

使用 Rich 作为 loguru 的输出后端，提供美观的日志输出。
与 Rich 进度条兼容，不会产生输出冲突。

核心功能:
    1. 基于 loguru 的日志记录
    2. 使用 Rich 美化输出
    3. 支持 Rich 的异常追踪
    4. 全局 Console 实例管理

使用方式:
    from utils.logging import setup_logging, logger
    
    setup_logging(level="DEBUG")
    logger.info("Hello World")
    logger.debug("Debug info")
    logger.error("Something went wrong")
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.theme import Theme

# 全局 Console 实例（单例）
# 所有 Rich 输出使用同一个 Console，避免与进度条冲突
_console: Console | None = None


def get_console() -> Console:
    """
    获取全局 Console 实例
    
    确保所有 Rich 输出使用同一个 Console，
    这样可以避免日志和进度条输出冲突。
    
    Returns:
        全局 Console 实例
    """
    global _console
    if _console is None:
        # 创建自定义主题
        custom_theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "debug": "dim",
            "success": "bold green",
        })
        _console = Console(theme=custom_theme, force_terminal=True)
    return _console


def rich_sink(message):
    """
    Rich sink for loguru
    
    将 loguru 的日志消息通过 Rich Console 输出。
    这是连接 loguru 和 Rich 的桥梁。
    
    输出格式:
        HH:MM:SS | LEVEL    | module:function:line - message
    
    颜色方案:
        - TRACE: dim
        - DEBUG: dim cyan
        - INFO: green
        - SUCCESS: bold green
        - WARNING: yellow
        - ERROR: bold red
        - CRITICAL: bold white on red
    
    Args:
        message: loguru 消息对象
    """
    console = get_console()
    record = message.record
    level = record["level"].name.lower()
    
    # 日志级别颜色映射
    level_colors = {
        "trace": "dim",
        "debug": "dim cyan",
        "info": "green",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
    
    color = level_colors.get(level, "white")
    level_str = f"[{color}]{record['level'].name:8}[/{color}]"
    
    # 格式化时间（只显示时分秒）
    time_str = f"[dim]{record['time'].strftime('%H:%M:%S')}[/dim]"
    
    # 格式化位置信息（模块:函数:行号）
    location = f"[dim]{record['name']}:{record['function']}:{record['line']}[/dim]"
    
    # 格式化消息
    msg = str(record["message"])
    
    # 组合输出
    output = f"{time_str} | {level_str} | {location} - {msg}"
    
    console.print(output, highlight=False)


def setup_logging(
    level: str = "INFO",
    show_path: bool = True,
    rich_tracebacks: bool = True,
) -> None:
    """
    配置日志系统
    
    设置 loguru 使用 Rich 作为输出后端。
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        show_path: 是否显示文件路径（保留参数，暂未使用）
        rich_tracebacks: 是否使用 Rich 格式化异常堆栈
            启用后，异常信息会更加美观，包含语法高亮和局部变量
    """
    # 移除 loguru 默认的 handler
    logger.remove()
    
    # 添加 Rich sink
    logger.add(
        rich_sink,
        level=level,
        format="{message}",  # 格式化在 sink 中处理
        colorize=False,      # 颜色由 Rich 处理
        backtrace=True,      # 异常时显示完整调用栈
        diagnose=True,       # 显示异常诊断信息
    )
    
    # 配置 Rich 异常追踪
    if rich_tracebacks:
        from rich.traceback import install
        install(
            console=get_console(), 
            show_locals=True,    # 显示局部变量
            width=120            # 输出宽度
        )


def setup_simple_logging(level: str = "INFO") -> None:
    """
    配置简单的日志系统（不使用 Rich）
    
    用于非交互环境（如 CI/CD、后台服务）。
    使用 loguru 的内置颜色格式化。
    
    Args:
        level: 日志级别
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )


def init_logging():
    """
    初始化日志系统
    
    使用默认配置初始化。
    """
    setup_logging(level="INFO")


# 导出
__all__ = [
    "logger",          # loguru 的 logger 实例
    "get_console",     # 获取全局 Console
    "setup_logging",   # 配置 Rich 日志
    "setup_simple_logging",  # 配置简单日志
    "init_logging",    # 默认初始化
]
