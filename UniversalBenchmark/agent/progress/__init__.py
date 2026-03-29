"""全局 Rich 进度管理（单 Console / Progress 实例）。"""

from .manager import (
    NullProgressManager,
    RichProgressManager,
    TaskHandle,
    get_console,
    get_progress,
    loguru_sink_message,
    progress_context,
    track_task,
)

__all__ = [
    "NullProgressManager",
    "RichProgressManager",
    "TaskHandle",
    "get_console",
    "get_progress",
    "loguru_sink_message",
    "progress_context",
    "track_task",
]
