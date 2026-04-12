"""全局 Rich 进度条管理：单进程内唯一 Progress/Console 显示实例。

通过 :func:`progress_context` 在入口开启；业务代码用 :func:`get_progress`
获取当前管理器（无上下文时为 no-op，不抛错）。

日志与进度共享同一 :class:`rich.console.Console`（TTY 下），避免 stdout 日志与
stderr Live 进度交织换行。支持 ``task_key`` 复用任务行以压制重复进度条。
"""

from __future__ import annotations

import re
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# 任务句柄（与 Rich TaskID 解耦）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskHandle:
    """进度任务句柄。``task_id`` 为 ``None`` 表示 no-op 任务。"""

    task_id: Any


# ---------------------------------------------------------------------------
# No-op 实现（无活动上下文时）
# ---------------------------------------------------------------------------


class NullProgressManager:
    """未启用全局进度时的占位实现，所有方法为空操作。"""

    def add_task(
        self,
        description: str = "",
        *,
        total: float | None = None,
        task_key: str | None = None,
    ) -> TaskHandle:
        return TaskHandle(task_id=None)

    def update(
        self,
        handle: TaskHandle,
        *,
        completed: float | None = None,
        total: float | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        return

    def advance(self, handle: TaskHandle, steps: float = 1) -> None:
        return

    def remove_task(self, handle: TaskHandle) -> None:
        return


_NULL = NullProgressManager()

# ---------------------------------------------------------------------------
# 全局状态（嵌套上下文 + 单实例 + 共享 Console）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_nesting_depth: int = 0
_active: Optional["RichProgressManager"] = None
_session_console: Any = None

DEFAULT_MIN_WIDTH = 40
DEFAULT_MAX_WIDTH = 256


def _clamp_width(min_w: int, max_w: int) -> int:
    """按终端列数 clamp 到 [min_w, max_w]。"""
    try:
        cols = shutil.get_terminal_size().columns
    except OSError:
        cols = 120
    cols = max(min_w, cols)
    return max(min_w, min(max_w, cols))


def get_console() -> Any:
    """返回当前会话的共享 Rich Console；无会话时为 ``None``。"""
    with _lock:
        return _session_console


def get_progress() -> NullProgressManager | "RichProgressManager":
    """返回当前活动的进度管理器；若无上下文则返回 no-op 单例。"""
    with _lock:
        return _active if _active is not None else _NULL


# ---------------------------------------------------------------------------
# Rich 实现
# ---------------------------------------------------------------------------


class RichProgressManager:
    """包装单个 :class:`rich.progress.Progress` 与 :class:`rich.console.Console`。"""

    def __init__(
        self,
        *,
        console: Any = None,
        disable: bool = False,
    ) -> None:
        self._disable = disable
        self._console = console
        self._progress: Any = None
        self._entered = False
        self._task_by_key: dict[str, TaskHandle] = {}
        self._tid_to_key: dict[Any, str] = {}

    def _ensure_started(self) -> None:
        if self._entered or self._disable:
            return
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TaskProgressColumn,
        )
        from rich.table import Column

        cons = self._console if self._console is not None else Console(stderr=True)
        self._console = cons
        w = getattr(cons, "width", None) or _clamp_width(
            DEFAULT_MIN_WIDTH, DEFAULT_MAX_WIDTH,
        )
        desc_max = max(16, min(96, w - 36))

        self._progress = Progress(
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(
                    no_wrap=True,
                    overflow="ellipsis",
                    min_width=12,
                    max_width=desc_max,
                ),
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=cons,
            transient=False,
            expand=False,
        )
        self._progress.__enter__()
        self._entered = True

    def _stop(self) -> None:
        if self._progress is not None and self._entered:
            try:
                self._progress.__exit__(None, None, None)
            finally:
                self._progress = None
                self._entered = False
        self._task_by_key.clear()
        self._tid_to_key.clear()

    def add_task(
        self,
        description: str = "",
        *,
        total: float | None = None,
        task_key: str | None = None,
    ) -> TaskHandle:
        if not self._entered or self._progress is None:
            return TaskHandle(task_id=None)
        if task_key:
            existing = self._task_by_key.get(task_key)
            if existing is not None and existing.task_id is not None:
                self.update(
                    existing,
                    total=total,
                    description=description,
                )
                return existing

        tid = self._progress.add_task(description, total=total)
        h = TaskHandle(task_id=tid)
        if task_key:
            self._task_by_key[task_key] = h
            self._tid_to_key[tid] = task_key
        return h

    def update(
        self,
        handle: TaskHandle,
        *,
        completed: float | None = None,
        total: float | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        if handle.task_id is None or self._progress is None:
            return
        kw: dict[str, Any] = {}
        if completed is not None:
            kw["completed"] = completed
        if total is not None:
            kw["total"] = total
        if description is not None:
            kw["description"] = description
        kw.update(kwargs)
        self._progress.update(handle.task_id, **kw)

    def advance(self, handle: TaskHandle, steps: float = 1) -> None:
        if handle.task_id is None or self._progress is None:
            return
        self._progress.advance(handle.task_id, advance=steps)

    def remove_task(self, handle: TaskHandle) -> None:
        if handle.task_id is None or self._progress is None:
            return
        tid = handle.task_id
        key = self._tid_to_key.pop(tid, None)
        if key is not None:
            self._task_by_key.pop(key, None)
        try:
            self._progress.remove_task(tid)
        except Exception:
            pass


class progress_context:
    """在 CLI/脚本入口使用 ``with progress_context():`` 开启 Rich 会话。

    - ``live=True``：启动多任务 Progress（与共享 Console 绑定）。
    - ``force_console=True``（TTY）：即使 ``live=False`` 也创建共享 Console，
      供日志与 :func:`get_console` 使用（例如 ``--no-progress``）。
    - ``disable=True``：完全跳过（兼容旧测试：不嵌套、不建 Console）。
    """

    def __init__(
        self,
        *,
        live: bool = True,
        disable: bool = False,
        force_console: bool = False,
        console: Any = None,
        min_width: int = DEFAULT_MIN_WIDTH,
        max_width: int = DEFAULT_MAX_WIDTH,
    ) -> None:
        if disable:
            live = False
            force_console = False
        self._live = live
        self._disable = disable
        self._force_console = force_console
        self._console_arg = console
        self._min_w = min_width
        self._max_w = max_width
        self._mgr: RichProgressManager | None = None

    def __enter__(self) -> NullProgressManager | RichProgressManager:
        global _nesting_depth, _active, _session_console
        with _lock:
            if self._disable:
                return _active if _active is not None else _NULL

            if _nesting_depth == 0:
                cons = self._console_arg
                tty = sys.stderr.isatty()
                need_console = cons is not None or (
                    tty and (self._live or self._force_console)
                )
                if need_console and cons is None:
                    w = _clamp_width(self._min_w, self._max_w)
                    from rich.console import Console

                    cons = Console(
                        stderr=True,
                        width=w,
                        force_terminal=True,
                        legacy_windows=False,
                    )
                _session_console = cons

                if self._live and cons is not None:
                    self._mgr = RichProgressManager(console=cons, disable=False)
                    self._mgr._ensure_started()
                    _active = self._mgr
                else:
                    self._mgr = None
                    _active = None
            else:
                self._mgr = _active  # type: ignore[assignment]

            _nesting_depth += 1

        return self._mgr if self._mgr is not None else _NULL

    def __exit__(self, *exc: Any) -> None:
        global _nesting_depth, _active, _session_console
        if self._disable:
            return
        with _lock:
            _nesting_depth = max(0, _nesting_depth - 1)
            if _nesting_depth == 0:
                if _active is not None:
                    _active._stop()
                    _active = None
                _session_console = None


@contextmanager
def track_task(
    description: str,
    *,
    total: float | None = None,
    task_key: str | None = None,
) -> Iterator[TaskHandle]:
    """上下文管理器：进入时 add_task，退出时 remove_task。"""
    mgr = get_progress()
    h = mgr.add_task(description, total=total, task_key=task_key)
    try:
        yield h
    finally:
        mgr.remove_task(h)


_LOG_LEVEL_RICH_STYLES: dict[str, str] = {
    "TRACE": "dim",
    "DEBUG": "cyan",
    "INFO": "green",
    "SUCCESS": "bold green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}

# 日志正文内高亮：前列模式优先（重叠时保留先匹配的片段）。
_LOG_BODY_STYLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\[[^\]]+\]"), "yellow"),
    (re.compile(r"\b(?:corpus_id|scene_id)=[^\s)|]+"), "magenta"),
    (re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s|]+"), "bright_blue"),
    (re.compile(r"\b[\w.\-]+(?:/[\w.\-]+)+\b"), "cyan"),
    (re.compile(r"\sFAIL\s"), "red"),
    (re.compile(r"\sPASS\s"), "green"),
]


def _log_body_spans(body: str) -> list[tuple[int, int, str]]:
    """返回不重叠的 (start, end, style) 列表，按在正文中的顺序排序。"""
    occupied: list[tuple[int, int]] = []
    out: list[tuple[int, int, str]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(max(s, os) < min(e, oe) for os, oe in occupied)

    for pat, style in _LOG_BODY_STYLE_PATTERNS:
        for m in pat.finditer(body):
            s, e = m.start(), m.end()
            if overlaps(s, e):
                continue
            occupied.append((s, e))
            out.append((s, e, style))

    out.sort(key=lambda x: x[0])
    return out


def _append_log_body_default_white(t: Any, body: str) -> None:
    """正文默认使用终端前景色；对括号标签、路径、slug 等做局部着色。"""
    spans = _log_body_spans(body)
    pos = 0
    for s, e, style in spans:
        if s > pos:
            t.append(body[pos:s])
        t.append(body[s:e], style=style)
        pos = e
    if pos < len(body):
        t.append(body[pos:])


def loguru_sink_message(message: Any) -> None:
    """供 loguru ``logger.add`` 使用的 sink：写入共享 Console（单行截断）。

    使用 ``message.record`` 拼 Rich ``Text``：时间/级别/源码位置保持区分色；**消息正文**
    默认终端前景色（通常为白/灰白），并对 ``[标签]``、Windows 路径、``corpus_id=``、
    ``a/b/c`` 式 slug 等做局部高亮，无需在各 ``logger.info`` 调用处手写标记。

    建议在 ``logger.add`` 中使用 ``format="{message}"``、``colorize=False``；若
    ``get_console()`` 为 ``None`` 则回退为 ``str(message)`` 写入 stderr。
    """
    from rich.text import Text

    c = get_console()
    record = message.record
    level_name = record["level"].name
    lvl_style = _LOG_LEVEL_RICH_STYLES.get(level_name, "")

    time_v = record["time"]
    ts = time_v.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    lev = f"{level_name:<8}"
    name = record["name"]
    function = record["function"]
    line = record["line"]
    body = str(record["message"])

    if c is not None:
        w = getattr(c, "width", None) or _clamp_width(
            DEFAULT_MIN_WIDTH, DEFAULT_MAX_WIDTH,
        )
        t = Text()
        t.append(ts, style="green")
        t.append(" | ", style="dim")
        t.append(lev, style=lvl_style or None)
        t.append(" | ", style="dim")
        t.append(f"{name}:{function}:{line}", style="cyan")
        t.append(" - ", style="dim")
        _append_log_body_default_white(t, body)
        t.no_wrap = True
        c.print(
            t,
            width=w,
            soft_wrap=False,
            overflow="ellipsis",
            crop=True,
        )
    else:
        raw = str(message).rstrip("\n")
        sys.stderr.write(raw + "\n")
