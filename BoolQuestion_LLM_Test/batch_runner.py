"""
批量任务执行器
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import BatchConfig, TaskConfig
from i18n import t

if TYPE_CHECKING:
    from boolq_evaluator import BoolQEvaluator


def list_tasks(config: BatchConfig, console: Console) -> None:
    """列出所有任务"""
    table = Table(title=t("批量任务列表"))

    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Style", style="magenta")
    table.add_column("Mode", style="blue")
    table.add_column("Limit", style="white")
    table.add_column("Enabled", style="green")

    for i, task in enumerate(config.tasks):
        enabled_str = "✓" if task.enabled else "✗"
        enabled_style = "green" if task.enabled else "red"
        table.add_row(
            str(i),
            task.name,
            task.model,
            task.style,
            task.eval_mode,
            str(task.limit) if task.limit > 0 else "all",
            f"[{enabled_style}]{enabled_str}[/{enabled_style}]"
        )

    console.print(table)
    console.print(f"\n{t('共 {count} 个任务', count=len(config.tasks))}")


async def run_task(
    task: TaskConfig,
    task_index: int,
    total_tasks: int,
    batch_config: BatchConfig,
    console: Console
) -> bool:
    """
    运行单个任务

    Returns:
        bool: 任务是否成功完成
    """
    console.print(Panel(
        f"[bold]{t('任务名称')}:[/bold] {task.name}\n"
        f"[bold]{t('模型')}:[/bold] {task.model}\n"
        f"[bold]{t('风格')}:[/bold] {task.style}\n"
        f"[bold]{t('评估模式')}:[/bold] {task.eval_mode}\n"
        f"[bold]{t('数据限制')}:[/bold] {task.limit if task.limit > 0 else t('全部')}\n"
        f"[bold]{t('并发数')}:[/bold] {task.concurrency}",
        title=f"[bold cyan]{t('任务')} [{task_index + 1}/{total_tasks}][/bold cyan]"
    ))

    try:
        # 延迟导入，避免 --list 时加载重量级依赖
        from boolq_evaluator import BoolQEvaluator

        exp_config = task.to_experiment_config()
        exp_config.output_dir = batch_config.output_dir
        exp_config.dirty_data_path = batch_config.dirty_data_path

        evaluator = BoolQEvaluator(exp_config, console)
        await evaluator.run()

        console.print(f"[green]✓ {t('任务完成')}: {task.name}[/green]\n")
        return True

    except Exception as e:
        console.print(f"[red]✗ {t('任务失败')}: {task.name}[/red]")
        console.print(f"[red]  {t('错误')}: {e}[/red]\n")
        logger.exception(f"Task failed: {task.name}")
        return False


async def run_batch(
    config: BatchConfig,
    task_indices: Optional[list[int]],
    console: Console
) -> None:
    """运行批量任务"""
    # 确定要运行的任务
    if task_indices is not None:
        tasks_to_run = [(i, config.tasks[i]) for i in task_indices if i < len(config.tasks)]
    else:
        tasks_to_run = [(i, task) for i, task in enumerate(config.tasks) if task.enabled]

    if not tasks_to_run:
        console.print(f"[yellow]{t('没有可执行的任务')}[/yellow]")
        return

    total = len(tasks_to_run)
    console.print(Panel(
        f"[bold]{t('总任务数')}:[/bold] {total}\n"
        f"[bold]{t('输出目录')}:[/bold] {config.output_dir}",
        title=f"[bold green]{t('批量评估开始')}[/bold green]"
    ))

    success_count = 0
    failed_count = 0
    failed_tasks = []

    for idx, (task_index, task) in enumerate(tasks_to_run):
        success = await run_task(task, idx, total, config, console)
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_tasks.append(task.name)

    console.print(Panel(
        f"[bold]{t('成功')}:[/bold] [green]{success_count}[/green]\n"
        f"[bold]{t('失败')}:[/bold] [red]{failed_count}[/red]" +
        (f"\n[bold]{t('失败任务')}:[/bold] {', '.join(failed_tasks)}" if failed_tasks else ""),
        title=f"[bold cyan]{t('批量评估完成')}[/bold cyan]"
    ))

