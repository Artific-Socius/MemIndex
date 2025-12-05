"""
BatchConfig - 批量运行配置

提供批量运行时的配置管理，支持多个任务并行执行。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type, TypeVar, List, Optional

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from .running_config import RunningConfig


class TaskConfig(BaseModel):
    """单个任务的配置"""

    # 任务名称
    name: str = Field(
        "",
        description="任务名称（用于显示）"
    )

    # 运行配置字段（和 RunningConfig 相同）
    chat_model: str = Field(
        "openai/gpt-4o-mini",
        description="Chat Model - 用于对话的目标模型"
    )
    eval_model: str = Field(
        "volcano/deepseek-v3-250324",
        description="Eval Model - 用于评估的模型"
    )
    memory_provider: str = Field(
        "llm",
        description="Memory Provider - Agent类型"
    )
    context_window: int = Field(
        16384,
        description="Context Window - 上下文窗口大小(tokens)"
    )
    benchmark_config: str = Field(
        "./data/config/1k.json",
        description="Benchmark Config - 基准测试配置文件路径"
    )
    report_dir: str = Field(
        "./data/reports",
        description="Report Directory - 报告输出目录"
    )

    # Prompt 配置
    chat_prompt: Optional[str] = Field(
        None,
        description="Chat Prompt - Chat模型使用的提示词key"
    )
    eval_prompt: Optional[str] = Field(
        None,
        description="Eval Prompt - Eval模型使用的提示词key"
    )

    # 评估模式配置
    eval_mode: str = Field(
        "binary",
        description="Eval Mode - 评估模式 (binary: 二元评估, score: 0-1分数评估)"
    )

    # 任务控制
    enabled: bool = Field(
        True,
        description="是否启用此任务"
    )
    priority: int = Field(
        0,
        description="任务优先级（数字越小优先级越高）"
    )
    times: int = Field(
        1,
        description="任务运行次数（创建多个完全相同的独立运行实例）"
    )

    # 内部字段：标记这是展开后的第几次运行
    _run_index: int = 0

    def to_running_config(self) -> RunningConfig:
        """转换为 RunningConfig"""
        return RunningConfig(
            chat_model=self.chat_model,
            eval_model=self.eval_model,
            memory_provider=self.memory_provider,
            context_window=self.context_window,
            benchmark_config=self.benchmark_config,
            report_dir=self.report_dir,
            chat_prompt=self.chat_prompt,
            eval_prompt=self.eval_prompt,
            eval_mode=self.eval_mode,
        )


class BatchConfig(BaseModel):
    """批量运行配置"""

    # 并行配置
    max_parallel: int = Field(
        2,
        description="同时运行的任务数量"
    )

    # 是否在任务失败时继续
    continue_on_error: bool = Field(
        True,
        description="任务失败时是否继续执行其他任务"
    )

    # 任务间延迟（秒）
    task_delay: float = Field(
        0.5,
        description="启动任务之间的延迟（秒）"
    )

    # 全局默认 Prompt 配置
    default_chat_prompt: Optional[str] = Field(
        None,
        description="默认 Chat Prompt key"
    )
    default_eval_prompt: Optional[str] = Field(
        None,
        description="默认 Eval Prompt key"
    )

    # 全局默认评估模式
    default_eval_mode: str = Field(
        "binary",
        description="默认评估模式 (binary: 二元评估, score: 0-1分数评估)"
    )

    # 任务列表
    tasks: List[TaskConfig] = Field(
        default_factory=list,
        description="任务配置列表"
    )


class BatchConfigManager:
    """
    批量配置管理器

    加载和管理批量运行配置。
    """

    def __init__(self, batch_config_path: str):
        """
        初始化批量配置管理器

        Args:
            batch_config_path: 批量配置文件路径
        """
        self.batch_config_path = batch_config_path
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    def load_config(self) -> BatchConfig:
        """
        加载批量配置

        Returns:
            批量配置对象
        """
        if not Path(self.batch_config_path).exists():
            raise FileNotFoundError(f"Batch config not found: {self.batch_config_path}")

        with open(self.batch_config_path, 'r', encoding='utf-8') as f:
            data = self.yaml.load(f) or {}

        # 获取全局默认配置
        default_chat_prompt = data.get('default_chat_prompt')
        default_eval_prompt = data.get('default_eval_prompt')
        default_eval_mode = data.get('default_eval_mode', 'binary')

        # 解析任务列表
        tasks_data = data.pop('tasks', [])
        tasks = []

        for i, task_data in enumerate(tasks_data):
            # 如果没有 name，使用索引生成
            if 'name' not in task_data or not task_data['name']:
                task_data['name'] = f"Task {i + 1}"

            # 如果任务没有设置，使用全局默认
            if task_data.get('chat_prompt') is None and default_chat_prompt:
                task_data['chat_prompt'] = default_chat_prompt
            if task_data.get('eval_prompt') is None and default_eval_prompt:
                task_data['eval_prompt'] = default_eval_prompt
            if task_data.get('eval_mode') is None or task_data.get('eval_mode') == 'binary':
                # 只有当任务没有明确设置或使用默认值时才使用全局默认
                if 'eval_mode' not in task_data:
                    task_data['eval_mode'] = default_eval_mode

            task = TaskConfig(**task_data)
            if task.enabled:
                # 根据 times 参数展开任务
                times = max(1, task.times)
                for run_idx in range(times):
                    # 创建任务副本
                    task_copy = task.model_copy()
                    task_copy._run_index = run_idx
                    # 如果 times > 1，在名称后添加运行序号
                    if times > 1:
                        task_copy.name = f"{task.name} (Run {run_idx + 1}/{times})"
                    tasks.append(task_copy)

        # 按优先级排序
        tasks.sort(key=lambda x: x.priority)

        return BatchConfig(
            max_parallel=data.get('max_parallel', 2),
            continue_on_error=data.get('continue_on_error', True),
            task_delay=data.get('task_delay', 0.5),
            default_chat_prompt=default_chat_prompt,
            default_eval_prompt=default_eval_prompt,
            default_eval_mode=default_eval_mode,
            tasks=tasks,
        )

    def get_enabled_tasks(self) -> List[TaskConfig]:
        """获取所有启用的任务"""
        config = self.load_config()
        return config.tasks

    def load_all_tasks(self) -> tuple[BatchConfig, List[TaskConfig]]:
        """
        加载所有任务（包括禁用的）

        Returns:
            (BatchConfig, 所有任务列表)
        """
        if not Path(self.batch_config_path).exists():
            raise FileNotFoundError(f"Batch config not found: {self.batch_config_path}")

        with open(self.batch_config_path, 'r', encoding='utf-8') as f:
            data = self.yaml.load(f) or {}

        # 获取全局默认配置
        default_chat_prompt = data.get('default_chat_prompt')
        default_eval_prompt = data.get('default_eval_prompt')
        default_eval_mode = data.get('default_eval_mode', 'binary')

        # 解析所有任务
        tasks_data = data.pop('tasks', [])
        all_tasks = []

        for i, task_data in enumerate(tasks_data):
            if 'name' not in task_data or not task_data['name']:
                task_data['name'] = f"Task {i + 1}"

            # 如果任务没有设置，使用全局默认
            if task_data.get('chat_prompt') is None and default_chat_prompt:
                task_data['chat_prompt'] = default_chat_prompt
            if task_data.get('eval_prompt') is None and default_eval_prompt:
                task_data['eval_prompt'] = default_eval_prompt
            if 'eval_mode' not in task_data:
                task_data['eval_mode'] = default_eval_mode

            task = TaskConfig(**task_data)

            # 根据 times 参数展开任务
            times = max(1, task.times)
            for run_idx in range(times):
                task_copy = task.model_copy()
                task_copy._run_index = run_idx
                if times > 1:
                    task_copy.name = f"{task.name} (Run {run_idx + 1}/{times})"
                all_tasks.append(task_copy)

        # 按优先级排序
        all_tasks.sort(key=lambda x: x.priority)

        config = BatchConfig(
            max_parallel=data.get('max_parallel', 2),
            continue_on_error=data.get('continue_on_error', True),
            task_delay=data.get('task_delay', 0.5),
            default_chat_prompt=default_chat_prompt,
            default_eval_prompt=default_eval_prompt,
            default_eval_mode=default_eval_mode,
            tasks=[t for t in all_tasks if t.enabled],
        )

        return config, all_tasks

    def save_config(self, config: BatchConfig) -> None:
        """保存批量配置"""
        data = {
            'max_parallel': config.max_parallel,
            'continue_on_error': config.continue_on_error,
            'task_delay': config.task_delay,
            'default_chat_prompt': config.default_chat_prompt,
            'default_eval_prompt': config.default_eval_prompt,
            'default_eval_mode': config.default_eval_mode,
            'tasks': [task.model_dump() for task in config.tasks],
        }

        with open(self.batch_config_path, 'w', encoding='utf-8') as f:
            self.yaml.dump(data, f)
