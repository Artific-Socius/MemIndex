#!/usr/bin/env python3
"""
BoolQ LLM评估实验 - 批量执行入口

使用方法:
    python main.py                           # 执行所有启用的任务
    python main.py --config my_config.yaml   # 指定配置文件
    python main.py --list                    # 列出所有任务
    python main.py --task 0,1,2              # 执行指定任务
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import BatchConfig
from batch_runner import list_tasks, run_batch
from i18n import t, set_language


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BoolQ LLM评估实验 - 批量执行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           # 使用默认配置
  python main.py --config my_config.yaml   # 指定配置文件
  python main.py --list                    # 列出任务
  python main.py --task 0,2,3              # 执行指定任务
        """
    )
    
    parser.add_argument("--config", "-c", type=str, default="batch_config.yaml",
                        help="配置文件路径 (默认: batch_config.yaml)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有任务")
    parser.add_argument("--task", "-t", type=str, default=None,
                        help="执行指定任务索引 (逗号分隔)")
    parser.add_argument("--lang", type=str, choices=["zh", "en", "auto"], default=None,
                        help="语言设置")
    
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    console = Console()
    
    # 查找配置文件
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    
    if not config_path.exists():
        console.print(f"[red]{t('配置文件不存在')}: {args.config}[/red]")
        console.print(f"[dim]请创建 batch_config.yaml 或使用 --config 指定配置文件[/dim]")
        sys.exit(1)
    
    # 加载配置
    try:
        config = BatchConfig.from_yaml(str(config_path))
    except Exception as e:
        console.print(f"[red]{t('配置文件解析失败')}: {e}[/red]")
        sys.exit(1)
    
    # 设置语言
    lang = args.lang or config.lang
    if lang and lang != "auto":
        set_language(lang)
    
    # 列出任务
    if args.list:
        list_tasks(config, console)
        return
    
    # 解析任务索引
    task_indices = None
    if args.task:
        try:
            task_indices = [int(x.strip()) for x in args.task.split(',')]
        except ValueError:
            console.print(f"[red]{t('无效的任务索引')}: {args.task}[/red]")
            sys.exit(1)
    
    # 运行批量任务
    asyncio.run(run_batch(config, task_indices, console))


if __name__ == "__main__":
    main()
