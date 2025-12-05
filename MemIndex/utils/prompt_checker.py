"""
Prompt Checker - 提示词配置检测工具

独立运行的工具脚本，用于：
1. 加载并验证 prompts.yaml 配置
2. 检测循环继承
3. 检测可以使用继承写法的重复 prompt
4. 显示继承关系

使用方法：
    python -m utils.prompt_checker [prompts_file]
    
    或者直接运行：
    python utils/prompt_checker.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ruamel.yaml import YAML
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

# 继承关键字
EXTENDS_KEY = "_extends"

# 相似度阈值（高于此值认为可以考虑使用继承）
SIMILARITY_THRESHOLD = 0.8


class PromptChecker:
    """
    提示词配置检测器
    """
    
    def __init__(self, prompts_file: str):
        """
        初始化检测器
        
        Args:
            prompts_file: prompts.yaml 文件路径
        """
        self.prompts_file = Path(prompts_file)
        self.console = Console()
        self.yaml = YAML()
        self.data: dict = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
    
    def load(self) -> bool:
        """
        加载配置文件
        
        Returns:
            是否加载成功
        """
        if not self.prompts_file.exists():
            self.errors.append(f"配置文件不存在: {self.prompts_file}")
            return False
        
        try:
            with self.prompts_file.open("r", encoding="utf-8") as f:
                self.data = self.yaml.load(f) or {}
            return True
        except Exception as e:
            self.errors.append(f"加载配置文件失败: {e}")
            return False
    
    def check_circular_inheritance(self, prompts: dict, category: str) -> List[str]:
        """
        检测循环继承
        
        Args:
            prompts: prompt 配置字典
            category: 配置类别（chat 或 eval）
            
        Returns:
            循环继承错误列表
        """
        errors = []
        
        def find_cycle(key: str, visited: Set[str], path: List[str]) -> List[str] | None:
            if key in visited:
                cycle_start = path.index(key)
                return path[cycle_start:] + [key]
            
            if key not in prompts:
                return None
            
            config = prompts[key]
            if not isinstance(config, dict):
                return None
            
            extends = config.get(EXTENDS_KEY)
            if extends is None:
                return None
            
            visited.add(key)
            path.append(key)
            
            result = find_cycle(extends, visited, path)
            
            visited.remove(key)
            path.pop()
            
            return result
        
        for key in prompts:
            cycle = find_cycle(key, set(), [])
            if cycle:
                cycle_str = " -> ".join(cycle)
                error = f"[{category}] 循环继承: {cycle_str}"
                if error not in errors:
                    errors.append(error)
        
        return errors
    
    def check_missing_parent(self, prompts: dict, category: str) -> List[str]:
        """
        检测缺失的父配置
        
        Args:
            prompts: prompt 配置字典
            category: 配置类别
            
        Returns:
            错误列表
        """
        errors = []
        
        for key, config in prompts.items():
            if not isinstance(config, dict):
                continue
            
            extends = config.get(EXTENDS_KEY)
            if extends and extends not in prompts:
                errors.append(f"[{category}.{key}] 继承的父配置 '{extends}' 不存在")
        
        return errors
    
    def find_duplicate_prompts(self, prompts: dict, category: str) -> List[Tuple[str, str, str, float]]:
        """
        找出重复或高度相似的 prompt
        
        Args:
            prompts: prompt 配置字典
            category: 配置类别
            
        Returns:
            重复列表: [(key1, key2, prompt_name, similarity)]
        """
        duplicates = []
        keys = list(prompts.keys())
        
        # 收集所有 prompt 字段
        all_prompt_names = set()
        for config in prompts.values():
            if isinstance(config, dict):
                for k in config:
                    if k != EXTENDS_KEY:
                        all_prompt_names.add(k)
        
        # 比较每对配置
        for i, key1 in enumerate(keys):
            config1 = prompts[key1]
            if not isinstance(config1, dict):
                continue
            
            # 检查是否已经使用继承
            extends1 = config1.get(EXTENDS_KEY)
            
            for key2 in keys[i + 1:]:
                config2 = prompts[key2]
                if not isinstance(config2, dict):
                    continue
                
                extends2 = config2.get(EXTENDS_KEY)
                
                # 比较每个 prompt 字段
                for prompt_name in all_prompt_names:
                    prompt1 = config1.get(prompt_name, "")
                    prompt2 = config2.get(prompt_name, "")
                    
                    if not prompt1 or not prompt2:
                        continue
                    
                    # 计算相似度
                    similarity = SequenceMatcher(None, prompt1, prompt2).ratio()
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        # 检查是否已经是继承关系
                        is_inherited = (extends1 == key2) or (extends2 == key1)
                        if not is_inherited:
                            duplicates.append((key1, key2, prompt_name, similarity))
        
        return duplicates
    
    def find_inheritance_opportunities(self, prompts: dict, category: str) -> List[str]:
        """
        找出可以使用继承的机会
        
        Args:
            prompts: prompt 配置字典
            category: 配置类别
            
        Returns:
            建议列表
        """
        suggestions = []
        duplicates = self.find_duplicate_prompts(prompts, category)
        
        # 按 (key1, key2) 分组
        grouped = defaultdict(list)
        for key1, key2, prompt_name, similarity in duplicates:
            grouped[(key1, key2)].append((prompt_name, similarity))
        
        for (key1, key2), items in grouped.items():
            if len(items) >= 2:  # 有多个相似字段
                avg_similarity = sum(s for _, s in items) / len(items)
                fields = ", ".join(f"{n}({s:.0%})" for n, s in items)
                suggestions.append(
                    f"[{category}] '{key1}' 和 '{key2}' 有 {len(items)} 个相似字段 [{fields}]，"
                    f"平均相似度 {avg_similarity:.0%}，建议使用继承"
                )
            elif items[0][1] == 1.0:  # 完全相同
                prompt_name, similarity = items[0]
                suggestions.append(
                    f"[{category}] '{key1}' 和 '{key2}' 的 '{prompt_name}' 完全相同，"
                    f"建议使用继承避免重复"
                )
        
        return suggestions
    
    def get_inheritance_tree(self, prompts: dict, category: str) -> Dict[str, List[str]]:
        """
        获取继承树
        
        Args:
            prompts: prompt 配置字典
            category: 配置类别
            
        Returns:
            继承关系字典 {parent: [children]}
        """
        tree = defaultdict(list)
        roots = []
        
        for key, config in prompts.items():
            if not isinstance(config, dict):
                continue
            
            extends = config.get(EXTENDS_KEY)
            if extends:
                tree[extends].append(key)
            else:
                roots.append(key)
        
        return dict(tree), roots
    
    def run_checks(self) -> None:
        """运行所有检测"""
        if not self.load():
            return
        
        chat_prompts = self.data.get("chat", {})
        eval_prompts = self.data.get("eval", {})
        
        # 检测循环继承
        self.errors.extend(self.check_circular_inheritance(chat_prompts, "chat"))
        self.errors.extend(self.check_circular_inheritance(eval_prompts, "eval"))
        
        # 检测缺失的父配置
        self.errors.extend(self.check_missing_parent(chat_prompts, "chat"))
        self.errors.extend(self.check_missing_parent(eval_prompts, "eval"))
        
        # 找出继承优化建议
        self.suggestions.extend(self.find_inheritance_opportunities(chat_prompts, "chat"))
        self.suggestions.extend(self.find_inheritance_opportunities(eval_prompts, "eval"))
    
    def print_report(self) -> None:
        """打印检测报告"""
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Prompt Configuration Checker[/bold cyan]\n"
            f"[dim]配置文件: {self.prompts_file}[/dim]",
            border_style="cyan",
        ))
        self.console.print()
        
        if not self.data:
            self.console.print("[red]无法加载配置文件[/red]")
            for error in self.errors:
                self.console.print(f"  [red]✗[/red] {error}")
            return
        
        # 显示配置概览
        self._print_overview()
        
        # 显示继承关系
        self._print_inheritance_tree()
        
        # 显示错误
        if self.errors:
            self.console.print()
            self.console.print("[bold red]❌ 错误[/bold red]")
            for error in self.errors:
                self.console.print(f"  [red]✗[/red] {error}")
        
        # 显示警告
        if self.warnings:
            self.console.print()
            self.console.print("[bold yellow]⚠ 警告[/bold yellow]")
            for warning in self.warnings:
                self.console.print(f"  [yellow]![/yellow] {warning}")
        
        # 显示优化建议
        if self.suggestions:
            self.console.print()
            self.console.print("[bold blue]💡 继承优化建议[/bold blue]")
            for suggestion in self.suggestions:
                self.console.print(f"  [blue]→[/blue] {suggestion}")
        
        # 显示总结
        self.console.print()
        if self.errors:
            self.console.print(Panel(
                f"[red]发现 {len(self.errors)} 个错误[/red]",
                border_style="red",
            ))
        elif self.suggestions:
            self.console.print(Panel(
                f"[green]配置有效[/green]，但有 [blue]{len(self.suggestions)} 个优化建议[/blue]",
                border_style="green",
            ))
        else:
            self.console.print(Panel(
                "[green]✓ 配置完全有效，无优化建议[/green]",
                border_style="green",
            ))
    
    def _print_overview(self) -> None:
        """打印配置概览"""
        chat_prompts = self.data.get("chat", {})
        eval_prompts = self.data.get("eval", {})
        defaults = self.data.get("defaults", {})
        
        table = Table(title="配置概览", box=box.ROUNDED)
        table.add_column("类别", style="cyan")
        table.add_column("Keys", style="green")
        table.add_column("默认", style="yellow")
        table.add_column("使用继承", style="magenta")
        
        # Chat
        chat_with_extends = sum(
            1 for c in chat_prompts.values() 
            if isinstance(c, dict) and EXTENDS_KEY in c
        )
        table.add_row(
            "chat",
            ", ".join(chat_prompts.keys()),
            defaults.get("chat", "default"),
            f"{chat_with_extends}/{len(chat_prompts)}"
        )
        
        # Eval
        eval_with_extends = sum(
            1 for c in eval_prompts.values() 
            if isinstance(c, dict) and EXTENDS_KEY in c
        )
        table.add_row(
            "eval",
            ", ".join(eval_prompts.keys()),
            defaults.get("eval", "default"),
            f"{eval_with_extends}/{len(eval_prompts)}"
        )
        
        self.console.print(table)
    
    def _print_inheritance_tree(self) -> None:
        """打印继承关系树"""
        chat_prompts = self.data.get("chat", {})
        eval_prompts = self.data.get("eval", {})
        
        has_inheritance = False
        
        for category, prompts in [("chat", chat_prompts), ("eval", eval_prompts)]:
            tree_data, roots = self.get_inheritance_tree(prompts, category)
            
            if tree_data:
                has_inheritance = True
                self.console.print()
                tree = Tree(f"[bold]{category}[/bold] 继承关系")
                
                def add_children(parent_tree, parent_key):
                    children = tree_data.get(parent_key, [])
                    for child in children:
                        child_tree = parent_tree.add(f"[green]{child}[/green] (extends: {parent_key})")
                        add_children(child_tree, child)
                
                for root in roots:
                    root_tree = tree.add(f"[cyan]{root}[/cyan] (root)")
                    add_children(root_tree, root)
                
                # 添加没有父也没有子的孤立节点
                all_in_tree = set(roots)
                for children in tree_data.values():
                    all_in_tree.update(children)
                for key in prompts:
                    if key not in all_in_tree and key not in tree_data:
                        tree.add(f"[dim]{key}[/dim] (standalone)")
                
                self.console.print(tree)
        
        if not has_inheritance:
            self.console.print()
            self.console.print("[dim]当前配置未使用继承[/dim]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Prompt Configuration Checker - 检测 prompts.yaml 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python -m utils.prompt_checker
    python -m utils.prompt_checker ./prompts/prompts.yaml
    python utils/prompt_checker.py --help
        """
    )
    
    parser.add_argument(
        "prompts_file",
        nargs="?",
        default=None,
        help="prompts.yaml 文件路径（默认: prompts/prompts.yaml）"
    )
    
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.prompts_file:
        prompts_file = args.prompts_file
    else:
        # 默认路径
        script_dir = Path(__file__).parent.parent
        prompts_file = script_dir / "prompts" / "prompts.yaml"
    
    # 运行检测
    checker = PromptChecker(prompts_file)
    checker.run_checks()
    checker.print_report()
    
    # 返回退出码
    return 1 if checker.errors else 0


if __name__ == "__main__":
    sys.exit(main())

