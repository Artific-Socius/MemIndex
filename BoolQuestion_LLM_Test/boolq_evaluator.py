#!/usr/bin/env python3
"""
BoolQ LLM评估实验 - 主入口文件

功能:
- 异步并发评估多个LLM模型的二值分类能力
- 支持多种提示词风格和输出格式
- 支持logprobs分析
- 实时进度展示
- Token使用量和成本统计

使用示例:
    python main.py --model google/gemini-2.0-flash-001 --style sse --limit 100
    python main.py --model deepseek-v3-250324 --style direct --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig
from utils.models_utils import BoolQItem, EvaluationResult, ExperimentStats, PromptStyle, TokenUsage, CostInfo, EvalMode
from llm_client import create_llm_client, BaseLLMClient
from prompt_manager import PromptManager
from utils import DatasetManager, ExperimentLogger, AsyncResultWriter, ResponseParser, StatisticsHelper
from i18n import t, set_language, get_language


class BoolQEvaluator:
    """
    BoolQ异步评估器
    
    负责:
    - 管理异步评估任务
    - 协调各组件工作
    - 收集和统计结果
    """
    
    def __init__(self, config: ExperimentConfig, console: Console):
        self.config = config
        self.console = console
        
        # 初始化组件
        self.logger_manager = ExperimentLogger(
            output_dir=config.output_dir,
            model_name=config.model,
            style=config.style.value,
            use_reasoning=config.use_reasoning,
            reason_order=config.reason_order.value,
            console=console,
        )
        
        self.dataset_manager = DatasetManager(
            split=config.split,
            limit=config.limit,
            dirty_data_path=config.dirty_data_path,
        )
        
        self.prompt_manager = PromptManager(
            style=config.style,
            eval_mode=config.eval_mode,
            use_reasoning=config.use_reasoning,
            reason_order=config.reason_order,
        )
        
        self.response_parser = ResponseParser(style=config.style)
        
        self.llm_client = create_llm_client(config.get_llm_config())
        
        # 统计数据
        self.stats = ExperimentStats()
        
        # 异步写入器
        self.result_writer: Optional[AsyncResultWriter] = None
        
        # 信号量控制并发
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # 任务计数器
        self._processing_count = 0  # 正在处理中的任务数
        self._completed_count = 0   # 已完成的任务数
        self._counter_lock = asyncio.Lock()
    
    async def _increment_processing(self) -> None:
        """增加处理中计数"""
        async with self._counter_lock:
            self._processing_count += 1
    
    async def _decrement_processing(self) -> None:
        """减少处理中计数，增加完成计数"""
        async with self._counter_lock:
            self._processing_count -= 1
            self._completed_count += 1
    
    # 解析失败最大重试次数
    PARSE_RETRY_COUNT = 2
    
    async def evaluate_single(
        self,
        item: BoolQItem,
        reversal: bool,
    ) -> EvaluationResult:
        """
        评估单条数据（带解析失败重试）
        
        Args:
            item: BoolQ数据项
            reversal: 是否反转预期答案 (仅在validate模式下有效)
            
        Returns:
            EvaluationResult: 评估结果
        """
        # 根据评估模式准备prompt
        if self.config.eval_mode == EvalMode.ANSWER:
            # answer模式：不需要preset_answer，不使用reversal
            prompt = self.prompt_manager.create_prompt(
                question=item.question,
                passage=item.passage,
            )
            # answer模式下reversal无效
            reversal = False
        else:
            # validate模式：需要preset_answer
            preset_answer = item.answer if not reversal else (not item.answer)
            prompt = self.prompt_manager.create_prompt(
                question=item.question,
                passage=item.passage,
                preset_answer=preset_answer,
            )
        
        await self._increment_processing()
        
        # 累计token和成本（用于多次重试时合并统计）
        total_latency = 0.0
        last_llm_response = None
        last_parse_result = None
        last_error = None
        
        try:
            # 解析失败重试循环
            for attempt in range(self.PARSE_RETRY_COUNT + 1):
                try:
                    # 使用信号量控制并发
                    async with self._semaphore:
                        llm_response = await self.llm_client.generate(prompt)
                    
                    last_llm_response = llm_response
                    total_latency += llm_response.latency
                    
                    # 解析响应
                    parse_result = self.response_parser.parse(llm_response.content)
                    last_parse_result = parse_result
                    
                    # 解析成功，跳出重试循环
                    if parse_result.success:
                        break
                    
                    # 解析失败，如果还有重试次数，记录日志并重试
                    if attempt < self.PARSE_RETRY_COUNT:
                        logger.warning(
                            f"解析失败 (index={item.index}, 尝试 {attempt + 1}/{self.PARSE_RETRY_COUNT + 1}): "
                            f"{parse_result.error_message[:80]}... 重试中"
                        )
                    
                except Exception as e:
                    last_error = e
                    # API错误，如果还有重试次数，记录并重试
                    if attempt < self.PARSE_RETRY_COUNT:
                        logger.warning(
                            f"API错误 (index={item.index}, 尝试 {attempt + 1}/{self.PARSE_RETRY_COUNT + 1}): "
                            f"{type(e).__name__}: {str(e)[:80]}... 重试中"
                        )
                    else:
                        raise  # 最后一次尝试仍然失败，抛出异常
            
            # 使用最后一次的响应结果
            llm_response = last_llm_response
            parse_result = last_parse_result
            
            if llm_response is None or parse_result is None:
                raise last_error or Exception("未知错误")
            
            # 判断正确性
            is_correct = False
            if parse_result.success and parse_result.answer is not None:
                if self.config.eval_mode == EvalMode.ANSWER:
                    # answer模式：直接比较LLM回答与原始答案
                    is_correct = parse_result.answer == item.answer
                else:
                    # validate模式：LLM回答True表示验证通过，比较是否与非反转一致
                    is_correct = parse_result.answer == (not reversal)
            
            result = EvaluationResult(
                status="success" if parse_result.success else "parse_error",
                question=item.question,
                passage=item.passage,
                expected=item.answer,
                is_reversal=reversal,
                predicted=parse_result.answer,
                is_correct=is_correct,
                raw_response=llm_response.content,
                parsed_reason=parse_result.reason,
                latency=total_latency,  # 使用累计延迟
                timestamp=datetime.now().isoformat(),
                index=item.index,
                item_hash=item.hash,
                avg_logprobs=llm_response.avg_logprobs,
                confidence=llm_response.confidence,
                logprob_diff=llm_response.logprob_diff,
                logprobs=[lp.__dict__ for lp in (llm_response.logprobs or [])] if llm_response.logprobs else None,
                token_usage=llm_response.token_usage.to_dict() if llm_response.token_usage else None,
                cost_info=llm_response.cost_info.to_dict() if llm_response.cost_info else None,
                error=parse_result.error_message if not parse_result.success else None,
            )
            
            # 记录单项结果日志 (带颜色)
            self.logger_manager.log_item_result(
                index=item.index,
                is_correct=is_correct,
                predicted=parse_result.answer,
                expected=item.answer,
                latency=total_latency,
                avg_logprobs=llm_response.avg_logprobs,
                token_usage=llm_response.token_usage,
                cost=llm_response.cost_info.total_cost if llm_response.cost_info else None,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"评估错误 (index={item.index}): {type(e).__name__}: {e}")
            return EvaluationResult(
                status="api_error",
                question=item.question,
                passage=item.passage,
                expected=item.answer,
                is_reversal=reversal,
                predicted=None,
                is_correct=False,
                raw_response=last_llm_response.content if last_llm_response else "",
                parsed_reason="",
                latency=total_latency,
                timestamp=datetime.now().isoformat(),
                index=item.index,
                item_hash=item.hash,
                error=str(e),
            )
        finally:
            await self._decrement_processing()
    
    async def run(self) -> None:
        """运行评估"""
        # 加载数据
        self.dataset_manager.load()
        total_items = len(self.dataset_manager)
        
        # 显示脏数据统计
        dirty_stats = self.dataset_manager.dirty_stats
        if dirty_stats and dirty_stats.valid_records > 0:
            dirty_in_current = self.dataset_manager.dirty_count
            self.console.print(
                f"[dim]📋 {t('脏数据: 已加载 {count} 条哈希, 当前数据集将跳过 {skip} 条', count=len(self.dataset_manager._dirty_hashes), skip=dirty_in_current)}[/dim]"
            )
        
        # 使用新的日志方法
        self.logger_manager.log_evaluation_start(total_items, self.config.concurrency)
        
        # 初始化信号量
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        
        # 初始化异步写入器
        self.result_writer = AsyncResultWriter(self.logger_manager.data_path)
        await self.result_writer.start()
        
        # 收集所有任务
        items = list(self.dataset_manager)
        
        # 创建进度条 - 改进的显示
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
            refresh_per_second=10,
        ) as progress:
            
            task_id = progress.add_task(
                f"[cyan]评估 {self._get_short_model_name()}[/cyan]",
                total=len(items),
            )
            
            # 创建所有评估任务
            async def evaluate_with_progress(item: BoolQItem) -> EvaluationResult:
                # answer模式不使用reversal
                if self.config.eval_mode == EvalMode.ANSWER:
                    reversal = False
                else:
                    reversal = random.random() < self.config.reversal_ratio
                result = await self.evaluate_single(item, reversal)
                
                # 构建token_usage和cost_info对象
                token_usage = None
                cost_info = None
                if result.token_usage:
                    token_usage = TokenUsage(**result.token_usage)
                if result.cost_info:
                    cost_info = CostInfo(**result.cost_info)
                
                # 更新统计
                self.stats.update(
                    is_correct=result.is_correct,
                    parsed_successfully=(result.status == "success"),
                    avg_logprobs=result.avg_logprobs,
                    filter_threshold=self.config.filter_threshold,
                    token_usage=token_usage,
                    cost_info=cost_info,
                )
                
                # 写入结果
                await self.result_writer.write(result.to_dict())
                
                # 更新进度条描述 - 新格式
                pending_write = self.result_writer.pending_count
                written = self.result_writer.written_count
                
                # Token和成本统计
                total_tokens = self.stats.total_token_usage.total_tokens
                total_cost = self.stats.total_cost.total_cost
                completed = self.stats.total  # 已完成的任务数
                total_items_count = len(items)  # 总任务数
                
                # 计算预估总成本
                if completed > 0:
                    avg_cost_per_item = total_cost / completed
                    estimated_total_cost = avg_cost_per_item * total_items_count
                else:
                    estimated_total_cost = 0.0
                
                # 格式化token数（K表示千）
                if total_tokens >= 1000:
                    token_str = f"{total_tokens/1000:.1f}K"
                else:
                    token_str = str(total_tokens)
                
                # 格式化成本：当前成本 → 预估总成本
                def format_cost(cost: float) -> str:
                    if cost < 0.01:
                        return f"${cost:.4f}"
                    elif cost < 1:
                        return f"${cost:.3f}"
                    else:
                        return f"${cost:.2f}"
                
                cost_str = f"{format_cost(total_cost)}→{format_cost(estimated_total_cost)}"
                
                progress.update(
                    task_id,
                    advance=1,
                    description=(
                        f"[cyan]评估[/cyan] | "
                        f"[green]Acc: {self.stats.accuracy:.1%}[/green] | "
                        f"[green]✓{self.stats.correct}[/green] "
                        f"[red]✗{self.stats.total - self.stats.correct}[/red] "
                        f"[yellow]⚠{self.stats.errors}[/yellow] | "
                        f"[dim]tok:{token_str}[/dim] "
                        f"[cyan]{cost_str}[/cyan]"
                    ),
                )
                
                return result
            
            # 并发执行所有任务
            tasks = [evaluate_with_progress(item) for item in items]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 停止写入器
        await self.result_writer.stop()
        
        # 处理异常结果
        exception_count = 0
        for result in results:
            if isinstance(result, Exception):
                exception_count += 1
                logger.error(f"任务异常: {type(result).__name__}: {result}")
        
        if exception_count > 0:
            logger.warning(f"共有 {exception_count} 个任务发生异常")
        
        # 打印总结
        self._print_summary()
    
    def _get_short_model_name(self) -> str:
        """获取简短的模型名称用于显示"""
        model = self.config.model
        if "/" in model:
            model = model.split("/")[-1]
        # 截断过长的名称
        if len(model) > 30:
            model = model[:27] + "..."
        return model
    
    def _print_summary(self) -> None:
        """打印评估总结"""
        # Token统计
        token_usage = self.stats.total_token_usage
        cost = self.stats.total_cost
        
        mode_desc = t("验证答案") if self.config.eval_mode == EvalMode.VALIDATE else t("直接回答")
        summary = (
            f"\n{'='*60}\n"
            f"{t('评估结果总结')}\n"
            f"{'='*60}\n"
            f"{t('模型')}: {self.config.model}\n"
            f"Provider: {self.config.provider.value}\n"
            f"{t('评估模式')}: {self.config.eval_mode.value} ({mode_desc})\n"
            f"{t('风格')}: {self.config.style.value}\n"
            f"{t('推理')}: {self.config.use_reasoning}\n"
            f"{'='*60}\n"
            f"{t('总样本数')}: {self.stats.total + self.stats.errors}\n"
            f"{t('有效解析')}: {self.stats.total}\n"
            f"{t('正确')}: {self.stats.correct}\n"
            f"{t('错误')}: {self.stats.total - self.stats.correct}\n"
            f"{t('解析/API错误')}: {self.stats.errors}\n"
            f"{'='*60}\n"
            f"{t('准确率')}: {self.stats.accuracy:.2%}\n"
            f"{'='*60}\n"
            f"{t('Token统计')}:\n"
            f"  {t('输入Token')}: {token_usage.prompt_tokens:,}\n"
            f"  {t('输出Token')}: {token_usage.completion_tokens:,}\n"
            f"  {t('总Token')}: {token_usage.total_tokens:,}\n"
        )
        
        if token_usage.reasoning_tokens > 0:
            summary += f"  {t('推理Token')}: {token_usage.reasoning_tokens:,}\n"
        
        summary += (
            f"{'='*60}\n"
            f"{t('成本统计 (USD)')}:\n"
            f"  {t('输入成本')}: ${cost.prompt_cost:.6f}\n"
            f"  {t('输出成本')}: ${cost.completion_cost:.6f}\n"
            f"  {t('总成本')}: ${cost.total_cost:.6f}\n"
        )
        
        if cost.prompt_price_per_m > 0:
            summary += (
                f"  Price (per M tokens): input=${cost.prompt_price_per_m:.2f}, output=${cost.completion_price_per_m:.2f}\n"
            )
        
        summary += (
            f"{'='*60}\n"
            f"{t('LogProbs统计')}:\n"
            f"  {t('平均LogProbs (全部)')}: {self.stats.avg_logprobs_all:.4f}\n"
            f"  {t('平均LogProbs (正确)')}: {self.stats.avg_logprobs_correct_samples:.4f}\n"
            f"  {t('平均LogProbs (错误)')}: {self.stats.avg_logprobs_fail_samples:.4f}\n"
            f"{'='*60}\n"
        )
        
        self.console.print(Panel(summary, title=f"[bold green]{t('评估完成')}[/bold green]"))
        
        # summary写入日志文件（不在控制台重复显示）
        self._write_to_log_file(summary)
        
        # LogProbs分布只输出到日志，不输出到控制台
        self._log_logprobs_distribution()
    
    def _write_to_log_file(self, content: str) -> None:
        """直接写入日志文件（不在控制台显示）"""
        from datetime import datetime
        log_path = self.logger_manager.log_path
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(log_path, "a", encoding="utf-8") as f:
            for line in content.strip().split("\n"):
                f.write(f"{timestamp} | INFO     | Summary - {line}\n")
    
    def _log_logprobs_distribution(self) -> None:
        """将LogProbs分布直接写入日志文件（不在控制台显示）"""
        import io
        import sys
        from datetime import datetime
        
        # 直接写入日志文件，不通过logger（避免输出到控制台）
        log_path = self.logger_manager.log_path
        
        def capture_output(func, *args, **kwargs) -> str:
            """捕获函数的stdout输出"""
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                func(*args, **kwargs)
                return sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
        
        def write_to_log(content: str) -> None:
            """直接写入日志文件"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(log_path, "a", encoding="utf-8") as f:
                for line in content.strip().split("\n"):
                    f.write(f"{timestamp} | INFO     | LogProbs分布 - {line}\n")
        
        if self.stats.avg_logprobs_list:
            write_to_log("\n=== LogProbs分布 (全部) ===")
            output = capture_output(StatisticsHelper.print_distribution_summary, self.stats.avg_logprobs_list)
            write_to_log(output)
            output = capture_output(StatisticsHelper.print_text_histogram_quantile, self.stats.avg_logprobs_list, 15, "█", 80)
            write_to_log(output)
        
        if self.stats.avg_logprobs_list_correct:
            write_to_log("\n=== LogProbs分布 (正确) ===")
            output = capture_output(StatisticsHelper.print_distribution_summary, self.stats.avg_logprobs_list_correct)
            write_to_log(output)
            output = capture_output(StatisticsHelper.print_text_histogram_quantile, self.stats.avg_logprobs_list_correct, 15, "▒", 80)
            write_to_log(output)
        
        if self.stats.avg_logprobs_list_fail:
            write_to_log("\n=== LogProbs分布 (错误) ===")
            output = capture_output(StatisticsHelper.print_distribution_summary, self.stats.avg_logprobs_list_fail)
            write_to_log(output)
            output = capture_output(StatisticsHelper.print_text_histogram_quantile, self.stats.avg_logprobs_list_fail, 15, "▓", 80)
            write_to_log(output)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BoolQ LLM评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用OpenRouter评估Gemini
  python main.py --model google/gemini-2.0-flash-001 --style sse --limit 100

  # 使用火山引擎评估DeepSeek
  python main.py --model deepseek-v3-250324 --style direct --concurrency 20

  # 使用Vertex AI评估Gemini (Gemini模型自动使用Vertex AI)
  python main.py --model gemini-2.5-flash --style json --reasoning
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="google/gemini-2.0-flash-001",
        help="模型名称 (默认: google/gemini-2.0-flash-001)"
    )
    
    parser.add_argument(
        "--style", "-s",
        type=str,
        choices=["direct", "sse", "json"],
        required=True,
        help="提示词和解析风格"
    )
    
    parser.add_argument(
        "--eval-mode", "-e",
        type=str,
        choices=["validate", "answer"],
        default="validate",
        help="评估模式: validate=验证答案正确性, answer=直接回答问题 (默认: validate)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="限制数据条数, 0表示全部 (默认: 0)"
    )
    
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="启用详细推理"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "all"],
        help="数据集分割: train, validation, 或 all (同时加载两者) (默认: validation)"
    )
    
    parser.add_argument(
        "--reversal", "-r",
        type=float,
        default=0.3,
        help="答案反转比例 (默认: 0.3)"
    )
    
    parser.add_argument(
        "--reason-order",
        type=str,
        default="reason-after",
        choices=["reason-first", "reason-after"],
        help="推理顺序 (默认: reason-after)"
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="最大并发数 (默认: 10)"
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "en"],
        default=None,
        help="语言 (zh=中文, en=English), 也可设置环境变量 BOOLQ_LANG"
    )
    
    return parser.parse_args()


async def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 设置语言
    if args.lang:
        set_language(args.lang)
    
    # 创建配置
    config = ExperimentConfig.from_args(args)
    
    # 创建Console (在配置后创建，确保日志使用同一个console)
    console = Console()
    
    # 打印配置信息
    mode_desc = t("验证答案") if config.eval_mode == EvalMode.VALIDATE else t("直接回答")
    reversal_info = f"\n[bold]{t('反转比例')}:[/bold] {config.reversal_ratio}" if config.eval_mode == EvalMode.VALIDATE else ""
    console.print(Panel(
        f"[bold]{t('模型')}:[/bold] {config.model}\n"
        f"[bold]Provider:[/bold] {config.provider.value}\n"
        f"[bold]{t('评估模式')}:[/bold] {config.eval_mode.value} ({mode_desc})\n"
        f"[bold]{t('风格')}:[/bold] {config.style.value}\n"
        f"[bold]{t('推理')}:[/bold] {config.use_reasoning}\n"
        f"[bold]{t('数据集')}:[/bold] {config.split} (limit={config.limit})\n"
        f"[bold]{t('并发数')}:[/bold] {config.concurrency}{reversal_info}",
        title=f"[bold cyan]{t('BoolQ评估配置')}[/bold cyan]"
    ))
    
    # 创建评估器并运行
    evaluator = BoolQEvaluator(config, console)
    
    try:
        await evaluator.run()
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{t('评估被用户中断')}[/yellow]")
    except Exception as e:
        console.print(f"\n[red]评估失败: {e}[/red]")
        logger.exception("评估异常")
        raise


if __name__ == "__main__":
    asyncio.run(main())
