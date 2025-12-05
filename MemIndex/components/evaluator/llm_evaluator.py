"""
LLMEvaluator - LLM 评估器

使用 LLM 对答案进行评估，支持二元评估和多分数评估。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel
from loguru import logger

if TYPE_CHECKING:
    from utils.controller import LLMController
    from prompts import PromptManager


def _is_prompt_debug() -> bool:
    """检查是否启用 prompt 调试日志"""
    return os.environ.get("PROMPT_DEBUG_LOG", "").lower() in ("1", "true", "yes", "on")


def _debug_log(message: str) -> None:
    """仅在 PROMPT_DEBUG_LOG 环境变量启用时打印调试日志"""
    if _is_prompt_debug():
        logger.info(f"[PROMPT_DEBUG] {message}")


class TokenUse(BaseModel):
    """Token 使用统计"""
    input: int = 0
    output: int = 0


class EvaluationResult(BaseModel):
    """评估结果"""
    score: float
    max_score: float
    reason: str
    raw_result: Any = None


class BinaryItemResult(BaseModel):
    """单个二元评分项的结果"""
    key: str
    weight: float
    result: bool
    reason: str


class WeightedBinaryResult(BaseModel):
    """加权二元评分的总体结果"""
    score: float
    max_score: float
    reason: str
    item_results: list[BinaryItemResult]


class LLMEvaluator:
    """
    LLM 评估器

    使用 LLM 对答案进行评估。
    """

    def __init__(
        self,
        llm_controller: "LLMController",
        eval_model: str = "volcano/deepseek-v3-250324",
        prompt_manager: Optional["PromptManager"] = None,
        eval_prompt_key: str = None,
    ):
        """
        初始化 LLM 评估器

        Args:
            llm_controller: LLM 控制器实例
            eval_model: 评估使用的模型
            prompt_manager: 提示词管理器（可选，不传则使用全局单例）
            eval_prompt_key: 评估提示词的 key（可选，不传则使用默认）
        """
        self.llm_controller = llm_controller
        self.eval_model = eval_model

        # 提示词管理
        if prompt_manager is None:
            from prompts import get_prompt_manager
            prompt_manager = get_prompt_manager()
        self.prompt_manager = prompt_manager
        self.eval_prompt_key = eval_prompt_key

        _debug_log(f"LLMEvaluator initialized with eval_prompt_key='{eval_prompt_key}'")

    def _get_binary_evaluation_prompt(self, ground: str, target: str, question: str = None) -> str:
        """获取二元评估提示词"""
        _debug_log(f"LLMEvaluator._get_binary_evaluation_prompt() using key='{self.eval_prompt_key}', has_question={bool(question)}")
        template = self.prompt_manager.get_eval_binary_prompt(self.eval_prompt_key, question=question)
        prompt = template.format(ground=ground, target=target)

        if _is_prompt_debug():
            preview = prompt[:200].replace('\n', '\\n') + ('...' if len(prompt) > 200 else '')
            _debug_log(f"Binary evaluation prompt preview: '{preview}'")

        return prompt

    def _get_multi_score_evaluation_prompt(self, ground: str, target: str, question: str = None) -> str:
        """获取多分数评估提示词"""
        _debug_log(f"LLMEvaluator._get_multi_score_evaluation_prompt() using key='{self.eval_prompt_key}', has_question={bool(question)}")
        template = self.prompt_manager.get_eval_multi_score_prompt(self.eval_prompt_key, question=question)
        prompt = template.format(ground=ground, target=target)

        if _is_prompt_debug():
            preview = prompt[:200].replace('\n', '\\n') + ('...' if len(prompt) > 200 else '')
            _debug_log(f"Multi-score evaluation prompt preview: '{preview}'")

        return prompt

    def _get_post_process_prompt(self, require: str, target: str) -> str:
        """获取后处理提示词"""
        _debug_log(f"LLMEvaluator._get_post_process_prompt() using key='{self.eval_prompt_key}'")
        template = self.prompt_manager.get_eval_post_process_prompt(self.eval_prompt_key)
        prompt = template.format(require=require, target=target)

        if _is_prompt_debug():
            preview = prompt[:200].replace('\n', '\\n') + ('...' if len(prompt) > 200 else '')
            _debug_log(f"Post-process prompt preview: '{preview}'")

        return prompt

    def _get_score_evaluation_prompt(self, ground: str, target: str, question: str = None) -> str:
        """获取分数评估提示词（0-1连续分数）"""
        _debug_log(f"LLMEvaluator._get_score_evaluation_prompt() using key='{self.eval_prompt_key}', has_question={bool(question)}")
        template = self.prompt_manager.get_eval_score_prompt(self.eval_prompt_key, question=question)
        prompt = template.format(ground=ground, target=target)

        if _is_prompt_debug():
            preview = prompt[:200].replace('\n', '\\n') + ('...' if len(prompt) > 200 else '')
            _debug_log(f"Score evaluation prompt preview: '{preview}'")

        return prompt

    async def evaluate_binary(
        self,
        ground_truth: str,
        target: str,
        max_score: float = 1.0,
        token_use: TokenUse = None,
        question: str = None,
    ) -> EvaluationResult:
        """
        二元评估（正确/错误）

        Args:
            ground_truth: 黄金标准答案
            target: 目标答案
            max_score: 最大分数
            token_use: Token 使用统计（可选）
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）

        Returns:
            评估结果
        """
        prompt = self._get_binary_evaluation_prompt(ground_truth, target, question)

        result = await self._completion_with_json_check(prompt, token_use)

        if result is None:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason="Evaluation failed: JSON parsing error",
                raw_result=None,
            )

        try:
            score = max_score if result.get("answer", False) else 0.0
            reason = result.get("reason", "")
            return EvaluationResult(
                score=score,
                max_score=max_score,
                reason=reason,
                raw_result=result,
            )
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason=f"Evaluation failed: {str(e)}",
                raw_result=result,
            )

    async def evaluate_score(
        self,
        ground_truth: str,
        target: str,
        max_score: float = 1.0,
        token_use: TokenUse = None,
        question: str = None,
    ) -> EvaluationResult:
        """
        分数评估（0-1 连续分数）

        让 LLM 根据标准答案和目标回答给出 0-1 的分数，最后乘以该题的最大分数。
        这是一种非二元的评估方式，可以给出更细粒度的评分。

        Args:
            ground_truth: 黄金标准答案/评估标准
            target: 目标答案
            max_score: 最大分数
            token_use: Token 使用统计（可选）
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）

        Returns:
            评估结果
        """
        prompt = self._get_score_evaluation_prompt(ground_truth, target, question)

        result = await self._completion_with_json_check(prompt, token_use)

        if result is None:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason="Evaluation failed: JSON parsing error",
                raw_result=None,
            )

        try:
            # 获取 0-1 之间的分数
            raw_score = float(result.get("score", 0))
            # 确保分数在 0-1 范围内
            raw_score = max(0.0, min(1.0, raw_score))
            # 乘以最大分数
            score = raw_score * max_score
            reason = result.get("reason", "")

            return EvaluationResult(
                score=score,
                max_score=max_score,
                reason=f"[Score: {raw_score:.2f}] {reason}",
                raw_result=result,
            )
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason=f"Evaluation failed: {str(e)}",
                raw_result=result,
            )

    async def evaluate_multi_score(
        self,
        ground_truth: str,
        target: str,
        max_score: float = 1.0,
        token_use: TokenUse = None,
        question: str = None,
    ) -> EvaluationResult:
        """
        多分数评估

        Args:
            ground_truth: 黄金标准答案
            target: 目标答案
            max_score: 最大分数
            token_use: Token 使用统计（可选）
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）

        Returns:
            评估结果
        """
        prompt = self._get_multi_score_evaluation_prompt(ground_truth, target, question)

        result = await self._completion_with_json_check(prompt, token_use)

        if result is None:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason="Evaluation failed: JSON parsing error",
                raw_result=None,
            )

        try:
            if not isinstance(result, list):
                raise ValueError("Expected list of scores")

            total_score = sum(float(item.get("score", 0)) for item in result)
            score = total_score * max_score

            if score > max_score:
                raise ValueError("Score exceeds maximum")

            reasons = [f'Score: [{item.get("score", 0)}] - {item.get("reason", "")}' for item in result]
            reason = "\n".join(reasons)

            return EvaluationResult(
                score=score,
                max_score=max_score,
                reason=reason,
                raw_result=result,
            )
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                max_score=max_score,
                reason=f"Evaluation failed: {json.dumps(result, ensure_ascii=False, indent=4)}",
                raw_result=result,
            )

    async def evaluate_weighted_binary(
        self,
        binary_items: list[dict],
        target: str,
        max_score: float = 1.0,
        token_use: TokenUse = None,
        question: str = None,
    ) -> WeightedBinaryResult:
        """
        加权二元评估 - 对多个独立的评分项进行二元判断，然后加权计算总分

        Args:
            binary_items: 二元评分项列表，每项包含 key, weight, answer
            target: 目标答案
            max_score: 最大分数
            token_use: Token 使用统计（可选）
            question: 可选的问题内容（本轮问答发送给 LLM 的 query）

        Returns:
            加权二元评估结果
        """
        item_results: list[BinaryItemResult] = []
        total_weight = sum(item.get("weight", 0) for item in binary_items)
        weighted_score = 0.0
        reasons = []

        for item in binary_items:
            key = item.get("key", "")
            weight = item.get("weight", 0)
            answer = item.get("answer", "")

            # 对每个项进行二元评估
            result = await self.evaluate_binary(
                ground_truth=answer,
                target=target,
                max_score=1.0,  # 内部使用 1.0，后面再按权重计算
                token_use=token_use,
                question=question,
            )

            is_correct = result.score > 0
            item_result = BinaryItemResult(
                key=key,
                weight=weight,
                result=is_correct,
                reason=result.reason,
            )
            item_results.append(item_result)

            # 计算加权分数
            if is_correct:
                weighted_score += weight

            # 记录理由
            status = "✓" if is_correct else "✗"
            reasons.append(f"[{status}] {key} (weight: {weight}): {result.reason}")

        # 归一化分数到 max_score
        if total_weight > 0:
            final_score = (weighted_score / total_weight) * max_score
        else:
            final_score = 0.0

        return WeightedBinaryResult(
            score=final_score,
            max_score=max_score,
            reason="\n".join(reasons),
            item_results=item_results,
        )

    async def post_process(
        self,
        require: str,
        target: str,
        token_use: TokenUse = None,
    ) -> str:
        """
        后处理文本

        Args:
            require: 处理要求
            target: 目标文本
            token_use: Token 使用统计（可选）

        Returns:
            处理后的文本
        """
        prompt = self._get_post_process_prompt(require, target)

        result = await self.llm_controller.completion_with_retry_async_t(
            self.eval_model,
            prompt,
        )

        if token_use and result:
            token_use.input += result.token_information.input_tokens
            token_use.output += result.token_information.output_tokens

        return result.completion if result else ""

    async def _completion_with_json_check(
        self,
        prompt: str,
        token_use: TokenUse = None,
        max_try: int = 3,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int=42,
    ) -> Dict[str, Any] | list | None:
        """
        执行 LLM 调用并解析 JSON 结果

        Args:
            prompt: 提示词
            token_use: Token 使用统计（可选）
            max_try: 最大重试次数
            temperature: 生成温度
            top_p: 核采样温度
            top_k: 核采样数量
            seed: 随机种子

        Returns:
            解析后的 JSON 结果
        """
        for _ in range(max_try):
            # 使用带重试的方法，网络错误会自动重试
            completion = await self.llm_controller.completion_with_retry_async_t(
                self.eval_model,
                prompt,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                extra_body={
                    "top_k": top_k,        # 核心设置：强制贪婪解码 (Greedy Decoding)
                }
            )

            # 如果 completion 为 None，说明重试后仍然失败
            if completion is None:
                logger.warning("LLM completion failed after retries, trying again...")
                continue

            if token_use:
                token_use.input += completion.token_information.input_tokens
                token_use.output += completion.token_information.output_tokens

            # 尝试解析 JSON
            markdown_regex = r"```((json)|(JSON))(?P<json>[\s\S]+?)```"
            try:
                match = re.search(markdown_regex, completion.completion)
                if match:
                    data = match.group("json")
                    result = json.loads(data)
                else:
                    result = json.loads(completion.completion)
                return result
            except json.JSONDecodeError:
                continue

        return None
