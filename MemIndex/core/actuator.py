"""
Actuator - 基准测试执行器

负责执行单个测试序列的步骤。每个 Actuator 对应一个测试主题（如颜色记忆、笑话记忆等）。

核心职责:
    1. 逐步执行测试序列中的各个步骤
    2. 管理消息发送和响应接收
    3. 在评分点调用 LLMEvaluator 进行评分
    4. 处理依赖关系和重试逻辑
    5. 管理懒评估任务

关键概念:
    - step(): 执行单个步骤，发送消息并可能评分
    - 依赖(depend): 某些步骤依赖前置步骤的成功
    - 重试(retry): 评分失败时可以使用重试消息再次尝试
    - 懒评估(lazy_score): 延迟到后续对话后再评分
    - 后处理(post_process): 使用 LLM 从响应中提取关键信息

数据流:
    BenchmarkItem (原始数据)
        → BenchmarkItemExtra (执行过程中的扩展数据)
            → 包含响应、评分结果、重试历史等
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from typing import Optional, TYPE_CHECKING

import tiktoken
from loguru import logger

from components.evaluator.llm_evaluator import LLMEvaluator, TokenUse
from utils.data_loader import BenchmarkItem, BenchmarkItemExtra, format_text

if TYPE_CHECKING:
    from components.agents.base_agent import BaseAgent
    from utils.controller import LLMController
    from prompts import PromptManager


class FakeActuator:
    """
    假执行器

    用于在冻结期间插入废话对话。
    提供与 Actuator 相同的 step() 接口，但实际执行的是废话发送。

    这是一个适配器模式的应用：让 Runner 可以统一处理真正的执行器和废话发送。
    """

    def __init__(self, wrapper):
        """
        初始化假执行器

        Args:
            wrapper: 异步包装函数，通常是 Runner._send_nonsense
        """
        self.wrapper = wrapper

    async def step(self) -> tuple[str, bool, bool, int]:
        """
        执行步骤（实际是发送废话）

        Returns:
            与 Actuator.step() 相同的返回格式
        """
        return await self.wrapper()


class Actuator:
    """
    基准测试执行器

    负责执行单个测试序列的步骤，包括消息发送和评分。

    工作原理:
        1. 每个 Actuator 持有一个测试序列（如 color.json 的数据）
        2. step() 方法逐步执行序列中的每个项
        3. 对于有 score 的项，调用 LLMEvaluator 进行评分
        4. 维护中间状态，支持依赖引用和重试逻辑

    测试序列示例:
        Step 1: "My favorite color is brown." (信息植入，无评分)
        Step 2: "My favorite color is red."   (信息更新，无评分)
        Step 3: "What is my favorite color?"  (提问，需要评分)

    属性说明:
        - data: 测试数据项列表
        - index: 当前执行到的步骤索引
        - intermediate_state: 中间状态，存储每个步骤的执行结果
        - global_tokens: 每次执行时的全局 token 数（用于记忆距离计算）
        - global_index: 每次执行时的全局对话轮次
    """

    def __init__(
        self,
        data: list[BenchmarkItem],
        llm_controller: "LLMController",
        agent: "BaseAgent",
        token_encoder=None,
        eval_model: str = "volcano/deepseek-v3-250324",
        prompt_manager: Optional["PromptManager"] = None,
        eval_prompt_key: str = None,
        eval_mode: str = "binary",
    ):
        """
        初始化执行器

        Args:
            data: 测试数据项列表（BenchmarkItem 对象）
            llm_controller: LLM 控制器，用于调用 LLM
            agent: 被测试的 Agent 实例
            token_encoder: Token 编码器（可选，默认使用 tiktoken）
            eval_model: 评估使用的模型
            prompt_manager: 提示词管理器
            eval_prompt_key: 评估提示词的 key（如 "strict", "lenient"）
            eval_mode: 评估模式 - "binary"(二元) 或 "score"(0-1分数)
        """
        self.data = data
        self.status = True
        self.index = 0  # 当前步骤索引

        # 中间状态数组：索引对应 item.index，存储执行结果
        # 使用 item.index 而非列表索引，因为 index 可能不连续
        self.intermediate_state: list[BenchmarkItemExtra | None] = [None] * (
            max(map(lambda x: x.index, data)) + 1
        )

        self.llm_controller = llm_controller
        self.agent = agent
        self.conversation_history = []  # 该执行器的对话历史

        # Token 编码器，用于计算 token 数量
        self.token_encoder = (
            tiktoken.encoding_for_model("gpt-4o-mini")
            if token_encoder is None
            else token_encoder
        )

        self.global_tokens = []   # 每步执行时的全局 token 数
        self.global_index = []    # 每步执行时的全局对话轮次
        self.eval_model = eval_model
        self.eval_mode = eval_mode

        # 初始化 LLM 评估器
        self.evaluator = LLMEvaluator(
            llm_controller,
            eval_model,
            prompt_manager=prompt_manager,
            eval_prompt_key=eval_prompt_key,
        )

    @property
    def first_mark(self) -> int:
        """
        获取第一个评分点的索引

        用于 Runner 验证执行器配置是否有效。

        Returns:
            第一个有 score 的项的索引，如果没有返回 -1
        """
        for item in self.data:
            if item.score:
                return item.index
        return -1

    @property
    def next_mark(self) -> int:
        """
        获取下一个评分点的索引

        从当前位置开始，找到下一个需要评分的步骤。

        Returns:
            下一个评分点索引，如果没有返回 -1
        """
        for item in self.data:
            if item.index < self.index:
                continue
            if item.score:
                return item.index
        return -1

    @property
    def count_marks(self) -> int:
        """
        统计评分点数量

        Returns:
            该序列中需要评分的步骤数量
        """
        count = 0
        for item in self.data:
            if item.score:
                count += 1
        return count

    @property
    def all_text(self) -> str:
        """
        获取所有对话文本

        将该执行器的对话历史拼接成文本。

        Returns:
            对话文本
        """
        texts = []
        for item in self.conversation_history:
            texts.append(f"{item['role']}:{item['content']}")
        return "\n".join(texts)

    @property
    def current_tokens(self) -> int:
        """
        计算当前 token 数量

        Returns:
            该执行器对话历史的 token 数量
        """
        tokens = self.token_encoder.encode(self.all_text)
        return len(tokens)

    @property
    def is_finished(self) -> bool:
        """检查是否已完成所有步骤"""
        return self.index >= len(self.data)

    @property
    def has_next(self) -> bool:
        """检查是否还有下一步"""
        return self.index < len(self.data)

    @property
    def current_item(self) -> BenchmarkItem | None:
        """获取当前要执行的项"""
        return self.data[self.index] if self.index < len(self.data) else None

    async def step(self) -> tuple[str, bool, bool, int]:
        """
        执行单个步骤

        这是执行器的核心方法，负责：
        1. 检查依赖是否满足
        2. 发送消息给 Agent
        3. 如果有 score，进行评分
        4. 处理重试逻辑
        5. 更新中间状态

        执行流程:
            1. 获取当前步骤数据
            2. 检查依赖（depend）是否都已激活
            3. 如果依赖满足，发送消息
            4. 如果有后处理要求，执行后处理
            5. 如果需要评分，调用评估器
            6. 如果评分失败且有重试消息，进行重试
            7. 更新中间状态并返回结果

        Returns:
            tuple: (响应文本, 是否激活, 是否懒评估, 步骤索引)
            - 响应文本: Agent 的回复
            - 是否激活: 该步骤是否成功（评分通过或无需评分）
            - 是否懒评估: 是否需要延迟评估
            - 步骤索引: 刚执行的步骤索引
        """
        if self.index >= len(self.data):
            return "", False, False, -1

        current_item = self.data[self.index]
        activate = True  # 默认激活

        # ========== 检查依赖 ==========
        # depend 列表中的步骤必须都已执行且激活
        for depend in current_item.depend:
            if int(depend) == current_item.index:
                continue  # 跳过自引用
            if self.intermediate_state[int(depend)] is None:
                activate = False  # 依赖项未执行
            elif not self.intermediate_state[int(depend)].activate:
                activate = False  # 依赖项未激活（评分失败）

        # 创建扩展的测试项，用于存储执行过程中的数据
        item = BenchmarkItemExtra(**current_item.model_dump())
        item.time = datetime.datetime.now()
        token_use = TokenUse(input=0, output=0)  # token 使用统计
        lazy_score = False
        response = ""

        if activate:
            # ========== 执行步骤 ==========
            item.executed = True
            first_times = True
            retry = False

            # 重试循环：第一次执行 + 可能的重试
            while retry or first_times:
                first_times = False
                temp_item = item

                # 如果是重试，保存原始结果并使用重试消息
                if retry:
                    temp_item.retry_frozen_item = deepcopy(item)
                    item = deepcopy(item)
                    if item.retry:
                        item.ask = str(item.retry)  # 使用重试消息
                    item.retry = ""

                # 发送消息给 Agent
                # format_text 会替换引用，如 {1} 替换为步骤1的响应
                response = await self.send_message(
                    format_text(
                        item.ask.replace("\\", ""),
                        self.intermediate_state,
                        [item]
                    )
                )
                item.response = response

                # ========== 后处理 ==========
                # 如果有后处理要求，使用 LLM 从响应中提取信息
                if item.post_process:
                    require = format_text(
                        item.post_process.replace("\\", ""),
                        self.intermediate_state,
                        [item]
                    )
                    item.processed = await self.evaluator.post_process(
                        require,
                        item.response,
                        token_use
                    )

                # ========== 评分 ==========
                if item.score:
                    if not item.score.is_lazy:
                        # 立即评分
                        activate, retry = await self._mark_score(
                            activate, item, response, retry, token_use
                        )
                    else:
                        # 懒评估：延迟到后续对话后评分
                        lazy_score = True

                # 处理重试结果
                if not item.retry and retry:
                    retry = False
                    temp_item.retry_history = item
                    temp_item.response = item.response
                    temp_item.processed = item.processed

                item = temp_item

        # ========== 更新状态 ==========
        item.activate = activate
        self.intermediate_state[item.index] = item  # 保存到中间状态
        self.index += 1

        return response, activate, lazy_score, self.index - 1

    async def _mark_score(
        self,
        activate: bool,
        item: BenchmarkItemExtra,
        response: str,
        retry: bool,
        token_use: TokenUse,
    ) -> tuple[bool, bool]:
        """
        执行评分

        根据评分配置选择合适的评估方式，并更新激活状态。

        评估方式优先级:
            1. binary_items (加权二元评分) - 最推荐，支持多个评分项
            2. is_multiple (多分数评估) - 已弃用，保留兼容性
            3. eval_mode == "score" (0-1分数评估)
            4. 默认 binary (二元评估)

        Args:
            activate: 当前激活状态
            item: 当前执行项
            response: Agent 响应
            retry: 当前重试状态
            token_use: Token 使用统计

        Returns:
            (新激活状态, 是否需要重试)
        """
        # 格式化标准答案（替换引用）
        ground_truth = format_text(
            item.score.answer.replace("\\", ""),
            self.intermediate_state,
            [item]
        )

        # ========== 方式1: 加权二元评分 ==========
        # 多个独立评分项，各自进行二元判断，按权重计算总分
        if item.score.binary_items and len(item.score.binary_items) > 0:
            # 准备评分项数据
            binary_items_data = [
                {
                    "key": bi.key,
                    "weight": bi.weight,
                    "answer": format_text(
                        bi.answer.replace("\\", ""),
                        self.intermediate_state,
                        [item]
                    ),
                }
                for bi in item.score.binary_items
            ]

            # 调用加权二元评估
            result = await self.evaluator.evaluate_weighted_binary(
                binary_items_data,
                response,
                item.score.score,
                token_use,
            )

            try:
                item.score.result = result.score
                item.score.reason = result.reason
                item.score.eval_method = "weighted_binary"

                # 更新每个评分项的结果
                for i, item_result in enumerate(result.item_results):
                    if i < len(item.score.binary_items):
                        item.score.binary_items[i].result = item_result.result
                        item.score.binary_items[i].reason = item_result.reason

            except Exception as e:
                logger.error(f"Weighted binary evaluation failed: {result}")
                item.score.result = 0.0
                item.score.reason = f"Evaluation failed: {str(e)}"
                item.score.eval_method = "weighted_binary"

        # ========== 方式2: 多分数评估（适用于所有 eval_mode）==========
        # is_multiple 表示需要将答案切分成多个子项分别评分
        # 无论是 binary 还是 score 模式，都使用 multi_score_evaluation
        # 这样可以获得更细粒度的评分（如 0.3/1.0, 0.7/1.0 等）
        elif item.score.is_multiple:
            result = await self.evaluator.evaluate_multi_score(
                ground_truth,
                response,
                item.score.score,
                token_use,
            )

            try:
                item.score.result = result.score
                item.score.reason = result.reason
                item.score.eval_method = "multi_score"

            except Exception as e:
                logger.error(f"Multi-score evaluation failed: {result}")
                item.score.result = 0.0
                item.score.reason = f"Evaluation failed: {str(e)}"
                item.score.eval_method = "multi_score"

        else:
            # ========== 方式3, 4: 单一评估 ==========
            if self.eval_mode == "score":
                # 方式3: 0-1 分数评估（更细粒度）
                result = await self.evaluator.evaluate_score(
                    ground_truth,
                    response,
                    item.score.score,
                    token_use,
                )

                try:
                    item.score.result = result.score
                    item.score.reason = result.reason
                    item.score.eval_method = "score"
                except Exception as e:
                    logger.error(f"Score evaluation failed: {result}")
                    item.score.result = 0.0
                    item.score.reason = f"Evaluation failed: {str(e)}"
                    item.score.eval_method = "score"
            else:
                # 方式4: 二元评估（正确/错误）
                result = await self.evaluator.evaluate_binary(
                    ground_truth,
                    response,
                    item.score.score,
                    token_use,
                )

                try:
                    item.score.result = result.score
                    item.score.reason = result.reason
                    item.score.eval_method = "binary"
                except Exception as e:
                    logger.error(f"Evaluation failed: {result}")
                    item.score.result = 0.0
                    item.score.reason = f"Evaluation failed: {str(e)}"
                    item.score.eval_method = "binary"

        # ========== 根据评分结果更新激活状态 ==========
        if self.eval_mode == "score":
            # score 模式：得分 > 0 视为部分成功
            if item.score.result == 0:
                if item.retry:
                    retry = True  # 尝试重试
                else:
                    activate = False  # 标记为失败
        else:
            # binary 模式：必须满分才算成功
            if item.score.result < item.score.score:
                if item.retry:
                    retry = True
                else:
                    activate = False

        return activate, retry

    async def execute_lazy_score(self, index: int, response: str) -> None:
        """
        执行懒评估

        懒评估用于某些需要看后续对话内容才能评分的场景。
        在 lazy_count 轮对话后触发评估。

        Args:
            index: 要评估的步骤索引
            response: 触发评估时的当前响应（可能用于评估上下文）
        """
        current_item = self.data[index]
        item = BenchmarkItemExtra(
            **current_item.model_dump(),
            response=self.intermediate_state[index].response,  # 使用原始响应
        )

        # 执行评分
        await self._mark_score(True, item, response, False, TokenUse(input=0, output=0))
        item.score.lazy_eval_response_snapshot = response  # 记录触发时的响应
        self.intermediate_state[item.index] = item

    async def send_message(self, message: str) -> str:
        """
        发送消息给 Agent

        维护该执行器的对话历史，并调用 Agent 获取响应。

        Args:
            message: 要发送的消息

        Returns:
            Agent 的响应
        """
        # 记录到该执行器的对话历史
        self.conversation_history.append({"role": "user", "content": message})
        # 调用 Agent（Agent 会维护自己的完整对话历史）
        answer = await self.agent.send_message(message)
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer
