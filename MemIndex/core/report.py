"""
Report - 基准测试报告

负责生成和保存测试报告。

核心职责:
    1. 汇总测试结果数据
    2. 计算总分和各项得分
    3. 生成结构化的 JSON 报告
    4. 安全地保存报告文件（支持并发写入）

报告结构:
    - 主报告文件: [report]-{benchmark_name}.json
        - 索引文件，记录所有测试结果文件
    - 结果文件: benchmark-result-[N]-{timestamp}.json
        - 具体的测试结果数据

文件锁机制:
    为支持批量并行运行，使用 portalocker 实现文件锁，
    确保多个进程同时写入时数据不会损坏。
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Literal, TYPE_CHECKING

import portalocker
from loguru import logger
from pydantic import BaseModel, Field

from utils.data_loader import BenchmarkItemExtra, ref_change

if TYPE_CHECKING:
    from core.runner import Runner


class ReportMetaData(BaseModel):
    """
    报告元数据
    
    用于存储额外的统计信息，如 Token 节省表格等。
    
    Attributes:
        type: 数据类型 (text/number/boolean/array/table)
        description: 描述
        value: 值
    """
    type: Literal["text", "number", "boolean", "array", "table"]
    description: str
    value: Any


class ReportStructure(BaseModel):
    """
    报告结构
    
    定义报告的完整数据结构，用于序列化为 JSON。
    
    Attributes:
        config_path: 测试配置文件路径
        time_start: 测试开始时间戳
        time_end: 测试结束时间戳
        time_usage: 总耗时（秒）
        score: 获得的总分
        total_score: 可能的最高分
        total_sequence: 完整的执行序列（按时间顺序）
        split_sequence: 按执行器分组的序列
        split_global_index: 各执行器步骤对应的全局对话轮次
        split_global_tokens: 各执行器步骤对应的全局 token 数
        extra_metadata: 额外元数据（如 Token 统计表）
        agent: Agent 类型名称
        benchmark_name: 测试名称
        full_tokens: 总 token 数量
        eval_mode: 评估模式 (binary/score)
        chat_prompt: Chat 提示词 key
        eval_prompt: Eval 提示词 key
    """
    config_path: str
    time_start: float
    time_end: float
    time_usage: float
    score: float
    total_score: float
    total_sequence: list[BenchmarkItemExtra]
    split_sequence: dict[str, list[BenchmarkItemExtra | None]]
    split_global_index: dict[str, list[int]] = Field(default_factory=dict)
    split_global_tokens: dict[str, list[int]] = Field(default_factory=dict)
    extra_metadata: dict[str, ReportMetaData] = Field(default_factory=dict)
    agent: str
    benchmark_name: str
    full_tokens: int = 0
    eval_mode: str = "binary"
    chat_prompt: str = ""
    eval_prompt: str = ""


class ReportMainFile(BaseModel):
    """
    报告主文件（索引文件）
    
    作为所有测试结果的索引，记录每次测试生成的结果文件。
    
    Attributes:
        abs_path: 主文件绝对路径
        folder: 结果文件存储目录
        current_benchmarks: 所有结果文件名列表
        create_time: 创建时间
        update_time: 最后更新时间
        agent: Agent 类型
        model: 使用的模型
    """
    abs_path: str
    folder: str
    current_benchmarks: list[str]
    create_time: str
    update_time: str
    agent: str
    model: str = "N/A"


class Report:
    """
    基准测试报告
    
    负责生成和保存测试报告。
    
    工作流程:
        1. 从 Runner 获取运行历史和执行器数据
        2. 构建完整的执行序列（包含废话对话）
        3. 计算总分和各项得分
        4. 处理时间字段格式
        5. 安全保存到文件（使用文件锁）
    
    文件结构:
        reports/
        ├── [report]-benchmark_name.json      # 主文件（索引）
        └── benchmark_name/
            ├── benchmark-result-[0]-time.json # 第1次测试结果
            ├── benchmark-result-[1]-time.json # 第2次测试结果
            └── ...
    """
    
    def __init__(
        self,
        report_path: str,
        config_path: str,
        time_start: float,
        time_end: float,
        runner: "Runner",
        actuator_names: list[str],
        full_tokens: int,
        agent: str = "",
        benchmark_name: str = "",
        model: str = "N/A",
        extra_metadata: dict[str, dict] = None,
        eval_mode: str = "binary",
        chat_prompt: str = "",
        eval_prompt: str = "",
    ):
        """
        初始化报告
        
        Args:
            report_path: 报告保存目录
            config_path: 测试配置文件路径
            time_start: 测试开始时间戳
            time_end: 测试结束时间戳
            runner: 运行器实例（包含完整的测试数据）
            actuator_names: 执行器名称列表（如 ["color", "joke"]）
            full_tokens: 总 token 数量
            agent: Agent 类型名称
            benchmark_name: 测试名称（用于文件命名）
            model: 使用的模型名称
            extra_metadata: 额外元数据（如 Agent 的统计信息）
            eval_mode: 评估模式 (binary/score)
            chat_prompt: Chat 提示词 key
            eval_prompt: Eval 提示词 key
        """
        self.report_path = report_path
        self.config_path = config_path
        self.time_start = time_start
        self.time_end = time_end
        self.time_usage = time_end - time_start
        self.runner = runner
        self.actuator_names = actuator_names
        self.agent = agent
        self.benchmark_name = benchmark_name
        self.full_tokens = full_tokens
        self.model = model.split('/')[-1]  # 只取模型名称部分
        self.eval_mode = eval_mode
        self.chat_prompt = chat_prompt or "default"
        self.eval_prompt = eval_prompt or "default"
        self.extra_metadata = {}
        
        # 解析额外元数据
        if extra_metadata:
            try:
                for k, v in extra_metadata.items():
                    try:
                        self.extra_metadata[k] = ReportMetaData(**v)
                    except Exception as e:
                        logger.warning(f"Invalid extra metadata for key {k}: {v}. Error: {str(e)}")
            except Exception as e:
                logger.warning(f"Invalid extra metadata format: {extra_metadata}. Error: {str(e)}")
        
        # 构建报告数据
        self.report = self._build_report()
    
    @property
    def report_main_file(self) -> str:
        """获取报告主文件路径"""
        return os.path.join(self.report_path, f"[report]-{self.benchmark_name}.json")
    
    def _build_report(self) -> ReportStructure:
        """
        构建报告数据结构
        
        从 Runner 的运行历史中提取数据，构建完整的报告。
        
        处理逻辑:
            1. 遍历运行历史，收集引用映射
            2. 更新各执行器的引用索引
            3. 构建完整的时间序列
            4. 计算总分
            5. 处理时间字段格式
        
        Returns:
            完整的报告结构
        """
        actuators = self.runner.actuators
        
        # 引用映射：记录原始索引到全局索引的映射
        # 用于更新报告中的引用（如 {1} 引用变成 {15}）
        ref_map: list[list[tuple[int, int]]] = [[] for _ in actuators]
        modified_sequence: list[list[BenchmarkItemExtra]] = []
        
        # 第一遍：收集引用映射
        for index, (actuator_type, actuator_idx, inner_idx) in enumerate(self.runner.running_history):
            if actuator_type == "actuator":
                actuator = actuators[actuator_idx]
                item = actuator.intermediate_state[inner_idx]
                # (原始索引, 全局索引)
                ref_map[actuator_idx].append((item.index, index + 1))
        
        # 第二遍：更新引用
        for index, sequence in enumerate([x.intermediate_state for x in actuators]):
            modified_sequence.append(ref_change(sequence, ref_map[index]))
        
        # 构建完整的时间序列（按执行顺序）
        total_sequence: list[BenchmarkItemExtra] = []
        score: float = 0       # 获得的总分
        total_score: float = 0 # 可能的最高分
        
        for index, (actuator_type, actuator_idx, inner_idx) in enumerate(self.runner.running_history):
            if actuator_type == "actuator":
                # 执行器步骤
                item = modified_sequence[actuator_idx][inner_idx]
                item.index = index + 1  # 更新为全局索引
                total_sequence.append(item)
                # 累计得分
                if item.score:
                    score += item.score.result
                    total_score += item.score.score
            if actuator_type == "nonsense":
                # 废话步骤
                item = BenchmarkItemExtra(
                    index=index + 1,
                    ask=self.runner.conversation_history[inner_idx]['content'],
                    response=self.runner.conversation_history[inner_idx + 1]['content'],
                )
                total_sequence.append(item)
        
        # 按执行器分组的序列
        split_sequence = {
            self.actuator_names[i]: x.intermediate_state 
            for i, x in enumerate(actuators)
        }
        
        # 处理时间字段：将 datetime 转换为 ISO 格式字符串
        process_queue = [x for x in total_sequence] + [
            y for x in list(split_sequence.values()) for y in x
        ]
        
        while process_queue:
            item = process_queue.pop(0)
            if item:
                # 转换时间格式
                if item.time and not isinstance(item.time, str):
                    item.time = item.time.isoformat()
                    
                # 递归处理重试历史
                if item.retry_history:
                    process_queue.append(item.retry_history)
                if item.retry_frozen_item:
                    process_queue.append(item.retry_frozen_item)
        
        return ReportStructure(
            config_path=self.config_path,
            time_start=self.time_start,
            time_end=self.time_end,
            time_usage=self.time_usage,
            score=score,
            total_score=total_score,
            total_sequence=total_sequence,
            split_sequence={
                self.actuator_names[i]: x.intermediate_state 
                for i, x in enumerate(actuators)
            },
            split_global_index={
                self.actuator_names[i]: x.global_index 
                for i, x in enumerate(actuators)
            },
            split_global_tokens={
                self.actuator_names[i]: x.global_tokens 
                for i, x in enumerate(actuators)
            },
            agent=self.agent,
            benchmark_name=self.benchmark_name,
            extra_metadata=self.extra_metadata,
            full_tokens=self.full_tokens,
            eval_mode=self.eval_mode,
            chat_prompt=self.chat_prompt,
            eval_prompt=self.eval_prompt,
        )
    
    def _safe_save_with_lock(self, max_retries: int = 30, retry_delay: float = 0.5) -> bool:
        """
        使用文件锁安全保存报告
        
        为支持批量并行运行，使用文件锁确保：
        1. 多个进程不会同时写入同一文件
        2. 报告索引正确更新
        3. 失败时自动重试
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            
        Returns:
            是否保存成功
        """
        for attempt in range(max_retries):
            try:
                # 确保报告目录存在
                if not os.path.exists(self.report_path):
                    os.makedirs(self.report_path, exist_ok=True)
                
                lock_file_path = self.report_main_file + ".lock"
                
                # 如果主文件不存在，创建空文件
                if not os.path.exists(self.report_main_file):
                    with open(self.report_main_file, 'w') as f:
                        json.dump({}, f)
                
                # 使用文件锁
                with open(lock_file_path, 'w') as lock_file:
                    try:
                        # 尝试获取排他锁（非阻塞）
                        portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                        return self._perform_save_operation()
                    except portalocker.LockException:
                        # 获取锁失败，等待后重试
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay + (attempt * 0.1))
                            continue
                        else:
                            raise Exception(f"无法在 {max_retries} 次尝试后获取文件锁")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"保存报告失败，尝试 {max_retries} 次后仍然失败: {str(e)}")
        
        raise Exception("保存操作异常结束")
    
    def _perform_save_operation(self) -> bool:
        """
        执行实际的保存操作
        
        在获取文件锁后执行的实际保存逻辑：
        1. 读取现有的主文件
        2. 保存新的结果文件
        3. 更新主文件索引
        4. 使用临时文件确保原子写入
        
        Returns:
            是否保存成功
        """
        # 读取现有的主文件
        try:
            with open(self.report_main_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    base = ReportMainFile(**json.loads(content))
                else:
                    raise ValueError("空文件")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            # 如果文件不存在或无效，创建新的
            base = ReportMainFile(
                abs_path=self.report_main_file,
                folder=os.path.join(self.report_path, self.benchmark_name),
                current_benchmarks=[],
                create_time=datetime.datetime.now().isoformat(),
                update_time=datetime.datetime.now().isoformat(),
                agent=self.agent,
                model=self.model,
            )
        
        # 确保结果文件夹存在
        if not os.path.exists(base.folder):
            os.makedirs(base.folder, exist_ok=True)
        
        # 生成结果文件名（带序号和时间戳）
        current_time = datetime.datetime.now().isoformat().replace(':', '_')
        name = f"benchmark-result-[{len(base.current_benchmarks)}]-{current_time}.json"
        data_file_path = os.path.join(base.folder, name)
        
        # 保存结果文件
        with open(data_file_path, "w", encoding="utf-8") as f:
            json.dump(self.report.model_dump(), f, ensure_ascii=False, indent=4)
        
        # 更新主文件
        base.current_benchmarks.append(name)
        base.update_time = datetime.datetime.now().isoformat()
        
        # 使用临时文件确保原子写入
        temp_main_file = self.report_main_file + ".tmp"
        with open(temp_main_file, "w", encoding="utf-8") as f:
            json.dump(base.model_dump(), f, ensure_ascii=False, indent=4)
        
        # 重命名（Windows 需要先删除目标文件）
        if os.name == 'nt':
            if os.path.exists(self.report_main_file):
                os.remove(self.report_main_file)
            os.rename(temp_main_file, self.report_main_file)
        else:
            os.rename(temp_main_file, self.report_main_file)
        
        return True
    
    def save(self) -> bool:
        """
        保存报告
        
        这是保存报告的主入口方法。
        
        Returns:
            是否保存成功
        """
        return self._safe_save_with_lock()
