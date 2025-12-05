"""
DataLoader - 数据加载器

提供基准测试数据的加载、保存和处理功能。

核心功能:
    1. 定义测试数据的数据模型（BenchmarkItem, ScoreCondition 等）
    2. 加载和保存测试数据集（JSON 格式）
    3. 处理引用替换（如 {1} 替换为步骤1的响应）
    4. 解析测试数据的 DSL 语法

数据结构层次:
    BenchmarkDataset (数据集)
        └── BenchmarkSequence (测试序列，如 color.json)
                └── BenchmarkItem (单个测试项)
                        └── ScoreCondition (评分条件)

引用语法:
    - {n}     : 引用步骤 n 的响应
    - {q:n}   : 引用步骤 n 的问题
    - {t:n}   : 引用步骤 n 的时间差（距今多久）
"""

from __future__ import annotations

import datetime
import json
import os
import re
from copy import deepcopy
from typing import Literal

from loguru import logger
from pydantic import BaseModel


class BinaryScoreItem(BaseModel):
    """
    二元评分项 - 用于加权二元评分
    
    支持将一个大的评分点分解为多个小的二元判断项，
    各自独立评分后按权重加总。
    
    示例:
        {
            "key": "名字正确",
            "weight": 0.4,
            "answer": "Alice"
        }
    
    Attributes:
        key: 评分项的标识/名称（如"名字正确"、"年龄正确"）
        weight: 该项的权重 (0.0 - 1.0)，所有项权重之和应为 1.0
        answer: 该项的评判标准/正确答案
        result: 评分结果 (True: 正确, False: 错误)
        reason: 评分理由
    """
    key: str
    weight: float
    answer: str
    result: bool = False
    reason: str = ""


class ScoreCondition(BaseModel):
    """
    评分条件 - 定义如何评估 Agent 的回答
    
    每个需要评分的步骤都有一个 ScoreCondition，指定：
    - 该题的满分是多少
    - 正确答案/评判标准是什么
    - 使用什么评估方式
    
    Attributes:
        score: 该题的满分值（如 0.1, 1.0）
        answer: 正确答案或评判标准
        is_multiple: 是否使用多分数评估（已弃用，推荐用 binary_items）
        is_lazy: 是否延迟评估（等待后续对话后再评分）
        lazy_count: 延迟评估的轮次数
        result: 实际得分（评估后填入）
        reason: 评分理由（评估后填入）
        lazy_eval_response_snapshot: 懒评估触发时的响应快照
        binary_items: 加权二元评分项列表（推荐的多项评分方式）
        eval_method: 实际使用的评估方式（评估后填入）
    """
    score: float                    # 满分值
    answer: str                     # 正确答案/评判标准
    is_multiple: bool = False       # 是否多分数评估（已弃用）
    is_lazy: bool = False           # 是否懒评估
    lazy_count: int = 0             # 懒评估轮次
    result: float = 0.0             # 实际得分
    reason: str = ""                # 评分理由
    lazy_eval_response_snapshot: str = ""  # 懒评估快照
    binary_items: list[BinaryScoreItem] | None = None  # 加权二元评分项
    eval_method: str = ""           # 评估方式: binary/score/weighted_binary/multi_score


class RefData(BaseModel):
    """
    引用数据 - 记录文本中的引用信息
    
    用于追踪测试数据中的引用关系，在执行时替换为实际值。
    
    引用类型:
        - answer: 引用某步骤的响应，语法: {n}
        - question: 引用某步骤的问题，语法: {q:n}
        - timedelta: 引用某步骤的时间差，语法: {t:n}
    
    Attributes:
        target: 引用的目标（步骤索引）
        type: 引用类型
    """
    target: str | int
    type: Literal["answer", "question", "timedelta"]


class PostProcess(BaseModel):
    """后处理配置"""
    prompt: str


class BenchmarkItem(BaseModel):
    """
    基准测试项 - 测试序列中的单个步骤
    
    这是测试数据的基本单元，表示一次对话交互。
    
    示例:
        {
            "index": 4,
            "ask": "What is my favorite color?",
            "score": {"score": 0.1, "answer": "red"},
            "depend": [1, 2, 3]
        }
    
    Attributes:
        index: 步骤索引（在序列内唯一）
        ask: 发送给 Agent 的消息
        score: 评分条件（如果需要评分）
        retry: 评分失败时的重试消息
        depend: 依赖的步骤索引列表
        refs: 引用数据列表（自动从 ask 中提取）
        post_process: 后处理要求（用 LLM 从响应中提取信息）
        time: 执行时间（执行时填入）
    """
    index: int                      # 步骤索引
    ask: str                        # 发送的消息
    score: ScoreCondition | None = None  # 评分条件
    retry: str | None = None        # 重试消息
    depend: list[int | str] = []    # 依赖项
    refs: list[RefData] = []        # 引用列表
    post_process: str | None = None # 后处理
    time: datetime.datetime | str | None = None  # 执行时间


class BenchmarkItemExtra(BenchmarkItem):
    """
    扩展的基准测试项 - 包含执行过程中的附加信息
    
    在执行器运行时，BenchmarkItem 会被扩展为 BenchmarkItemExtra，
    添加响应、评分结果等运行时数据。
    
    Attributes:
        response: Agent 的响应
        processed: 后处理的结果
        activate: 是否激活（评分通过或无需评分）
        executed: 是否已执行
        retry_history: 重试历史（如果有重试）
        retry_frozen_item: 重试前的冻结状态
    """
    response: str | None = None
    processed: str | None = None
    activate: bool = False
    executed: bool = False
    retry_history: "BenchmarkItemExtra | None" = None
    retry_frozen_item: "BenchmarkItemExtra | None" = None


# 解决自引用的前向引用问题
BenchmarkItemExtra.model_rebuild()


class BenchmarkSequence(BaseModel):
    """
    基准测试序列 - 一组相关的测试步骤
    
    每个序列对应一个 JSON 文件（如 color.json），
    包含一组相关的测试步骤（信息植入 + 提问验证）。
    
    Attributes:
        items: 测试项列表
    """
    items: list[BenchmarkItem]


class BenchmarkDatasetFile(BaseModel):
    """
    基准测试数据集文件 - 数据集配置
    
    对应配置文件（如 1k.json），定义：
    - 包含哪些测试序列
    - 开场提示
    - 记忆距离等参数
    
    Attributes:
        files: 序列名称到文件路径的映射
        head_prompts: 开场提示列表
        nonsense_list: 废话列表（可选）
        memory_distance: 记忆距离（tokens）
        memory_distance_level: 记忆距离计算级别
    """
    files: dict[str, str]
    head_prompts: list[str] = []
    nonsense_list: list[str]
    memory_distance: int
    memory_distance_level: Literal["total", "each_first", "each_all"] = "each_first"


class BenchmarkDataset(BenchmarkDatasetFile):
    """
    基准测试数据集 - 完整的数据集（包含加载后的数据）
    
    继承 BenchmarkDatasetFile，添加实际加载的序列数据。
    
    Attributes:
        data: 序列名称到序列对象的映射
    """
    data: dict[str, BenchmarkSequence]


# 引用类型的中文描述映射
ref_type_map = {
    "answer": "对前文的回答引用",
    "question": "对前文的问题引用",
    "timedelta": "对前文的距今时间差的引用",
}


def ref_change(
    data: list[BenchmarkItem | BenchmarkItemExtra | None],
    index_map: list[tuple[int, int]],
) -> list[BenchmarkItemExtra]:
    """
    更新引用索引
    
    在生成报告时，需要将序列内的局部索引映射到全局索引。
    例如：序列内的 {1} 可能需要变成全局的 {15}。
    
    Args:
        data: 数据列表
        index_map: 索引映射列表 [(旧索引, 新索引), ...]
        
    Returns:
        更新后的数据列表
    """
    result = deepcopy(data)
    queue = [x for x in result]
    
    while queue:
        item = queue.pop(0)
        if item:
            # 更新引用
            if item.refs:
                for ref in item.refs:
                    if isinstance(ref.target, str):
                        ref.target = int(ref.target)
                    for old_index, new_index in index_map:
                        if ref.target == old_index:
                            # 替换 ask 中的引用
                            item.ask = (
                                item.ask
                                .replace(rf"{{{old_index}}}", rf"{{{new_index}}}")
                                .replace(rf"\{{q:{old_index}\}}", rf"\{{q:{new_index}\}}")
                                .replace(rf"{{q:{old_index}}}", rf"{{q:{new_index}}}")
                                .replace(rf"\{{t:{old_index}\}}", rf"\{{t:{new_index}\}}")
                                .replace(rf"{{t:{old_index}}}", rf"{{t:{new_index}}}")
                            )
                            # 替换 answer 中的引用
                            if item.score:
                                item.score.answer = (
                                    item.score.answer
                                    .replace(rf"{{{old_index}}}", rf"{{{new_index}}}")
                                    .replace(rf"\{{q:{old_index}\}}", rf"\{{q:{new_index}\}}")
                                    .replace(rf"{{q:{old_index}}}", rf"{{q:{new_index}}}")
                                    .replace(rf"\{{t:{old_index}\}}", rf"\{{t:{new_index}\}}")
                                    .replace(rf"{{t:{old_index}}}", rf"{{t:{new_index}}}")
                                )
                            break
            
            # 递归处理重试历史
            if hasattr(item, 'retry_history') and item.retry_history:
                queue.append(item.retry_history)
            if hasattr(item, 'retry_frozen_item') and item.retry_frozen_item:
                queue.append(item.retry_frozen_item)
    
    return result


def format_data(
    text: str,
    benchmark_list: list[BenchmarkItem],
    answers: list[str] = None,
    _format: bool = True,
    extra_pack: str = "{value}",
) -> str:
    """
    格式化数据（用于预览，不执行实际替换）
    
    Args:
        text: 原始文本
        benchmark_list: 基准测试列表
        answers: 答案列表
        _format: 是否格式化
        extra_pack: 额外包装格式
        
    Returns:
        格式化后的文本
    """
    if not _format:
        return text
    
    # 构建索引到数据的映射
    benchmark_data: list[BenchmarkItem | None] = [None] * (
        max(map(lambda x: x.index, benchmark_list)) + 1
    )
    for benchmark in benchmark_list:
        benchmark_data[benchmark.index] = benchmark
    
    # 引用匹配正则：{n}, {t:n}, {q:n}
    ref_pattern = re.compile(
        r'\{(?P<answer>[0-9]+)}|\{(?P<timedelta>t:[0-9]+)}|\{(?P<question>q:[0-9]+)}'
    )
    text = f"{text}"
    
    if matches := ref_pattern.finditer(text):
        for match in matches:
            answer_ref = match.group('answer')
            timedelta_ref = match.group('timedelta')
            question_ref = match.group('question')
            
            if answer_ref:
                # 答案引用
                index = int(answer_ref)
                if answers:
                    text = text.replace(
                        match.group(0),
                        extra_pack.format(value=f"{{{answers[index]}}}")
                    )
                else:
                    text = text.replace(match.group(0), f"{{{answer_ref}}}")
            elif timedelta_ref:
                # 时间差引用
                index = int(timedelta_ref[2:])
                if benchmark_data[index].time:
                    if isinstance(benchmark_data[index].time, str):
                        delta = datetime.datetime.now() - datetime.datetime.fromisoformat(
                            benchmark_data[index].time
                        )
                    else:
                        delta = datetime.datetime.now() - benchmark_data[index].time
                    delta_seconds = delta.total_seconds()
                    text = text.replace(
                        match.group(0),
                        extra_pack.format(
                            value=f"{int(delta_seconds / 3600)}小时{int((delta_seconds % 3600) / 60)}分钟"
                        )
                    )
            elif question_ref:
                # 问题引用
                index = int(question_ref[2:])
                text = text.replace(
                    match.group(0),
                    extra_pack.format(value=f"{benchmark_data[index].ask}")
                )
    
    return text


def format_text(
    text: str,
    benchmark_list: list[BenchmarkItemExtra],
    extra: list[BenchmarkItemExtra] = None,
    extra_pack: str = "{value}",
    return_highlight_position: bool = False,
):
    """
    格式化文本（执行实际的引用替换）
    
    这是 Actuator 执行时使用的核心函数，将引用替换为实际值。
    
    替换规则:
        - {n} -> 步骤 n 的响应（或 processed 结果）
        - {q:n} -> 步骤 n 的问题
        - {t:n} -> 步骤 n 的时间差（如"2小时30分钟"）
    
    Args:
        text: 原始文本（包含引用）
        benchmark_list: 基准测试列表（包含执行结果）
        extra: 额外项列表
        extra_pack: 包装格式
        return_highlight_position: 是否返回高亮位置
        
    Returns:
        格式化后的文本（可选返回高亮位置）
    """
    # 合并额外项
    if extra and len(extra) > 0:
        for item in extra:
            while item.index >= len(benchmark_list):
                benchmark_list.append(None)
            benchmark_list[item.index] = item
    
    # 引用匹配正则
    ref_pattern = re.compile(
        r'\{(?P<answer>[0-9]+)}|\{(?P<timedelta>t:[0-9]+)}|\{(?P<question>q:[0-9]+)}'
    )
    text = f"{text}"
    highlight_position = []
    
    if matches := ref_pattern.finditer(text):
        for match in matches:
            answer_ref = match.group('answer')
            timedelta_ref = match.group('timedelta')
            question_ref = match.group('question')
            
            if answer_ref:
                # 答案引用：使用 processed（如果有）或 response
                index = int(answer_ref)
                if benchmark_list[index] and benchmark_list[index].response:
                    value = (
                        benchmark_list[index].processed 
                        if benchmark_list[index].processed 
                        else benchmark_list[index].response
                    )
                else:
                    value = ""
                target = extra_pack.format(value=value)
                text = text.replace(match.group(0), target)
                highlight_position.append((match.start(), match.start() + len(target), 'Answer Reference'))
            elif timedelta_ref:
                # 时间差引用
                index = int(timedelta_ref[2:])
                if benchmark_list[index].time:
                    if isinstance(benchmark_list[index].time, str):
                        delta = datetime.datetime.now() - datetime.datetime.fromisoformat(
                            benchmark_list[index].time
                        )
                    else:
                        delta = datetime.datetime.now() - benchmark_list[index].time
                    delta_seconds = delta.total_seconds()
                    target = extra_pack.format(
                        value=f"{int(delta_seconds / 3600)}小时{int((delta_seconds % 3600) / 60)}分钟"
                    )
                    text = text.replace(match.group(0), target)
                    highlight_position.append((match.start(), match.start() + len(target), 'Time Delta Reference'))
            elif question_ref:
                # 问题引用
                index = int(question_ref[2:])
                target = extra_pack.format(value=f"{benchmark_list[index].ask}")
                text = text.replace(match.group(0), target)
                highlight_position.append((match.start(), match.start() + len(target), 'Question Reference'))
    
    if return_highlight_position:
        return text, highlight_position
    else:
        return text


def sequence_to_json(sequence: BenchmarkSequence, file_path: str) -> None:
    """
    将测试序列保存为 JSON 文件
    
    Args:
        sequence: 测试序列
        file_path: 输出文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sequence.model_dump(), f, indent=4, ensure_ascii=False)


def load_sequence_from_json(file_path: str) -> BenchmarkSequence:
    """
    从 JSON 文件加载测试序列
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        测试序列对象
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return BenchmarkSequence(**data)


def load_dataset_files(files: dict[str, str], base_dir: str = None) -> dict[str, BenchmarkSequence]:
    """
    批量加载测试序列文件
    
    Args:
        files: 序列名称到文件路径的映射
        base_dir: 基础目录（用于解析相对路径）
        
    Returns:
        序列名称到序列对象的映射
    """
    dataset = {}
    for name, file in files.items():
        try:
            # 解析相对路径
            if base_dir and not os.path.isabs(file):
                file = os.path.normpath(os.path.join(base_dir, file))
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sequence = BenchmarkSequence(**data)
            dataset[name] = sequence
        except Exception as e:
            logger.error(e)
            logger.error(f"Error when load sequence from name:{file}")
    return dataset


def load_dataset(config_file: str) -> BenchmarkDataset:
    """
    加载完整的测试数据集
    
    从配置文件（如 1k.json）加载数据集，包括：
    - 解析配置参数
    - 加载所有引用的测试序列
    
    Args:
        config_file: 配置文件路径（如 data/config/1k.json）
        
    Returns:
        完整的数据集对象
    """
    # 获取配置文件所在目录，用于解析相对路径
    config_dir = os.path.dirname(os.path.abspath(config_file))
    
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解析配置文件结构
    dataset_file = BenchmarkDatasetFile(**data)
    dataset = BenchmarkDataset(
        data={},
        files=dataset_file.files,
        memory_distance=dataset_file.memory_distance,
        nonsense_list=dataset_file.nonsense_list,
        head_prompts=dataset_file.head_prompts,
    )
    # 加载所有测试序列
    dataset.data = load_dataset_files(dataset_file.files, config_dir)
    return dataset


def save_dataset(dataset: BenchmarkDataset, file_path: str) -> None:
    """
    保存数据集
    
    Args:
        dataset: 数据集对象
        file_path: 输出文件路径
    """
    # 保存各个序列文件
    for key, value in dataset.data.items():
        sequence_to_json(value, dataset.files[key])
    
    # 保存配置文件
    dataset_file = BenchmarkDatasetFile(**dataset.model_dump())
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_file.model_dump(), f, indent=4, ensure_ascii=False)


def parse_content(content: str) -> list[BenchmarkItem]:
    """
    解析 DSL 格式的测试内容
    
    支持的 DSL 语法:
        [索引] {问题内容} S:{分数,答案} D:{依赖} R:{重试} P:{后处理}
        
    评分类型:
        - S: 普通评分
        - MS: 多分数评分
        - LS(n): 懒评估（n轮后评估）
    
    示例:
        [1] {My favorite color is red.}
        [2] {What is my favorite color?} S:{0.1,red} D:{1}
    
    Args:
        content: DSL 格式的内容
        
    Returns:
        解析后的测试项列表
    """
    # 主模式：匹配完整的测试项
    pattern = re.compile(
        r'\[(?P<index>\d+)]\s*\{(?P<content>(?:\\.|[^\\{}])*)}\s*'
        r'(?P<mark>(((?P<marktype>S:|MS:|LS\((?P<lazycount>[0-9]+)\):)'
        r'\{(?P<score>[0-9.]+),(?P<answer>(?:\\.|[^\\{,}])*)})|.{0}))\s*'
        r'((D:\{(?P<depend>[0-9,]+)})|.{0})\s*'
        r'((R:\{(?P<retry>(?:\\.|[^\\{}])*)})|.{0})\s*'
        r'((P:\{(?P<process>(?:\\.|[^\\{}])*)})|.{0})'
    )
    
    # 引用模式：匹配引用语法
    ref_pattern = re.compile(
        r'\{(?P<answer>[0-9]+)}|\{(?P<timedelta>t:[0-9]+)}|\{(?P<question>q:[0-9]+)}'
    )
    
    matches = pattern.finditer(content)
    parsed_items: list[BenchmarkItem] = []
    
    for match in matches:
        index = int(match.group('index'))
        content_text = match.group('content').replace('\\', '')
        score = match.group('score')
        answer = match.group('answer').replace('\\', '') if match.group('answer') else None
        depend = match.group('depend')
        retry = match.group('retry')
        process = match.group('process').replace('\\', '') if match.group('process') else None
        
        # 解析评分类型
        if score is not None:
            score = float(score)
            is_multiple = match.group('marktype') == "MS:"
            is_lazy = (
                match.group('marktype').startswith("LS") 
                and match.group('lazycount') is not None
            )
            lazy_count = match.group('lazycount') if match.group('lazycount') else 0
        else:
            score = None
            is_multiple = False
            is_lazy = False
            lazy_count = 0
        
        # 解析依赖
        depend_list = [
            int(d) if d.isdigit() else d 
            for d in depend.split(',')
        ] if depend else []
        
        # 自动提取引用
        refs = []
        if content_text and (ref_matches := ref_pattern.finditer(content_text)):
            _auto_refs(ref_matches, refs)
        if answer and (ref_matches := ref_pattern.finditer(answer)):
            _auto_refs(ref_matches, refs)
        if retry and (ref_matches := ref_pattern.finditer(retry)):
            _auto_refs(ref_matches, refs)
        if process and (ref_matches := ref_pattern.finditer(process)):
            _auto_refs(ref_matches, refs)
        
        # 构建测试项
        parsed_items.append(BenchmarkItem(
            index=index,
            ask=content_text,
            score=ScoreCondition(
                score=score,
                answer=answer,
                is_multiple=is_multiple,
                is_lazy=is_lazy,
                lazy_count=int(lazy_count),
            ) if score else None,
            retry=retry,
            depend=list(set(depend_list + [ref.target for ref in refs])),
            refs=refs,
            post_process=process,
        ))
    
    return parsed_items


def _auto_refs(ref_matches, refs: list[RefData]) -> None:
    """
    自动从正则匹配中提取引用
    
    Args:
        ref_matches: 正则匹配迭代器
        refs: 引用列表（会被修改）
    """
    for ref_match in ref_matches:
        answer_ref = ref_match.group('answer')
        timedelta_ref = ref_match.group('timedelta')
        question_ref = ref_match.group('question')
        
        if answer_ref:
            refs.append(RefData(target=answer_ref, type="answer"))
        elif timedelta_ref:
            refs.append(RefData(target=int(timedelta_ref[2:]), type="timedelta"))
        elif question_ref:
            refs.append(RefData(target=int(question_ref[2:]), type="question"))


def benchmark_def_to_logseq_markdown(
    data: list[BenchmarkItem],
    enable_format: bool = False,
) -> str:
    """
    将测试定义转换为 Logseq Markdown 格式
    
    用于可视化查看测试定义。
    
    Args:
        data: 测试项列表
        enable_format: 是否格式化引用
        
    Returns:
        Markdown 格式的文本
    """
    content = ""
    for item in data:
        content += f"- Index: {item.index}\n"
        sub_content = ""
        sub_content += (
            f"- Content: \n\t- " 
            + format_data(item.ask.replace('\\', ''), data, _format=enable_format).replace('\n', '\n\t') 
            + "\n"
        )
        
        if item.score:
            sub_content += (
                f"- Score: \n" 
                if not item.score.is_multiple 
                else f"Score[细粒度化判定]: \n"
            )
            sub_content += (
                f"\t- 可得分: {item.score.score}\n" 
                if not item.score.is_multiple 
                else f"\t可得分[每个子得分点]: {item.score.score}\n"
            )
            sub_content += (
                f"\t- 比对答案: ```" 
                + format_data(item.score.answer.replace('\\', ''), data, _format=enable_format) 
                + "```\n"
            )
        
        if item.depend:
            sub_content += f"- Depend: \n"
            for dep in item.depend:
                sub_content += f"\t- {dep}\n"
        
        if item.refs:
            sub_content += f"- References: \n"
            for ref in item.refs:
                sub_content += f"\t- {ref.target}({ref_type_map[ref.type]})\n"
        
        if item.post_process:
            sub_content += f"- Post Process: \n"
            sub_content += (
                '\t- ' 
                + format_data(item.post_process.replace('\\', ''), data, _format=enable_format).replace('\n', '\n\t') 
                + "\n"
            )
        
        content += '\t' + sub_content.replace('\n', '\n\t') + "\n"
    
    return content


def benchmark_result_to_logseq_markdown(
    data: list[BenchmarkItemExtra],
    time_str: str,
    enable_format: bool = True,
) -> str:
    """
    将测试结果转换为 Logseq Markdown 格式
    
    用于可视化查看测试结果。
    
    Args:
        data: 测试结果列表
        time_str: 时间字符串
        enable_format: 是否格式化引用
        
    Returns:
        Markdown 格式的文本
    """
    score = 0.0
    pscore = 0.0
    total_score = 0.0
    
    # 计算总分
    try:
        visited = []
        for item in data:
            if item:
                if item.score and not item.retry_history:
                    score += item.score.result
                elif item.retry_history:
                    score += item.retry_history.score.result
                if item.score and item.index not in visited:
                    total_score += item.score.score
                    visited.append(item.index)
        pscore = score / total_score
        logger.info(f"Score: {pscore * 100:.2f}% ({score}/{total_score})")
    except Exception:
        pass
    
    data = list(filter(lambda x: x is not None, data))
    benchmark_data: list[BenchmarkItemExtra | None] = [None] * (
        max(map(lambda x: x.index, data)) + 1
    )
    for benchmark in data:
        benchmark_data[benchmark.index] = benchmark
    
    content = f"## mark: {pscore * 100:.2f}% ({score:.4f}/{total_score:.4f}) {time_str}\n\n"
    data = deepcopy(data)
    
    while data:
        raw_item = data.pop(0)
        if raw_item.retry_history:
            raw_item.retry_history.retry = "当前为上一轮的重试结果"
            data.insert(0, raw_item.retry_history)
        
        item = raw_item if not raw_item.retry_frozen_item else raw_item.retry_frozen_item
        content += f"- Index: {item.index}\n"
        sub_content = "" if item.retry != "当前为上一轮的重试结果" else f"\t- {item.retry}: \n"
        sub_content += f"- 执行: {item.executed}\n"
        sub_content += f"- 激活: {item.activate}\n"
        sub_content += (
            f"- Content: \n\t- " 
            + format_text((item.ask if item.ask else '').replace('\\', ''), benchmark_data).replace('\n', '\n\t').replace('```', '`') 
            + "\n"
        )
        
        if item.response:
            sub_content += (
                f"- Response: \n\t- " 
                + item.response.replace('\n', '\n\t').replace('```', '`') 
                + (f"({item.processed})" if item.processed else '') 
                + "\n"
            )
        
        if item.retry:
            sub_content += (
                f"- Retry: \n\t- " 
                + item.retry.replace('\n', '\n\t').replace('```', '`') 
                + "\n"
            )
        
        if item.score:
            sub_content += (
                f"- Score: \n" 
                if not item.score.is_multiple 
                else f"Score[细粒度化判定]: \n"
            )
            sub_content += (
                f"\t- 可得分: {item.score.score}\n" 
                if not item.score.is_multiple 
                else f"\t可得分[每个子得分点]: {item.score.score}\n"
            )
            sub_content += f"\t- 得分: {item.score.result}/{item.score.score}\n"
            sub_content += f"\t- Reason: {item.score.reason}\n"
            sub_content += (
                f"\t- 比对答案: ```" 
                + format_text(item.score.answer.replace('\\', ''), benchmark_data) 
                + "```\n"
            )
        
        if item.depend:
            sub_content += f"- Depend: \n"
            for dep in item.depend:
                sub_content += f"\t- {dep}\n"
        
        if item.refs:
            sub_content += f"- References: \n"
            for ref in item.refs:
                sub_content += f"\t- {ref.target}({ref_type_map[ref.type]})\n"
        
        if item.post_process:
            sub_content += f"- Post Process: \n"
            sub_content += (
                '\t- ' 
                + format_text(item.post_process.replace('\\', ''), benchmark_data).replace('\n', '\n\t') 
                + "\n"
            )
        
        content += '\t' + sub_content.replace('\n', '\n\t') + "\n"
    
    return content
