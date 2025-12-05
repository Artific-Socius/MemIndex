"""
NonsenseGenerator - 废话生成器

生成用于填充对话的废话内容，用于测试记忆系统的长期记忆能力。

核心功能:
    1. 生成指定 Token 数量的填充对话
    2. 用于模拟"记忆距离"（memory_distance）

工作原理:
    在正式的测试内容之间插入无关对话（废话），
    以测试 Agent 在大量无关信息干扰下的记忆能力。
    
    例如: 
        - 用户说: "我最喜欢的颜色是红色"
        - [插入 1024 tokens 的废话]  <- 这就是记忆距离
        - 用户问: "我最喜欢的颜色是什么？"

废话内容:
    使用问答题库（trivia questions）作为废话内容，
    这些问答对 Agent 来说是无关的，但格式上是有意义的对话。

代码来源: 
    https://github.com/GoodAI/goodai-ltm-benchmark/blob/main/utils/filling_task.py
"""

from __future__ import annotations

import json
import os
import random
from typing import Callable

# 问答数据缓存（避免重复加载）
TRIVIA_CACHE = None


def get_trivia(data_path: str = "data/nonsense.json") -> list[dict]:
    """
    获取问答数据
    
    从 JSON 文件加载问答数据，使用缓存避免重复读取。
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        问答数据列表，每项包含 "Question" 和 "AnswerValue"
    """
    global TRIVIA_CACHE
    if TRIVIA_CACHE is None:
        with open(data_path, "r", encoding="utf8") as file:
            TRIVIA_CACHE = json.load(file)["Data"]
    return TRIVIA_CACHE


def filler_no_response_tokens_trivia(
    num_tokens: int,
    max_message_size: int,
    token_len_function: Callable[[str], int],
    data_path: str = "data/nonsense.json",
) -> tuple[str, str]:
    """
    生成指定 Token 数量的废话
    
    随机选取问答题目，拼接成达到指定 Token 数量的消息。
    消息格式设计为让 Agent 处理这些问答，以保持对话的合理性。
    
    生成的消息示例:
        "Here are some trivia questions and answers for you to process.
         Please extract all of the answers in json form as a single message:
         E.g ["answer 1", "answer 2", ...]
         Q: What is the capital of France?, A: Paris
         Q: What color is the sky?, A: Blue
         ..."
    
    Args:
        num_tokens: 目标 Token 数量
        max_message_size: 最大消息大小（字符数）
        token_len_function: 计算 Token 长度的函数
        data_path: 问答数据文件路径
        
    Returns:
        (废话消息, 预期答案的 JSON 字符串)
    """
    data = get_trivia(data_path)
    
    # 提示语：要求 Agent 处理这些问答
    message = (
        "Here are some trivia questions and answers for you to process. "
        'Please extract all of the answers in json form as a single message: '
        'E.g ["answer 1", "answer 2", ...]\n'
    )
    
    tokens_to_return = min(num_tokens, max_message_size)
    total_tokens = token_len_function(message)
    messages = [message]
    answers = []
    at_least_one_trivia = False
    est_response_tokens = 0
    
    # 持续添加问答直到达到目标 Token 数
    while not at_least_one_trivia or (total_tokens + est_response_tokens) < tokens_to_return:
        trivia = random.choice(data)
        trivia_msg = f"Q: {trivia['Question']}, A: {trivia['AnswerValue']}\n"
        answers.append(trivia['AnswerValue'])
        total_tokens += token_len_function(trivia_msg)
        est_response_tokens = token_len_function(str(answers))
        messages.append(trivia_msg)
        at_least_one_trivia = True
    
    return "".join(messages), str(answers)


class NonsenseGenerator:
    """
    废话生成器类
    
    封装废话生成功能，提供更方便的接口。
    
    使用方式:
        generator = NonsenseGenerator(
            data_path="data/nonsense.json",
            token_len_function=lambda x: len(x.split())
        )
        message, answer = generator.generate(num_tokens=1024)
    
    Attributes:
        data_path: 问答数据文件路径
        token_len_function: Token 长度计算函数
    """
    
    def __init__(
        self,
        data_path: str = "data/nonsense.json",
        token_len_function: Callable[[str], int] = None,
    ):
        """
        初始化废话生成器
        
        Args:
            data_path: 问答数据文件路径
            token_len_function: 计算 Token 长度的函数
                如果未提供，默认使用按空格分词的近似计算
        """
        self.data_path = data_path
        self.token_len_function = token_len_function or (lambda x: len(x.split()))
        self._data = None
    
    @property
    def data(self) -> list[dict]:
        """
        获取问答数据（懒加载）
        
        Returns:
            问答数据列表
        """
        if self._data is None:
            self._data = get_trivia(self.data_path)
        return self._data
    
    def generate(
        self,
        num_tokens: int,
        max_message_size: int = 10240,
    ) -> tuple[str, str]:
        """
        生成指定 Token 数量的废话
        
        Args:
            num_tokens: 目标 Token 数量
            max_message_size: 最大消息大小（字符数），默认 10KB
            
        Returns:
            (废话消息, 预期答案的 JSON 字符串)
        """
        return filler_no_response_tokens_trivia(
            num_tokens=num_tokens,
            max_message_size=max_message_size,
            token_len_function=self.token_len_function,
            data_path=self.data_path,
        )
