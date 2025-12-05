"""
DatasetCompiler - 数据集编译器

用于将 Markdown/DSL 格式的测试用例编译为 JSON 数据集。

核心功能:
    1. 解析 DSL 格式的测试用例文件
    2. 编译生成 JSON 格式的测试序列
    3. 生成数据集配置文件

DSL 语法示例:
    [1] {My favorite color is red.}
    [2] {What is my favorite color?} S:{0.1,red}
    
    其中:
    - [n] 是步骤索引
    - {内容} 是发送给 Agent 的消息
    - S:{分数,答案} 是评分条件

使用场景:
    当你需要创建新的测试用例时，可以用 Markdown/DSL 格式
    编写测试内容，然后使用此编译器转换为 JSON 格式。

使用方式:
    from utils.dataset_compiler import DatasetCompiler
    
    compiler = DatasetCompiler(
        files={"color": "./data/processed/color.md"},
        target={"color": "./data/json/color.json"},
        memory_distance=1024
    )
    dataset, total_score = compiler.compile("data/config/1k.json")
"""

from __future__ import annotations

from loguru import logger

from .data_loader import (
    parse_content,
    BenchmarkSequence,
    BenchmarkDataset,
    save_dataset,
)


class DatasetCompiler:
    """
    数据集编译器
    
    将 Markdown/DSL 格式的测试用例编译为 JSON 数据集。
    
    工作流程:
        1. 读取源文件（Markdown/DSL 格式）
        2. 解析 DSL 语法，提取测试项
        3. 生成 BenchmarkSequence 对象
        4. 保存为 JSON 文件
        5. 生成数据集配置文件
    
    Attributes:
        files: 源文件映射 {序列名称: 源文件路径}
        target: 目标文件映射 {序列名称: 输出 JSON 路径}
        head_prompts: 开场提示列表
        memory_distance: 记忆距离（tokens）
        memory_distance_level: 记忆距离计算级别
    """
    
    def __init__(
        self,
        files: dict[str, str],
        target: dict[str, str],
        head_prompts: list[str] = None,
        memory_distance: int = 1024,
        memory_distance_level: str = "each_first",
    ):
        """
        初始化编译器
        
        Args:
            files: 源文件映射 {序列名称: 源文件路径}
                例如: {"color": "./data/processed/color.md"}
            target: 目标文件映射 {序列名称: 输出JSON路径}
                例如: {"color": "./data/json/color.json"}
            head_prompts: 开场提示列表
                发送给 Agent 的首条消息，说明测试规则
            memory_distance: 记忆距离（tokens）
                在测试项之间插入的废话 Token 数量
            memory_distance_level: 记忆距离计算级别
                - "total": 总记忆距离
                - "each_first": 每个序列首项的记忆距离
                - "each_all": 每个序列所有项的记忆距离
        """
        self.files = files
        self.target = target
        self.head_prompts = head_prompts or [
            "I'm going to give you a long-term memory benchmark test. "
            "As I go along, I'll provide different kinds of information, "
            "and you'll be expected to give very brief responses, "
            "providing only the necessary responses. "
            "Otherwise, just a brief acknowledgment. Understood?"
        ]
        self.memory_distance = memory_distance
        self.memory_distance_level = memory_distance_level
    
    def compile(self, output_path: str) -> tuple[BenchmarkDataset, float]:
        """
        编译数据集
        
        读取所有源文件，解析 DSL 语法，生成 JSON 数据集。
        
        编译过程:
            1. 遍历所有源文件
            2. 读取文件内容
            3. 使用 parse_content 解析 DSL
            4. 创建 BenchmarkSequence
            5. 统计总分
            6. 保存数据集
        
        Args:
            output_path: 输出配置文件路径
                例如: "data/config/1k.json"
            
        Returns:
            (数据集对象, 总分)
        """
        sequences = {}
        total_score = 0
        
        # 遍历所有源文件
        for key, file in self.files.items():
            with open(file, 'r', encoding='utf-8') as f:
                data = f.read()
            
            # 解析 DSL 语法
            items = parse_content(data)
            sequence = BenchmarkSequence(items=items)
            sequences[key] = sequence
            
            # 统计该序列的分数
            score = 0
            for item in items:
                if item.score:
                    total_score += item.score.score
                    score += item.score.score
            score = round(score, 6)
            logger.info(f"{key}: {score}")
        
        total_score = round(total_score, 6)
        
        # 创建数据集
        dataset = BenchmarkDataset(
            data=sequences,
            files=self.target,
            nonsense_list=[],
            memory_distance=self.memory_distance,
            memory_distance_level=self.memory_distance_level,
            head_prompts=self.head_prompts,
        )
        
        # 保存数据集
        save_dataset(dataset, output_path)
        logger.info(f"Max Score: {total_score}")
        
        return dataset, total_score
    
    @staticmethod
    def compile_from_config(
        files: dict[str, str],
        target: dict[str, str],
        output_path: str,
        head_prompts: list[str] = None,
        memory_distance: int = 1024,
        memory_distance_level: str = "each_first",
    ) -> tuple[BenchmarkDataset, float]:
        """
        从配置编译数据集（便捷静态方法）
        
        一步完成编译器创建和编译操作。
        
        Args:
            files: 源文件映射
            target: 目标文件映射
            output_path: 输出配置文件路径
            head_prompts: 开场提示列表
            memory_distance: 记忆距离
            memory_distance_level: 记忆距离级别
            
        Returns:
            (数据集对象, 总分)
        """
        compiler = DatasetCompiler(
            files=files,
            target=target,
            head_prompts=head_prompts,
            memory_distance=memory_distance,
            memory_distance_level=memory_distance_level,
        )
        return compiler.compile(output_path)


def main():
    """
    示例用法
    
    展示如何使用 DatasetCompiler 编译测试数据集。
    """
    # 源文件映射：Markdown/DSL 格式的测试用例
    files = {
        "color": "./data/processed/color.md",
        "joke": "./data/processed/joke.md",
        "location": "./data/processed/location.md",
        "namelist": "./data/processed/namelist.md",
        "attachment": "./data/processed/attachment.md",
        "restaurant": "./data/processed/restaurant.md",
        "tv": "./data/processed/tv.md",
        "shopping": "./data/processed/shopping.md",
        "spy": "./data/processed/spy.md",
        "sweet": "./data/processed/sweet.md",
    }
    
    # 目标文件映射：输出的 JSON 文件
    target = {
        "color": "./data/json/color.json",
        "joke": "./data/json/joke.json",
        "location": "./data/json/location.json",
        "namelist": "./data/json/namelist.json",
        "attachment": "./data/json/attachment.json",
        "restaurant": "./data/json/restaurant.json",
        "tv": "./data/json/tv.json",
        "shopping": "./data/json/shopping.json",
        "spy": "./data/json/spy.json",
        "sweet": "./data/json/sweet.json",
    }
    
    # 编译数据集
    DatasetCompiler.compile_from_config(
        files=files,
        target=target,
        output_path="data/config/1k.json",
    )


if __name__ == "__main__":
    main()
