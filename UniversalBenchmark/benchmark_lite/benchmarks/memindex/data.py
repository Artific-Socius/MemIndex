"""MemIndex 数据加载。

从 MemIndex 的 JSON 格式配置和数据文件中加载测试数据集。
不依赖原始 MemIndex 代码，完全自包含。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ── 数据模型 ───────────────────────────────────────────────────


@dataclass
class BinaryItem:
    """加权二元评分中的单个子项。"""

    key: str
    weight: float
    answer: str
    result: bool = False
    reason: str = ""


@dataclass
class ItemScore:
    """评分条件：满分值 + 正确答案 + 评估方式配置。"""

    score: float
    answer: str
    is_multiple: bool = False
    is_lazy: bool = False
    lazy_count: int = 0
    binary_items: list[BinaryItem] = field(default_factory=list)
    result: float = 0.0
    reason: str = ""
    eval_method: str = ""


@dataclass
class SequenceItem:
    """单个测试步骤。

    - 没有 ``score`` 的步骤是信息注入 (CONVERSATION)
    - 有 ``score`` 的步骤是评估点 (EVALUATION)
    """

    index: int
    ask: str
    score: Optional[ItemScore] = None
    retry: Optional[str] = None
    depend: list[int] = field(default_factory=list)
    post_process: Optional[str] = None
    response: Optional[str] = None
    activate: bool = True
    executed: bool = False


@dataclass
class Sequence:
    """一组测试步骤，对应 MemIndex 中的一个 JSON 文件（如 color.json）。"""

    items: list[SequenceItem]


@dataclass
class DatasetConfig:
    """完整的测试数据集：配置参数 + 加载后的序列数据。"""

    files: dict[str, str]
    head_prompts: list[str]
    nonsense_list: list[str]
    memory_distance: int
    memory_distance_level: str = "each_first"
    data: dict[str, Sequence] = field(default_factory=dict)


# ── 加载函数 ───────────────────────────────────────────────────


def load_dataset(config_path: str) -> DatasetConfig:
    """从 MemIndex 配置文件（如 ``1k.json``）加载完整数据集。"""
    config_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = DatasetConfig(
        files=raw.get("files", {}),
        head_prompts=raw.get("head_prompts", []),
        nonsense_list=raw.get("nonsense_list", []),
        memory_distance=raw.get("memory_distance", 2048),
        memory_distance_level=raw.get("memory_distance_level", "each_first"),
    )

    for name, file_path in dataset.files.items():
        if not os.path.isabs(file_path):
            file_path = os.path.normpath(
                os.path.join(config_dir, file_path)
            )
        dataset.data[name] = _load_sequence(file_path)

    return dataset


def _load_sequence(file_path: str) -> Sequence:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: list[SequenceItem] = []
    for item_raw in raw.get("items", []):
        score = None
        score_raw = item_raw.get("score")
        if score_raw:
            binary_items = [
                BinaryItem(
                    key=bi["key"],
                    weight=bi["weight"],
                    answer=bi["answer"],
                )
                for bi in (score_raw.get("binary_items") or [])
            ]
            score = ItemScore(
                score=score_raw["score"],
                answer=score_raw.get("answer", ""),
                is_multiple=score_raw.get("is_multiple", False),
                is_lazy=score_raw.get("is_lazy", False),
                lazy_count=score_raw.get("lazy_count", 0),
                binary_items=binary_items,
            )

        items.append(SequenceItem(
            index=item_raw["index"],
            ask=item_raw["ask"],
            score=score,
            retry=item_raw.get("retry"),
            depend=[int(d) for d in item_raw.get("depend", [])],
            post_process=item_raw.get("post_process"),
        ))

    return Sequence(items=items)


# ── 引用替换 ───────────────────────────────────────────────────

_REF_PATTERN = re.compile(
    r"\{(?P<answer>[0-9]+)\}"
    r"|\{(?P<question>q:[0-9]+)\}"
    r"|\{(?P<timedelta>t:[0-9]+)\}"
)


def resolve_refs(
    text: str,
    intermediate: dict[int, SequenceItem],
) -> str:
    """将文本中的引用替换为实际值。

    支持:
        - ``{n}``   → 步骤 n 的 response
        - ``{q:n}`` → 步骤 n 的 ask
    """
    def _replacer(match: re.Match[str]) -> str:
        answer_ref = match.group("answer")
        question_ref = match.group("question")

        if answer_ref:
            idx = int(answer_ref)
            item = intermediate.get(idx)
            if item and item.response:
                return str(item.response)
        elif question_ref:
            idx = int(question_ref[2:])
            item = intermediate.get(idx)
            if item:
                return str(item.ask)
        return match.group(0)

    return _REF_PATTERN.sub(_replacer, text)
