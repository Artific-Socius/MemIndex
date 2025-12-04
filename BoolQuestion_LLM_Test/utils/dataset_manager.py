"""
BoolQ数据集管理器
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from datasets import load_dataset, Dataset
from loguru import logger

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models_utils import BoolQItem
from i18n import t


@dataclass
class DirtyDataStats:
    """脏数据加载统计"""
    total_files: int = 0
    valid_files: int = 0
    skipped_files: int = 0
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    file_details: list[dict] = field(default_factory=list)


class DatasetManager:
    """
    BoolQ数据集管理器
    
    功能:
    - 加载BoolQ数据集
    - 支持数据过滤 (排除脏数据)
    - 支持数据限制 (用于调试)
    - 提供迭代器接口
    """
    
    # 脏数据文件必须包含的字段
    REQUIRED_DIRTY_DATA_FIELDS = {"hash"}
    # 可选但推荐的字段
    OPTIONAL_DIRTY_DATA_FIELDS = {"index", "question", "passage", "original_answer", "audit_result"}
    
    def __init__(
        self,
        split: str = "validation",
        limit: int = 0,
        dirty_data_path: Optional[str] = None,
    ):
        """
        初始化数据集管理器
        
        Args:
            split: 数据集分割 ("train", "validation", 或 "all" 同时加载两者)
            limit: 限制数据条数 (0表示不限制)
            dirty_data_path: 脏数据文件夹或文件路径 (JSONL格式)
        """
        self.split = split
        self.limit = limit
        self.dirty_data_path = dirty_data_path
        
        self._dataset: Optional[Dataset] = None
        self._dirty_hashes: set[str] = set()
        self._dirty_stats: Optional[DirtyDataStats] = None
        self._loaded = False
    
    def load(self) -> DatasetManager:
        """
        加载数据集
        
        Returns:
            self: 支持链式调用
        """
        if self._loaded:
            return self
        
        logger.info(t("正在加载 BoolQ 数据集 (split: {split})...", split=self.split))
        
        # 加载数据集
        if self.split == "all":
            # 加载 train 和 validation 并合并
            from datasets import concatenate_datasets
            train_dataset = load_dataset("google/boolq", split="train")
            val_dataset = load_dataset("google/boolq", split="validation")
            logger.info(t("Train: {train_count} 条, Validation: {val_count} 条", 
                        train_count=len(train_dataset), val_count=len(val_dataset)))
            self._dataset = concatenate_datasets([train_dataset, val_dataset])
        else:
            self._dataset = load_dataset("google/boolq", split=self.split)
        
        total_rows = len(self._dataset)
        logger.info(t("数据集加载完成, 共 {count} 条数据", count=total_rows))
        
        # 应用限制
        if self.limit > 0:
            actual_limit = min(self.limit, total_rows)
            if actual_limit < self.limit:
                logger.warning(t("请求限制 {limit} 超过数据集大小 {total}, 使用全部数据", limit=self.limit, total=total_rows))
            else:
                logger.info(t("限制数据条数为 {limit}", limit=actual_limit))
            self._dataset = self._dataset.select(range(actual_limit))
        
        # 加载脏数据哈希
        if self.dirty_data_path:
            self._load_dirty_hashes()
        
        self._loaded = True
        return self
    
    def _load_dirty_hashes(self) -> None:
        """
        加载脏数据哈希列表
        
        支持两种模式:
        - 文件夹模式: 加载文件夹中所有.jsonl文件
        - 文件模式: 加载单个.jsonl文件
        """
        dirty_path = Path(self.dirty_data_path)
        
        if not dirty_path.exists():
            logger.info(t("脏数据路径不存在: {path}, 跳过过滤", path=dirty_path))
            return
        
        # 初始化统计
        self._dirty_stats = DirtyDataStats()
        
        # 确定要加载的文件列表
        if dirty_path.is_dir():
            jsonl_files = list(dirty_path.glob("*.jsonl"))
            self._dirty_stats.total_files = len(jsonl_files)
            
            if not jsonl_files:
                logger.warning(t("脏数据文件夹为空: {path}", path=dirty_path))
                return
            
            logger.info(t("发现 {count} 个脏数据文件", count=len(jsonl_files)))
        else:
            jsonl_files = [dirty_path]
            self._dirty_stats.total_files = 1
        
        # 加载每个文件
        for file_path in jsonl_files:
            self._load_single_dirty_file(file_path)
        
        # 输出加载摘要
        self._log_dirty_data_summary()
    
    def _validate_dirty_record(self, record: dict, line_num: int, file_name: str) -> bool:
        """
        验证脏数据记录格式
        
        Args:
            record: JSON记录
            line_num: 行号
            file_name: 文件名
            
        Returns:
            bool: 是否有效
        """
        # 检查必需字段
        missing_fields = self.REQUIRED_DIRTY_DATA_FIELDS - set(record.keys())
        if missing_fields:
            logger.debug(f"[{file_name}:{line_num}] 缺少必需字段: {missing_fields}")
            return False
        
        # 检查hash字段格式 (应该是hex字符串)
        hash_value = record.get("hash", "")
        if not isinstance(hash_value, str) or len(hash_value) < 32:
            logger.debug(f"[{file_name}:{line_num}] hash字段格式无效")
            return False
        
        return True
    
    def _load_single_dirty_file(self, file_path: Path) -> None:
        """
        加载单个脏数据文件
        
        Args:
            file_path: 文件路径
        """
        file_name = file_path.name
        file_stats = {
            "file": file_name,
            "valid": 0,
            "invalid": 0,
            "total": 0,
        }
        
        try:
            # 先检查文件头部结构
            if not self._check_file_header(file_path):
                logger.warning(f"[{file_name}] 文件头部结构无效, 跳过")
                self._dirty_stats.skipped_files += 1
                file_stats["skipped"] = True
                self._dirty_stats.file_details.append(file_stats)
                return
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    file_stats["total"] += 1
                    self._dirty_stats.total_records += 1
                    
                    try:
                        record = json.loads(line)
                        
                        if self._validate_dirty_record(record, line_num, file_name):
                            self._dirty_hashes.add(record["hash"])
                            file_stats["valid"] += 1
                            self._dirty_stats.valid_records += 1
                        else:
                            file_stats["invalid"] += 1
                            self._dirty_stats.invalid_records += 1
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"[{file_name}:{line_num}] JSON解析失败: {e}")
                        file_stats["invalid"] += 1
                        self._dirty_stats.invalid_records += 1
            
            self._dirty_stats.valid_files += 1
            self._dirty_stats.file_details.append(file_stats)
            logger.debug(f"[{file_name}] 加载: {file_stats['valid']}/{file_stats['total']} 有效")
            
        except Exception as e:
            logger.error(f"[{file_name}] 加载失败: {e}")
            self._dirty_stats.skipped_files += 1
            file_stats["error"] = str(e)
            self._dirty_stats.file_details.append(file_stats)
    
    def _check_file_header(self, file_path: Path, check_lines: int = 3) -> bool:
        """
        检查文件头部结构是否符合预期
        
        Args:
            file_path: 文件路径
            check_lines: 检查的行数
            
        Returns:
            bool: 是否符合预期结构
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                valid_count = 0
                for i, line in enumerate(f):
                    if i >= check_lines:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        # 检查是否包含hash字段
                        if "hash" in record and isinstance(record["hash"], str):
                            valid_count += 1
                    except json.JSONDecodeError:
                        continue
                
                # 至少有一条有效记录才认为文件有效
                return valid_count > 0
                
        except Exception:
            return False
    
    def _log_dirty_data_summary(self) -> None:
        """输出脏数据加载摘要到日志"""
        stats = self._dirty_stats
        
        if stats.valid_records == 0:
            logger.warning(t("未加载到任何有效的脏数据"))
            return
        
        # 简洁的控制台输出
        logger.info(
            t("脏数据加载完成: {valid}/{total} 文件, {hashes} 条唯一哈希 ({valid_records} 有效/{invalid} 无效)",
              valid=stats.valid_files, total=stats.total_files, 
              hashes=len(self._dirty_hashes),
              valid_records=stats.valid_records, invalid=stats.invalid_records)
        )
        
        # 详细信息记录到DEBUG级别
        for detail in stats.file_details:
            if detail.get("skipped"):
                logger.debug(f"  ⊘ {detail['file']}: 已跳过")
            elif detail.get("error"):
                logger.debug(f"  ✗ {detail['file']}: 错误 - {detail['error']}")
            else:
                logger.debug(f"  ✓ {detail['file']}: {detail['valid']}/{detail['total']} 有效")
    
    @staticmethod
    def compute_hash(question: str, passage: str, answer: bool) -> str:
        """计算数据项哈希值"""
        return hashlib.sha256(
            f"{question}_{passage}_{answer}".encode("utf-8")
        ).hexdigest()
    
    def is_dirty(self, item: BoolQItem) -> bool:
        """检查数据项是否为脏数据"""
        return item.hash in self._dirty_hashes
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if not self._loaded:
            self.load()
        return len(self._dataset)
    
    def __iter__(self) -> Iterator[BoolQItem]:
        """迭代数据集 (跳过脏数据)"""
        if not self._loaded:
            self.load()
        
        skipped = 0
        skipped_indices = []
        
        for i, row in enumerate(self._dataset):
            item = BoolQItem(
                question=row["question"],
                passage=row["passage"],
                answer=row["answer"],
                index=i,
            )
            
            # 跳过脏数据
            if self.is_dirty(item):
                skipped += 1
                skipped_indices.append(i)
                continue
            
            yield item
        
        # 记录跳过的脏数据
        if skipped > 0:
            # 简洁的控制台/日志输出
            if skipped <= 10:
                indices_str = str(skipped_indices)
            else:
                indices_str = f"[{', '.join(map(str, skipped_indices[:5]))}, ... , {', '.join(map(str, skipped_indices[-3:]))}]"
            
            logger.info(t("跳过 {count} 条脏数据, 索引: {indices}", count=skipped, indices=indices_str))
    
    def iter_all(self) -> Iterator[BoolQItem]:
        """迭代所有数据 (包括脏数据)"""
        if not self._loaded:
            self.load()
        
        for i, row in enumerate(self._dataset):
            yield BoolQItem(
                question=row["question"],
                passage=row["passage"],
                answer=row["answer"],
                index=i,
            )
    
    def get_item(self, index: int) -> BoolQItem:
        """获取指定索引的数据项"""
        if not self._loaded:
            self.load()
        
        row = self._dataset[index]
        return BoolQItem(
            question=row["question"],
            passage=row["passage"],
            answer=row["answer"],
            index=index,
        )
    
    @property
    def total_count(self) -> int:
        """总数据条数"""
        return len(self)
    
    @property
    def clean_count(self) -> int:
        """干净数据条数"""
        if not self._loaded:
            self.load()
        
        dirty_in_dataset = sum(
            1 for item in self.iter_all() if self.is_dirty(item)
        )
        return len(self._dataset) - dirty_in_dataset
    
    @property
    def dirty_count(self) -> int:
        """脏数据条数 (在当前数据集中)"""
        if not self._loaded:
            self.load()
        
        return sum(1 for item in self.iter_all() if self.is_dirty(item))
    
    @property
    def dirty_hash_count(self) -> int:
        """已加载的脏数据哈希总数"""
        return len(self._dirty_hashes)
    
    @property
    def dirty_stats(self) -> Optional[DirtyDataStats]:
        """脏数据加载统计"""
        return self._dirty_stats

