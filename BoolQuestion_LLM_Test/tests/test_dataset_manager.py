"""
DatasetManager单元测试

注意: 这些测试需要网络连接来下载数据集
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_manager import DatasetManager
from utils.models_utils import BoolQItem


class TestDatasetManager:
    """DatasetManager测试"""
    
    def test_init(self):
        """测试初始化"""
        dm = DatasetManager(split="validation", limit=10)
        
        assert dm.split == "validation"
        assert dm.limit == 10
        assert dm._loaded is False
    
    def test_compute_hash(self):
        """测试哈希计算"""
        hash1 = DatasetManager.compute_hash("Q1", "P1", True)
        hash2 = DatasetManager.compute_hash("Q1", "P1", True)
        hash3 = DatasetManager.compute_hash("Q2", "P1", True)
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    @pytest.mark.skipif(True, reason="需要网络连接")
    def test_load_dataset(self):
        """测试加载数据集"""
        dm = DatasetManager(split="validation", limit=5)
        dm.load()
        
        assert dm._loaded is True
        assert len(dm) == 5
    
    @pytest.mark.skipif(True, reason="需要网络连接")
    def test_iterate_dataset(self):
        """测试迭代数据集"""
        dm = DatasetManager(split="validation", limit=3)
        dm.load()
        
        items = list(dm)
        
        assert len(items) == 3
        assert all(isinstance(item, BoolQItem) for item in items)
    
    def test_is_dirty(self):
        """测试脏数据检查"""
        dm = DatasetManager()
        dm._dirty_hashes = {"hash1", "hash2"}
        dm._loaded = True
        
        item_dirty = MagicMock()
        item_dirty.hash = "hash1"
        
        item_clean = MagicMock()
        item_clean.hash = "hash3"
        
        assert dm.is_dirty(item_dirty) is True
        assert dm.is_dirty(item_clean) is False


class TestBoolQItemIntegration:
    """BoolQItem集成测试"""
    
    def test_item_creation(self):
        """测试数据项创建"""
        item = BoolQItem(
            question="Is Python a programming language?",
            passage="Python is a high-level programming language.",
            answer=True,
            index=0,
        )
        
        assert item.question == "Is Python a programming language?"
        assert item.answer is True
        assert item.hash is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
