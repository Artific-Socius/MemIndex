"""
工具类单元测试
"""
import pytest
import sys
import time
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import Timer
from utils.utils import StatisticsHelper, ProgressManager


class TestTimer:
    """Timer测试"""
    
    def test_basic_timing(self):
        """测试基本计时功能"""
        timer = Timer("test")
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        assert timer.elapsed is not None
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2
    
    def test_auto_start(self):
        """测试自动启动"""
        timer = Timer("test", auto_start=True)
        
        assert timer.start_time is not None
    
    def test_chain_call(self):
        """测试链式调用"""
        timer = Timer("test")
        result = timer.start()
        
        assert result is timer
    
    def test_format_time_ns(self):
        """测试纳秒格式化"""
        timer = Timer()
        formatted = timer._format_time(1e-9)
        
        assert "ns" in formatted
    
    def test_format_time_us(self):
        """测试微秒格式化"""
        timer = Timer()
        formatted = timer._format_time(1e-6)
        
        assert "μs" in formatted
    
    def test_format_time_ms(self):
        """测试毫秒格式化"""
        timer = Timer()
        formatted = timer._format_time(0.001)
        
        assert "ms" in formatted
    
    def test_format_time_s(self):
        """测试秒格式化"""
        timer = Timer()
        formatted = timer._format_time(1.5)
        
        assert "s" in formatted
    
    def test_format_time_minutes(self):
        """测试分钟格式化"""
        timer = Timer()
        formatted = timer._format_time(90)
        
        assert "m" in formatted
    
    def test_format_time_hours(self):
        """测试小时格式化"""
        timer = Timer()
        formatted = timer._format_time(3700)
        
        assert "h" in formatted
    
    def test_get_format(self):
        """测试获取格式化字符串"""
        timer = Timer("TestTimer", auto_start=True)
        time.sleep(0.05)
        timer.stop()
        
        formatted = timer.get_format()
        
        assert "TestTimer" in formatted
        assert "耗时" in formatted
        assert "CPU" in formatted


class TestStatisticsHelper:
    """StatisticsHelper测试"""
    
    def test_distribution_summary_empty(self):
        """测试空数据分布摘要"""
        result = StatisticsHelper.print_distribution_summary([])
        
        assert "No data" in result
    
    def test_distribution_summary(self):
        """测试分布摘要"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = StatisticsHelper.print_distribution_summary(data)
        
        assert "样本数" in result
        assert "平均值" in result
        assert "中位数" in result
        assert "标准差" in result
    
    def test_histogram_empty(self):
        """测试空数据直方图"""
        result = StatisticsHelper.print_text_histogram_quantile([])
        
        assert "空" in result or "Error" in result
    
    def test_histogram_basic(self):
        """测试基本直方图"""
        import random
        data = [random.gauss(0, 1) for _ in range(100)]
        
        result = StatisticsHelper.print_text_histogram_quantile(data, num_bins=5)
        
        assert "直方图" in result
        assert "█" in result or "Bins" in result


class TestProgressManager:
    """ProgressManager测试"""
    
    def test_init(self):
        """测试初始化"""
        pm = ProgressManager(total=100, description="Test")
        
        assert pm.total == 100
        assert pm.description == "Test"
        assert pm.completed == 0
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with ProgressManager(total=10, description="Test") as pm:
            assert pm._progress is not None
            assert pm._task_id is not None
    
    def test_advance(self):
        """测试进度推进"""
        with ProgressManager(total=10, description="Test") as pm:
            pm.advance(accuracy=0.5, errors=0, avg_logprobs=-0.3)
            
            assert pm.completed == 1
            assert pm.accuracy == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
