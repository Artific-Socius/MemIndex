"""MemIndex benchmark adapter for benchmark_lite.

Usage::

    python run_benchmark.py \\
        --benchmark benchmark_lite.benchmarks.memindex.MemIndexBenchmark \\
        --memory buffer \\
        --model openrouter/google/gemini-2.5-flash
"""

from .benchmark import MemIndexBenchmark

__all__ = ["MemIndexBenchmark"]
