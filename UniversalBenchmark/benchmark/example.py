#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EverMemBench-Static 用法示例（无命令行参数，改顶部常量即可）。

运行（在 MemIndex 仓库根目录）:
  py -3 UniversalBenchmark/benchmark/example.py
或在 UniversalBenchmark 目录:
  py -3 benchmark/example.py

需已克隆子模块: benchmark/data/raw/EverMind-AI/EverMemBench-Static
"""
from __future__ import annotations

import sys
from pathlib import Path

# ========== 可调常量（写死，不用 argparse）==========
BENCHMARK_NAME = "EverMind-AI/EverMemBench-Static"  # 注册表里的 benchmark 名称
SCENE_ID = "0"  # 纯数字字符串：第几个上下文规模场景（0=最小 512K 档）
QUESTION_ID = "0"  # 题目在合并 QAR 列表中的下标（先 test 再 train）
MAX_BG_PREVIEW_CHARS = 800  # 打印背景语料时的最大字符数（避免刷屏）
USE_LAZY_CORPUS = False  # True：语料 unique_reference.pkl 延后到首次 background_text 再加载


def _ensure_import_path() -> None:
    """把 UniversalBenchmark 根目录加入 sys.path，才能 import benchmark。"""
    ub_root = Path(__file__).resolve().parents[1]
    s = str(ub_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def main() -> None:
    _ensure_import_path()
    import benchmark.data as ubdata

    # 打印注册表与元数据摘要（import 时已扫描目录；QAR 全量在 Benchmark 构造时已读入 jsonl）
    ubdata.print_summary()

    # 遍历 BENCHMARKS：共有几个 benchmark，各自有多少个 scene（list_scenes 的长度）
    print("\n=== 注册表概览：Benchmark 数量与每个 Benchmark 的 Scene 数量 ===")
    print(f"  benchmark 总数: {len(ubdata.BENCHMARKS)}")
    for name, b in ubdata.BENCHMARKS.items():
        scene_ids = list(b.list_scenes())
        print(f"  • 名称 {name!r} → scene 数量: {len(scene_ids)}")

    # 取得 Benchmark：一般用注册表里的单例；惰性语料时需单独 new 一个实例
    if USE_LAZY_CORPUS:
        from benchmark.data.providers.evermind_ai.evermembench_static import (
            EverMemBenchStaticBenchmark,
        )

        bench = EverMemBenchStaticBenchmark(lazy_corpus=True)
        print("\n[提示] USE_LAZY_CORPUS=True：上面 main 里用 bench 取的 Scene 会延迟加载语料；")
        print("       末尾 inspect_scene 仍用注册表单例，会立刻加载语料（二者可对比）。")
    else:
        # 与 ubdata.BENCHMARKS[BENCHMARK_NAME] 是同一实例
        bench = ubdata.get_benchmark(BENCHMARK_NAME)

    # ---------- 维度：多少 scene、每 scene 多少题（各 scene 共用同一份全量 QAR）----------
    qc = bench.qar_counts()
    rows = bench.scene_dimension_table()
    print("\n=== 数据规模 ===")
    print(f"  scene 数量: {len(rows)}")
    print(f"  已加载 QAR: test={qc['qar_test']}, train={qc['qar_train']}, 合计={qc['qar_total']}")
    print("  各 scene（scene_name=语料规模目录名, question_count=该场景下题目数）:")
    for r in rows:
        print(f"    id={r['scene_id']!s}  name={r['scene_name']!s}  questions={r['question_count']}")

    # ---------- 默认评测 prompt（本 benchmark 自带）----------
    ep = bench.eval_prompt
    print(f"\n=== eval_prompt（共 {len(ep)} 字符，仅预览前 120 字）===")
    print(repr(ep[:120] + ("..." if len(ep) > 120 else "")))

    # ---------- 按 scene_id 取 Scene，再按 question_id 取 Question ----------
    scene_now = bench.get_scene_by_id(SCENE_ID)
    print(f"\n=== Scene === id={scene_now.scene_id!r} name={scene_now.scene_name!r}")
    print(f"  question_count: {scene_now.question_count()}")
    bg = scene_now.background_text(max_chars=MAX_BG_PREVIEW_CHARS)
    print(f"  background_text 预览长度: {len(bg)}（max_chars={MAX_BG_PREVIEW_CHARS}）")

    q = scene_now.get_question_by_id(QUESTION_ID)
    print(f"\n=== Question === id={q.question_id!r}")
    print(f"  evidence.n_refs: {q.evidence.payload.get('n_refs')}")

    # 工具函数：按 benchmark 名 + scene_id 打印截断详情（内部同样是 get_benchmark → get_scene_by_id）
    print(f"\n=== inspect_scene（汇总预览）===")
    ubdata.inspect_scene(BENCHMARK_NAME, SCENE_ID)


if __name__ == "__main__":
    main()
