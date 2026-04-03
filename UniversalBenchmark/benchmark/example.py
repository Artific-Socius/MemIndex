#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark 数据层用法示例（无命令行参数，改顶部 ``BENCHMARK_NAME`` / ``SCENE_ID`` 即可）。

运行（在 MemIndex 仓库根目录）:
  py -3 UniversalBenchmark/benchmark/example.py
或在 UniversalBenchmark 目录:
  py -3 benchmark/example.py

数据子模块:
  ``EverMind-AI/EverMemBench-Static`` → ``UniversalBenchmark/benchmark/init_raw.py``
  ``Percena/LoCoMo-MC10`` → 同上 ``--only percena/locomo-mc10``
"""
from __future__ import annotations

import sys
from pathlib import Path

# ========== 可调常量（写死，不用 argparse）==========
# EverMem 示例:
#   BENCHMARK_NAME = "EverMind-AI/EverMemBench-Static"
#   SCENE_ID = "0"  # 第几个 512K+ 语料场景
# LoCoMo-MC10 示例:
#   BENCHMARK_NAME = "Percena/LoCoMo-MC10"
#   SCENE_ID = "conv-26_q0"  # JSONL 里的 question_id，每题一个 scene
BENCHMARK_NAME = "Percena/LoCoMo-MC10"
SCENE_ID = "conv-26_q0"
QUESTION_ID = "0"  # EverMem：QAR 下标；LoCoMo：每 scene 仅一题，固定 "0"
MAX_BG_PREVIEW_CHARS = 800  # 打印背景语料时的最大字符数（避免刷屏）
USE_LAZY_CORPUS = False  # 仅 EverMem：True 时延后加载 unique_reference.pkl


def _ensure_import_path() -> None:
    """把 UniversalBenchmark 根目录加入 sys.path，才能 import benchmark。"""
    ub_root = Path(__file__).resolve().parents[1]
    s = str(ub_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _one_line_benchmark_snapshot(b, *, max_bg: int) -> str:
    """每个已注册 benchmark 至少打印一行：用 list_scenes 首项探测 Scene。"""
    ids = list(b.list_scenes())
    if not ids:
        return "scenes=0 (empty)"
    sid = ids[0]
    try:
        sc = b.get_scene_by_id(sid)
    except Exception as ex:
        return f"首 scene_id={sid!r} 加载失败: {ex!r}"
    nq = sc.question_count()
    nconv = len(sc.conversation_history())
    bg_len = len(sc.background_text(max_chars=max_bg))
    ep_len = len(getattr(b, "eval_prompt", "") or "")
    sn = sc.scene_name
    return (
        f"首 scene_id={sid!r} scene_name={sn!r} questions={nq} "
        f"conv_turns={nconv} bg_preview_len={bg_len} eval_prompt_chars={ep_len}"
    )


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
        print(f"  - 名称 {name!r} -> scene 数量: {len(scene_ids)}")

    print("\n=== 各 Benchmark 一行快照（每注册项必有输出）===")
    for name, b in ubdata.BENCHMARKS.items():
        snap = _one_line_benchmark_snapshot(b, max_bg=MAX_BG_PREVIEW_CHARS)
        print(f"  [{name}] {snap}")

    # 取得 Benchmark：一般用注册表里的单例；惰性语料仅 EverMem 支持
    if USE_LAZY_CORPUS and BENCHMARK_NAME == "EverMind-AI/EverMemBench-Static":
        from benchmark.data.providers.evermind_ai.evermembench_static import (
            EverMemBenchStaticBenchmark,
        )

        bench = EverMemBenchStaticBenchmark(lazy_corpus=True)
        print("\n[提示] USE_LAZY_CORPUS=True：本段 bench 延迟加载语料；")
        print("       末尾 inspect_scene 仍用注册表单例，会立刻加载语料（二者可对比）。")
    elif USE_LAZY_CORPUS:
        bench = ubdata.get_benchmark(BENCHMARK_NAME)
        print("\n[提示] USE_LAZY_CORPUS 仅对 EverMem 生效；当前 benchmark 使用注册表单例。")
    else:
        bench = ubdata.get_benchmark(BENCHMARK_NAME)

    # ---------- 数据规模（EverMem：共享 QAR + 多语料 scene；LoCoMo：一题一 scene）----------
    print("\n=== 数据规模 ===")
    if hasattr(bench, "qar_counts") and hasattr(bench, "scene_dimension_table"):
        qc = bench.qar_counts()
        rows = bench.scene_dimension_table()
        print(f"  scene 数量: {len(rows)}")
        print(f"  已加载 QAR: test={qc['qar_test']}, train={qc['qar_train']}, 合计={qc['qar_total']}")
        print("  各 scene（scene_name=语料规模目录名, question_count=该场景下题目数）:")
        for r in rows:
            print(f"    id={r['scene_id']!s}  name={r['scene_name']!s}  questions={r['question_count']}")
    elif hasattr(bench, "row_count") and hasattr(bench, "jsonl_path"):
        n = bench.row_count()
        print(f"  benchmark: {bench.benchmark_name!r}（LoCoMo-MC10：每条 JSONL = 1 scene）")
        print(f"  已索引 scene（题目）数: {n}")
        print(f"  JSONL 路径: {bench.jsonl_path}")
        if n:
            ids = list(bench.list_scenes())
            print(f"  scene_id 示例（前 3 个）: {ids[:3]!r}")
    else:
        scene_ids = list(bench.list_scenes())
        print(f"  scene 数量: {len(scene_ids)}（未识别的 benchmark 类型，仅列出 scene 数）")

    # ---------- 默认评测 prompt（本 benchmark 自带）----------
    ep = bench.eval_prompt
    print(f"\n=== eval_prompt（共 {len(ep)} 字符，仅预览前 120 字）===")
    print(repr(ep[:120] + ("..." if len(ep) > 120 else "")))

    # ---------- 按 scene_id 取 Scene，再按 question_id 取 Question ----------
    scene_now = bench.get_scene_by_id(SCENE_ID)
    print(f"\n=== Scene === id={scene_now.scene_id!r} name={scene_now.scene_name!r}")
    print(f"  question_count: {scene_now.question_count()}")
    hist = scene_now.conversation_history()
    print(f"  conversation_history 轮数（user/assistant 对）: {len(hist)}")
    if hist:
        u0 = hist[0].user_message
        a0 = hist[0].assistant_response
        print(f"  首轮 user 预览 ({min(80, len(u0))} 字): {u0[:80]!r}{'...' if len(u0) > 80 else ''}")
        print(f"  首轮 assistant 预览 ({min(80, len(a0))} 字): {a0[:80]!r}{'...' if len(a0) > 80 else ''}")
    bg = scene_now.background_text(max_chars=MAX_BG_PREVIEW_CHARS)
    print(f"  background_text 预览长度: {len(bg)}（max_chars={MAX_BG_PREVIEW_CHARS}；LoCoMo 默认走对话预载时多为 0）")

    q = scene_now.get_question_by_id(QUESTION_ID)
    print(f"\n=== Question === id={q.question_id!r}")
    pl = q.evidence.payload
    if "n_refs" in pl:
        print(f"  evidence.n_refs: {pl.get('n_refs')}")
    else:
        print(f"  evidence_type: {q.evidence.evidence_type!r}  payload keys: {list(pl.keys())}")

    # 工具函数：按 benchmark 名 + scene_id 打印截断详情（内部同样是 get_benchmark → get_scene_by_id）
    print("\n=== inspect_scene（汇总预览）===")
    ubdata.inspect_scene(BENCHMARK_NAME, SCENE_ID)


if __name__ == "__main__":
    main()
