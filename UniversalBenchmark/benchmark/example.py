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
  ``EverMind-AI/EverMemBench-Dynamic`` → 同上 ``--only evermind/EverMemBench-Dynamic``
  ``Percena/LoCoMo-MC10`` → 同上 ``--only percena/locomo-mc10``

EverMemBench-Dynamic 数据形态（与 Static 对照）:
  每个 topic 目录 ``01``..``05`` 下 ``dialogue.json``（按日期/群组的多轮对话）+
  ``qa_{topic}.json``（``Q``/``A``/``R``/``options``）。详见
  https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic
"""
from __future__ import annotations

import sys
from pathlib import Path

# ========== 可调常量（写死，不用 argparse）==========
# EverMem 示例:
#   BENCHMARK_NAME = "EverMind-AI/EverMemBench-Static"
#   SCENE_ID = "0"  # 第几个 512K+ 语料场景
# EverMemBench-Dynamic 示例:
#   BENCHMARK_NAME = "EverMind-AI/EverMemBench-Dynamic"
#   SCENE_ID = "01"  # topic 目录名（01..05），每 topic 一份 dialogue + qa
# LoCoMo-MC10 示例:
#   BENCHMARK_NAME = "Percena/LoCoMo-MC10"
#   SCENE_ID = "conv-26_q0"  # JSONL 里的 question_id，每题一个 scene
BENCHMARK_NAME = "Percena/LoCoMo-MC10"
SCENE_ID = "conv-26_q0"
QUESTION_ID = "0"  # Static/Dynamic：场景内题目下标字符串；LoCoMo 每 scene 仅 "0"
MAX_BG_PREVIEW_CHARS = 800  # 打印背景语料时的最大字符数（避免刷屏）
USE_LAZY_CORPUS = False  # 仅 EverMem Static：True 时延后加载 unique_reference.pkl
# 为 True 时，在 main 末尾额外打印 EverMemBench-Dynamic 专节（与上面 BENCHMARK_NAME 无关）
SHOW_EVERMEMBENCH_DYNAMIC_WALKTHROUGH = True
_DYNAMIC_HF = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic"


def _ensure_import_path() -> None:
    """把 UniversalBenchmark 根目录加入 sys.path，才能 import benchmark。"""
    ub_root = Path(__file__).resolve().parents[1]
    s = str(ub_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _normalize_benchmark_name(name: str) -> str:
    """避免从终端复制快照行时带上 ``[...]`` 方括号。"""
    s = name.strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        return s[1:-1].strip()
    return s


def _print_evermembench_dynamic_demo(ubdata, *, max_bg: int, q_preview: int, doc_preview: int) -> None:
    """EverMemBench-Dynamic 独立示例：topic、语料块、首题与 R 解析后的 evidence。"""
    dyn_name = "EverMind-AI/EverMemBench-Dynamic"
    print(f"\n=== EverMemBench-Dynamic 示例（HF: {_DYNAMIC_HF}）===")
    print("  子模块: raw/EverMind-AI/EverMemBench-Dynamic 下各 topic 的 dialogue.json + qa_NN.json")
    b = ubdata.get_benchmark(dyn_name)
    raw = b.raw_root
    print(f"  raw_root: {raw}")
    topics = b.topic_ids()
    if not topics:
        print(
            "  (当前无 topic：请执行 init_raw 并在子模块内 git lfs pull，"
            "确保 01/dialogue.json 与 01/qa_01.json 等为真实文件)"
        )
        return
    print(f"  topic_ids: {topics}")
    print(f"  各 topic 题目数: {b.qar_counts_by_topic()}")
    sid = "01" if "01" in topics else topics[0]
    scene = b.get_scene_by_id(sid)
    docs = scene.background_documents()
    print(f"  选取 scene_id={sid!r}  background_documents 块数: {len(docs)}")
    if docs:
        head = docs[0]
        tail = "..." if len(head) > doc_preview else ""
        print(f"  首块预览 ({min(doc_preview, len(head))} 字): {head[:doc_preview]!r}{tail}")
    bg_len = len(scene.background_text(max_chars=max_bg))
    print(f"  background_text(max_chars={max_bg}) 长度: {bg_len}")
    if scene.question_count() == 0:
        print("  (该 topic 无题目)")
        return
    q = scene.get_question_by_id("0")
    qt = q.question_text
    print(f"  首题 question_id=0  Q 预览: {qt[:q_preview]!r}{'...' if len(qt) > q_preview else ''}")
    print(f"  首题 ground_truth 预览: {str(q.ground_truth)[:q_preview]!r}")
    pl = q.evidence.payload
    nrefs = pl.get("n_refs")
    raws = pl.get("references") or []
    drefs = pl.get("documents") or []
    print(f"  evidence: type={q.evidence.evidence_type!r} n_refs={nrefs} R条数={len(raws)} documents条数={len(drefs)}")
    if isinstance(drefs, list) and drefs:
        d0 = str(drefs[0])
        print(f"  首条 resolved document 预览: {d0[:doc_preview]!r}{'...' if len(d0) > doc_preview else ''}")
    print("  说明: R 中 date/group/message_index 与对话对齐方式见 HF Dataset Card 的 Locating reference evidence。")


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
        bench = ubdata.get_benchmark(_normalize_benchmark_name(BENCHMARK_NAME))

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
    elif hasattr(bench, "qar_counts_by_topic") and hasattr(bench, "topic_ids"):
        print(f"  benchmark: {bench.benchmark_name!r}（每 topic 目录 01..05 一 scene）")
        print(f"  topic 列表: {bench.topic_ids()!r}")
        print(f"  各 topic 题目数: {bench.qar_counts_by_topic()}")
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
    ndoc = len(scene_now.background_documents())
    print(
        f"  background_text 预览长度: {len(bg)}（max_chars={MAX_BG_PREVIEW_CHARS}；"
        f"background_documents 块数={ndoc}；Dynamic 为多块语料、LoCoMo 默认对话预载时 bg 多为 0）"
    )

    q = scene_now.get_question_by_id(QUESTION_ID)
    print(f"\n=== Question === id={q.question_id!r}")
    pl = q.evidence.payload
    if "n_refs" in pl:
        print(f"  evidence.n_refs: {pl.get('n_refs')}")
    else:
        print(f"  evidence_type: {q.evidence.evidence_type!r}  payload keys: {list(pl.keys())}")

    # 工具函数：按 benchmark 名 + scene_id 打印截断详情（内部同样是 get_benchmark → get_scene_by_id）
    print("\n=== inspect_scene（汇总预览）===")
    ubdata.inspect_scene(_normalize_benchmark_name(BENCHMARK_NAME), SCENE_ID)

    if SHOW_EVERMEMBENCH_DYNAMIC_WALKTHROUGH:
        _print_evermembench_dynamic_demo(
            ubdata,
            max_bg=MAX_BG_PREVIEW_CHARS,
            q_preview=160,
            doc_preview=220,
        )


if __name__ == "__main__":
    main()
