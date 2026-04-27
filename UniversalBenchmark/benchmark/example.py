#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark 数据层用法示例（无命令行参数，改顶部 ``BENCHMARK_NAME`` / ``SCENE_ID`` 即可）。

运行（在 MemIndex 仓库根目录）:
  py -3 UniversalBenchmark/benchmark/example.py
或在 UniversalBenchmark 目录:
  py -3 benchmark/example.py

仅输出 EverMemBench-Dynamic（不跑 ``BENCHMARK_NAME`` 主流程、不打印全量 ``print_summary``）::

  将下方 ``OUTPUT_MODE`` 设为 ``\"dynamic_only\"`` 后再运行。

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
# 输出模式: "full" = 主流程 +（可选）Dynamic 专节；"dynamic_only" = 只打印 Dynamic 专节
OUTPUT_MODE = "full"  # 改为 "dynamic_only" 即单独输出 Dynamic 示例

# EverMem 示例:
#   BENCHMARK_NAME = "EverMind-AI/EverMemBench-Static"
#   SCENE_ID = "0"  # 第几个 512K+ 语料场景
# EverMemBench-Dynamic 示例:
#   BENCHMARK_NAME = "EverMind-AI/EverMemBench-Dynamic"
#   SCENE_ID = "01"  # topic 目录名（01..05），每 topic 一份 dialogue + qa
# LoCoMo-MC10 示例:
#   BENCHMARK_NAME = "Percena/LoCoMo-MC10"
#   SCENE_ID = "conv-26"  # scene_id 为对话 sample_id（同一 conv 下多题合并为一个 scene）
BENCHMARK_NAME = "Percena/LoCoMo-MC10"
SCENE_ID = "conv-26"
QUESTION_ID = "conv-26_q0"  # Static/Dynamic：场景内题目下标字符串；LoCoMo：真实 question_id（如 conv-26_q0）
MAX_BG_PREVIEW_CHARS = 800  # 打印背景语料时的最大字符数（避免刷屏）
USE_LAZY_CORPUS = False  # 仅 EverMem Static：True 时延后加载 unique_reference.pkl
# 为 True 且在 OUTPUT_MODE=="full" 时，在 main 末尾额外打印 EverMemBench-Dynamic 专节（与 BENCHMARK_NAME 无关）
SHOW_EVERMEMBENCH_DYNAMIC_WALKTHROUGH = True
_DYNAMIC_HF = "https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic"

# LongMemEval-cleaned
_LONGMEMEVAL_HF = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"


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


def _print_longmemeval_cleaned_demo(ubdata, *, doc_preview: int) -> None:
    """LongMemEval-cleaned 独立示例：oracle split 的首条样本。"""
    name = "xiaowu0162/longmemeval-cleaned:oracle"
    print(f"\n=== LongMemEval-cleaned 示例（HF: {_LONGMEMEVAL_HF}）===")
    b = ubdata.get_benchmark(name)
    print(f"  benchmark: {name!r}")
    print(f"  raw_root: {b.raw_root}")
    print(f"  source_path: {b.source_path}")
    print(f"  scenes/items: {b.row_count()}")
    if b.row_count() == 0:
        print("  (无数据：请先 init_raw + git lfs pull)")
        return
    s = b.get_scene_by_id("oracle:0")
    print(f"  scene_id='oracle:0' scene_name={s.scene_name!r} conv_turns={len(s.conversation_history())}")
    q = s.get_question_by_id("0")
    print(f"  Q: {q.question_text!r}")
    print(f"  A: {str(q.ground_truth)!r}")
    pl = q.evidence.payload
    print(f"  evidence keys: {list(pl.keys())}")
    # show a bit of the first two turns so it's obvious it's conversation-style
    hist = s.conversation_history()
    if hist:
        u0 = hist[0].user_message
        a0 = hist[0].assistant_response
        print(f"  first user turn preview: {u0[:doc_preview]!r}{'...' if len(u0) > doc_preview else ''}")
        print(f"  first assistant turn preview: {a0[:doc_preview]!r}{'...' if len(a0) > doc_preview else ''}")


def _interactive_demo_menu(ubdata) -> None:
    """不刷全量 Registered benchmarks；固定序号菜单，选择后才输出详情。"""
    demos: list[tuple[str, str, callable]] = []

    def _show_benchmark_detail(bench_name: str) -> None:
        print(f"\n=== Benchmark detail: {bench_name} ===")
        b = ubdata.get_benchmark(bench_name)
        scene_ids = list(b.list_scenes())
        print(f"  scenes: {len(scene_ids)}")
        if not scene_ids:
            return
        ubdata.inspect_scene(bench_name, scene_ids[0])

    def _show_lite_detail(lite_name: str) -> None:
        print(f"\n=== BenchmarkLite detail: {lite_name} ===")
        b = ubdata.get_benchmark_lite(lite_name)
        print(f"  name: {b.name!r}")
        rr = getattr(b, "raw_root", None)
        if rr is not None:
            print(f"  raw_root: {rr}")
        sc = getattr(b, "scenario_count", None)
        if sc is not None:
            print(f"  scenario_count: {sc}")
        ids_fn = getattr(b, "list_scenario_ids", None)
        if callable(ids_fn):
            ids = ids_fn()
            print(f"  scenario_ids preview: {ids[:5]!r}{'...' if len(ids) > 5 else ''}")

    def _warn_longmemeval_big(split_name: str) -> None:
        if split_name in ("s_cleaned", "m_cleaned"):
            print(
                "  [提示] s/m split 首次查看会构建 offset 索引，可能需要较长时间，"
                "并在 raw/.indexes 下生成 offsets.json。"
            )

    # 固定序号：删除旧的 2（oracle demo），并让 6/7/8 对应 longmemeval 三个 split
    demos.append(("1", "EverMemBench-Dynamic (topic=01 demo)", lambda: _print_evermembench_dynamic_demo(ubdata, max_bg=MAX_BG_PREVIEW_CHARS, q_preview=160, doc_preview=220)))
    demos.append(("2", "[Benchmark] EverMind-AI/EverMemBench-Static", lambda: _show_benchmark_detail("EverMind-AI/EverMemBench-Static")))
    demos.append(("3", "[Benchmark] EverMind-AI/EverMemBench-Dynamic", lambda: _show_benchmark_detail("EverMind-AI/EverMemBench-Dynamic")))
    demos.append(("4", "[Benchmark] Percena/LoCoMo-MC10", lambda: _show_benchmark_detail("Percena/LoCoMo-MC10")))
    demos.append(("5", "[BenchmarkLite] Self_Version/LTM", lambda: _show_lite_detail("Self_Version/LTM")))

    demos.append(("6", "[Benchmark] xiaowu0162/longmemeval-cleaned:oracle", lambda: _show_benchmark_detail("xiaowu0162/longmemeval-cleaned:oracle")))
    demos.append(("7", "[Benchmark] xiaowu0162/longmemeval-cleaned:s_cleaned", lambda: (_warn_longmemeval_big("s_cleaned"), _show_benchmark_detail("xiaowu0162/longmemeval-cleaned:s_cleaned"))))
    demos.append(("8", "[Benchmark] xiaowu0162/longmemeval-cleaned:m_cleaned", lambda: (_warn_longmemeval_big("m_cleaned"), _show_benchmark_detail("xiaowu0162/longmemeval-cleaned:m_cleaned"))))

    def _print_all() -> None:
        """全量打印（便于排查注册表/路径/元数据）。"""
        print("\n=== 全量打印 ===")
        ubdata.print_summary()
        print("\n=== Registered BENCHMARKS keys ===")
        for k in ubdata.BENCHMARKS.keys():
            print(f"  - {k}")
        print("\n=== Registered BENCHMARK_LITE keys ===")
        for k in getattr(ubdata, "BENCHMARK_LITE", {}).keys():
            print(f"  - {k}")

    demos.insert(0, ("0", "[All] print_summary + registries", _print_all))

    print("\n=== 选择序号显示详情（空输入退出）===")
    for k, title, _ in demos:
        print(f"  {k}. {title}")
    choice = input("选择> ").strip()
    if not choice:
        return
    for k, _, fn in demos:
        if choice == k:
            fn()
            return
    print(f"未知选择: {choice!r}")


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


def main_dynamic_only(ubdata: object | None = None) -> None:
    """只输出 EverMemBench-Dynamic 示例（无 print_summary、无其它 benchmark 主流程）。"""
    if ubdata is None:
        _ensure_import_path()
        import benchmark.data as ubdata_mod

        ubdata = ubdata_mod

    print("=== OUTPUT_MODE=dynamic_only：仅 EverMemBench-Dynamic ===\n")
    _print_evermembench_dynamic_demo(
        ubdata,
        max_bg=MAX_BG_PREVIEW_CHARS,
        q_preview=160,
        doc_preview=220,
    )


def main() -> None:
    _ensure_import_path()
    import benchmark.data as ubdata

    if OUTPUT_MODE == "dynamic_only":
        main_dynamic_only(ubdata)
        return
    if OUTPUT_MODE != "full":
        raise ValueError(f"未知 OUTPUT_MODE={OUTPUT_MODE!r}，请使用 'full' 或 'dynamic_only'")

    # 改为交互选择：按序号输出详情（不全量 print_summary / Registered benchmarks）
    _interactive_demo_menu(ubdata)


if __name__ == "__main__":
    main()
