"""
Benchmark data registry: import-time registration, metadata, inspection helpers.

数据层 :class:`~benchmark.interfaces.benchmark.Benchmark` 与
:class:`benchmark_lite.base.BenchmarkLite` 并存：后者用于交错对话+评测脚本流（如 Self_Version/LTM）。
"""
from __future__ import annotations

from typing import Any

from benchmark_lite import BenchmarkLite

from ..interfaces.benchmark import Benchmark
from .providers.evermind_ai.evermembench_static import (
    POOL_NAME,
    EverMemBenchStaticBenchmark,
)
from .providers.evermind_ai.evermembench_dynamic import (
    POOL_NAME as EVERMEMBENCH_DYNAMIC_POOL_NAME,
    EverMemBenchDynamicBenchmark,
)
from .providers.percena.Locomo.locomo_mc10 import (
    POOL_NAME as LOCOMO_MC10_POOL_NAME,
    LocomoMc10Benchmark,
)
from .providers.xiaowu0162.LongMemEval.longmemeval_cleaned import (
    POOL_NAME as LONGMEMEVAL_CLEANED_POOL_NAME,
    LongMemEvalCleanedBenchmark,
)
from .providers.self_version.LTM import (
    POOL_NAME as LTM_POOL_NAME,
    LTMBenchmarkLite,
)
from .providers.beam.beam_benchmark import BeamBenchmark

__all__ = [
    "BENCHMARKS",
    "BENCHMARK_LITE",
    "POOL_NAME",
    "EVERMEMBENCH_DYNAMIC_POOL_NAME",
    "LOCOMO_MC10_POOL_NAME",
    "LONGMEMEVAL_CLEANED_POOL_NAME",
    "LTM_POOL_NAME",
    "EverMemBenchStaticBenchmark",
    "EverMemBenchDynamicBenchmark",
    "LocomoMc10Benchmark",
    "LongMemEvalCleanedBenchmark",
    "LTMBenchmarkLite",
    "BeamBenchmark",
    "get_benchmark",
    "get_benchmark_lite",
    "build_metadata",
    "print_summary",
    "describe",
    "inspect_scene",
]

_evermembench_instance = EverMemBenchStaticBenchmark()
_evermembench_dynamic_instance = EverMemBenchDynamicBenchmark()
_locomo_mc10_instance = LocomoMc10Benchmark()
_longmemeval_oracle_instance = LongMemEvalCleanedBenchmark(split_id="oracle")
_longmemeval_s_instance = LongMemEvalCleanedBenchmark(split_id="s_cleaned")
_longmemeval_m_instance = LongMemEvalCleanedBenchmark(split_id="m_cleaned")
_ltm_lite_instance = LTMBenchmarkLite()
_beam_100k_instance = BeamBenchmark(scale="100K")
_beam_500k_instance = BeamBenchmark(scale="500K")
_beam_1m_instance = BeamBenchmark(scale="1M")
_beam_10m_instance = BeamBenchmark(scale="10M")

BENCHMARKS: dict[str, Benchmark] = {
    _evermembench_instance.benchmark_name: _evermembench_instance,
    _evermembench_dynamic_instance.benchmark_name: _evermembench_dynamic_instance,
    _locomo_mc10_instance.benchmark_name: _locomo_mc10_instance,
    _longmemeval_oracle_instance.benchmark_name: _longmemeval_oracle_instance,
    _longmemeval_s_instance.benchmark_name: _longmemeval_s_instance,
    _longmemeval_m_instance.benchmark_name: _longmemeval_m_instance,
    _beam_100k_instance.benchmark_name: _beam_100k_instance,
    _beam_500k_instance.benchmark_name: _beam_500k_instance,
    _beam_1m_instance.benchmark_name: _beam_1m_instance,
    _beam_10m_instance.benchmark_name: _beam_10m_instance,
}

BENCHMARK_LITE: dict[str, BenchmarkLite] = {
    _ltm_lite_instance.name: _ltm_lite_instance,
}

_metadata: dict[str, Any] | None = None


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {list(BENCHMARKS)!r}")
    return BENCHMARKS[name]


def get_benchmark_lite(name: str) -> BenchmarkLite:
    if name not in BENCHMARK_LITE:
        raise KeyError(
            f"Unknown benchmark_lite {name!r}. Known: {list(BENCHMARK_LITE)!r}"
        )
    return BENCHMARK_LITE[name]


def build_metadata() -> dict[str, Any]:
    """Scan layout once; safe to call multiple times (cached)."""
    global _metadata
    if _metadata is not None:
        return _metadata

    bench = _evermembench_instance
    raw_exists = bench.raw_root.is_dir()
    context_table = bench.scene_index_table()
    qar_line_counts = bench.split_line_counts()
    qar_loaded = bench.qar_counts()

    locomo = _locomo_mc10_instance
    locomo_jsonl_present = locomo.jsonl_path.is_file()

    ltm = _ltm_lite_instance
    ltm_raw = ltm.raw_root

    dyn = _evermembench_dynamic_instance
    dyn_raw = dyn.raw_root
    dyn_topics = dyn.topic_ids()

    lme_o = _longmemeval_oracle_instance
    lme_s = _longmemeval_s_instance
    lme_m = _longmemeval_m_instance
    lme_raw = lme_o.raw_root

    _metadata = {
        "benchmarks": list(BENCHMARKS.keys()),
        "benchmark_lite": list(BENCHMARK_LITE.keys()),
        "evermembench_raw_root": str(bench.raw_root),
        "evermembench_raw_present": raw_exists,
        "evermembench_context_scenes": context_table,
        "evermembench_context_scene_count": len(context_table),
        "evermembench_qar_jsonl_line_counts": qar_line_counts,
        "evermembench_qar_loaded_counts": qar_loaded,
        "evermembench_dynamic_raw_root": str(dyn_raw),
        "evermembench_dynamic_raw_present": dyn_raw.is_dir(),
        "evermembench_dynamic_topic_count": len(dyn_topics),
        "evermembench_dynamic_qa_counts_by_topic": dyn.qar_counts_by_topic(),
        "locomo_mc10_raw_root": str(locomo.raw_root),
        "locomo_mc10_jsonl": str(locomo.jsonl_path),
        "locomo_mc10_jsonl_present": locomo_jsonl_present,
        "locomo_mc10_scene_count": locomo.row_count(),
        "longmemeval_cleaned_raw_root": str(lme_raw),
        "longmemeval_cleaned_oracle_path": str(lme_o.source_path),
        "longmemeval_cleaned_oracle_present": lme_o.source_path.is_file(),
        # oracle is small; safe to index for exact count
        "longmemeval_cleaned_oracle_scene_count": lme_o.row_count(),
        "longmemeval_cleaned_s_path": str(lme_s.source_path),
        "longmemeval_cleaned_s_present": lme_s.source_path.is_file(),
        # s/m can be huge; prefer fast count only if index already exists
        "longmemeval_cleaned_s_scene_count": lme_s.indexed_row_count_fast(),
        "longmemeval_cleaned_m_path": str(lme_m.source_path),
        "longmemeval_cleaned_m_present": lme_m.source_path.is_file(),
        "longmemeval_cleaned_m_scene_count": lme_m.indexed_row_count_fast(),
        "ltm_lite_name": ltm.name,
        "ltm_pool_name": LTM_POOL_NAME,
        "ltm_raw_root": str(ltm_raw),
        "ltm_raw_present": ltm_raw.is_dir(),
        "ltm_scenario_count": ltm.scenario_count,
    }
    return _metadata


def describe() -> None:
    """Alias for :func:`print_summary` (plan API name)."""
    print_summary()


def print_summary() -> None:
    """Print registry + context scene index + qar jsonl line counts (if present)."""
    m = build_metadata()
    print("=== UniversalBenchmark.data summary ===")
    print(f"Registered benchmarks (data layer): {m['benchmarks']}")
    print(f"Registered benchmark_lite (scripted): {m['benchmark_lite']}")
    print(f"EverMemBench raw root: {m['evermembench_raw_root']}")
    print(f"EverMemBench raw present: {m['evermembench_raw_present']}")
    print(f"EverMemBench context scenes (512K+): {m['evermembench_context_scene_count']}")
    for row in m["evermembench_context_scenes"][:12]:
        print(f"  scene_id={row['scene_id']!r} scene_name={row['scene_name']!r}")
    if m["evermembench_context_scene_count"] > 12:
        print(f"  ... ({m['evermembench_context_scene_count'] - 12} more)")
    print(f"EverMemBench qar/ jsonl line counts: {m['evermembench_qar_jsonl_line_counts']}")
    print(
        "EverMemBench QAR loaded in memory (test+train jsonl, full load): "
        f"{m['evermembench_qar_loaded_counts']}"
    )
    print(f"EverMemBench-Dynamic raw root: {m['evermembench_dynamic_raw_root']}")
    print(f"EverMemBench-Dynamic raw present: {m['evermembench_dynamic_raw_present']}")
    print(f"EverMemBench-Dynamic topics: {m['evermembench_dynamic_topic_count']}")
    print(f"EverMemBench-Dynamic QA counts by topic: {m['evermembench_dynamic_qa_counts_by_topic']}")
    print(f"LoCoMo-MC10 raw root: {m['locomo_mc10_raw_root']}")
    print(f"LoCoMo-MC10 JSONL: {m['locomo_mc10_jsonl']}")
    print(f"LoCoMo-MC10 JSONL present: {m['locomo_mc10_jsonl_present']}")
    print(f"LoCoMo-MC10 indexed scenes: {m['locomo_mc10_scene_count']}")
    print(f"LongMemEval-cleaned raw root: {m['longmemeval_cleaned_raw_root']}")
    print(f"LongMemEval-cleaned oracle path: {m['longmemeval_cleaned_oracle_path']}")
    print(f"LongMemEval-cleaned oracle present: {m['longmemeval_cleaned_oracle_present']}")
    print(f"LongMemEval-cleaned oracle scenes: {m['longmemeval_cleaned_oracle_scene_count']}")
    print(f"LongMemEval-cleaned s path: {m['longmemeval_cleaned_s_path']}")
    print(f"LongMemEval-cleaned s present: {m['longmemeval_cleaned_s_present']}")
    print(f"LongMemEval-cleaned s scenes (indexed): {m['longmemeval_cleaned_s_scene_count']}")
    print(f"LongMemEval-cleaned m path: {m['longmemeval_cleaned_m_path']}")
    print(f"LongMemEval-cleaned m present: {m['longmemeval_cleaned_m_present']}")
    print(f"LongMemEval-cleaned m scenes (indexed): {m['longmemeval_cleaned_m_scene_count']}")
    print(f"LTM ({m['ltm_lite_name']!r}) raw root: {m['ltm_raw_root']}")
    print(f"LTM raw present: {m['ltm_raw_present']}  scenarios: {m['ltm_scenario_count']}")
    if not m["evermembench_raw_present"]:
        print(
            "Hint: clone EverMemBench-Static via "
            "UniversalBenchmark/benchmark/init_raw.py --only evermind/EverMemBench-Static"
        )
    if m["evermembench_dynamic_raw_present"] and m["evermembench_dynamic_topic_count"] == 0:
        print(
            "Hint: EverMemBench-Dynamic raw dir exists but no topic folders with "
            "dialogue.json + qa_*.json; run git lfs pull in the submodule root."
        )
    if not m["evermembench_dynamic_raw_present"]:
        print(
            "Hint: clone EverMemBench-Dynamic via "
            "UniversalBenchmark/benchmark/init_raw.py --only evermind/EverMemBench-Dynamic"
        )
    if not m["locomo_mc10_jsonl_present"]:
        print(
            "Hint: clone LoCoMo-MC10 submodule via "
            "UniversalBenchmark/benchmark/init_raw.py --only percena/locomo-mc10"
        )
    if not m["longmemeval_cleaned_oracle_present"]:
        print(
            "Hint: clone longmemeval-cleaned via "
            "UniversalBenchmark/benchmark/init_raw.py --only xiaowu0162/longmemeval-cleaned"
        )


def inspect_scene(
    benchmark_name: str,
    scene_id: str,
    *,
    query_preview: int = 200,
    answer_preview: int = 200,
    evidence_preview: int = 100,
    max_refs_shown: int = 2,
    background_preview_chars: int = 120,
    eval_prompt_preview: int = 240,
) -> None:
    """Load scene via registered benchmark and print truncated details (stdout)."""
    bench = get_benchmark(benchmark_name)
    scene = bench.get_scene_by_id(scene_id)
    print("=== scene inspect ===")
    print(f"benchmark: {benchmark_name!r}")
    print(f"scene_id: {scene.scene_id!r}")
    print(f"scene_name: {scene.scene_name!r}")
    print(f"task_type: {scene.task_type}")
    ep = bench.eval_prompt
    print(
        f"benchmark.eval_prompt ({len(ep)} chars): "
        f"{ep[:eval_prompt_preview]!r}{'...' if len(ep) > eval_prompt_preview else ''}"
    )
    bg = scene.background_text(max_chars=background_preview_chars)
    print(
        f"background_text preview ({len(bg)} chars, max_chars={background_preview_chars}): "
        f"{bg[:background_preview_chars]!r}{'...' if len(bg) > background_preview_chars else ''}"
    )
    hist = scene.conversation_history()
    print(f"conversation_history (turns): {len(hist)}")
    if hist:
        t0 = hist[0]
        print(f"  turn[0] user: {t0.user_message[:background_preview_chars]!r}...")
        print(f"  turn[0] asst: {t0.assistant_response[:background_preview_chars]!r}...")

    print(f"question_count: {scene.question_count()}")
    if scene.question_count() == 0:
        print("(no questions)")
        return
    q = scene.get_question_by_id("0")
    print(f"question_id (sample): {q.question_id!r}")
    qt = q.question_text
    print(f"question_text ({len(qt)} chars): {qt[:query_preview]!r}{'...' if len(qt) > query_preview else ''}")
    gt = str(q.ground_truth)
    print(f"ground_truth ({len(gt)} chars): {gt[:answer_preview]!r}{'...' if len(gt) > answer_preview else ''}")
    print(f"evidence_type: {q.evidence.evidence_type}")
    print(f"evidence payload keys: {list(q.evidence.payload.keys())}")
    docs = q.evidence.payload.get("documents") or []
    if isinstance(docs, list):
        print(f"reference_list len: {len(docs)}")
        for i, doc in enumerate(docs[:max_refs_shown]):
            s = str(doc)
            print(f"  ref[{i}] ({len(s)} chars): {s[:evidence_preview]!r}{'...' if len(s) > evidence_preview else ''}")
        if len(docs) > max_refs_shown:
            print(f"  ... ({len(docs) - max_refs_shown} more refs)")
    print(f"scoring: mode={q.scoring.eval_mode!r} key={q.scoring.eval_prompt_key!r} max={q.scoring.max_score}")


# Warm metadata once on import (directory scan only; does not load pkl)
build_metadata()
