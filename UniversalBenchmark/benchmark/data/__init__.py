"""
Benchmark data registry: import-time registration, metadata, inspection helpers.
"""
from __future__ import annotations

from typing import Any

from ..interfaces.benchmark import Benchmark
from .providers.evermind_ai.evermembench_static import (
    POOL_NAME,
    EverMemBenchStaticBenchmark,
)

__all__ = [
    "BENCHMARKS",
    "POOL_NAME",
    "EverMemBenchStaticBenchmark",
    "get_benchmark",
    "build_metadata",
    "print_summary",
    "describe",
    "inspect_scene",
]

_evermembench_instance = EverMemBenchStaticBenchmark()

BENCHMARKS: dict[str, Benchmark] = {
    _evermembench_instance.benchmark_name: _evermembench_instance,
}

_metadata: dict[str, Any] | None = None


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {list(BENCHMARKS)!r}")
    return BENCHMARKS[name]


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

    _metadata = {
        "benchmarks": list(BENCHMARKS.keys()),
        "evermembench_raw_root": str(bench.raw_root),
        "evermembench_raw_present": raw_exists,
        "evermembench_context_scenes": context_table,
        "evermembench_context_scene_count": len(context_table),
        "evermembench_qar_jsonl_line_counts": qar_line_counts,
        "evermembench_qar_loaded_counts": qar_loaded,
    }
    return _metadata


def describe() -> None:
    """Alias for :func:`print_summary` (plan API name)."""
    print_summary()


def print_summary() -> None:
    """Print registry + context scene index + qar jsonl line counts (if present)."""
    m = build_metadata()
    print("=== UniversalBenchmark.data summary ===")
    print(f"Registered benchmarks: {m['benchmarks']}")
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
    if not m["evermembench_raw_present"]:
        print(
            "Hint: clone submodule via "
            "UniversalBenchmark/benchmark/data/TEMP_init_single_benchmark.py"
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
