from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from rich.table import Table

from agent import Agent
from agent.memory import BufferMemory
from agent.progress import get_console, get_progress, loguru_sink_message, progress_context
from benchmark_lite.evaluators import get_evaluator
from benchmark_lite.types import BenchmarkResult

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. The following question may contain traps. "
    "Carefully distinguish whether it is answerable from the provided context. "
    "If it is answerable, answer precisely; if it is not answerable, refuse and say it is not answerable."
)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    colorize=sys.stderr.isatty(),
)


def _load_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"无法解析 benchmark 路径: {dotted_path}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ValueError(f"benchmark 路径不是有效类: {dotted_path}")
    return cls


def _try_recover_benchmark_prompt(result_payload: dict[str, Any]) -> str:
    run_cfg = result_payload.get("run_config")
    if not isinstance(run_cfg, dict):
        return ""
    extra = run_cfg.get("extra", {})
    if not isinstance(extra, dict):
        return ""
    benchmark_path = extra.get("benchmark_path", "")
    if not isinstance(benchmark_path, str) or not benchmark_path.strip():
        return ""
    try:
        benchmark_cls = _load_class(benchmark_path.strip())
        benchmark_obj = benchmark_cls()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"恢复 benchmark eval_prompt 失败（已回退默认 evaluator prompt）: {exc}")
        return ""
    prompt = getattr(benchmark_obj, "eval_prompt", "") or ""
    if not isinstance(prompt, str):
        return ""
    return prompt


def _score_with_evaluator(
    *,
    eval_mode: str,
    eval_model: str,
    benchmark_prompt_template: str,
    question_text: str,
    ground_truth: Any,
    replay_answer: str,
    max_score: float,
    evidence: Any,
    evaluator_cache: dict[tuple[str, str, str], Any],
) -> tuple[dict[str, Any], str]:
    prompt_template = benchmark_prompt_template if benchmark_prompt_template else ""
    cache_key = (eval_mode, eval_model, prompt_template)
    evaluator = evaluator_cache.get(cache_key)
    if evaluator is None:
        kwargs: dict[str, Any] = {"model": eval_model}
        if prompt_template:
            kwargs["prompt_template"] = prompt_template
        evaluator = get_evaluator(eval_mode, **kwargs)
        evaluator_cache[cache_key] = evaluator

    ts = evaluator.evaluate(
        question_text=question_text,
        ground_truth=ground_truth,
        response=replay_answer,
        max_score=max_score,
        evidence=evidence,
    )
    score = {
        "score": float(ts.score),
        "passed": bool(ts.passed),
        "detail": str(ts.detail),
        "metadata": dict(ts.metadata) if isinstance(ts.metadata, dict) else {},
    }
    return score, f"evaluator:{eval_mode}"


def _with_replay_system_prompt(
    original_messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    if not system_prompt.strip():
        return [dict(m) for m in original_messages if isinstance(m, dict)]

    out: list[dict[str, Any]] = [dict(m) for m in original_messages if isinstance(m, dict)]
    if out and out[0].get("role") == "system":
        out[0]["content"] = system_prompt
        return out
    return [{"role": "system", "content": system_prompt}] + out


def _safe_model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-").lower()


def _default_output_path(input_json: Path, model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_json.with_name(
        f"{input_json.stem}_replay_{_safe_model_slug(model)}_{stamp}.json"
    )


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    parent[key] = {}
    return parent[key]


def _to_benchmark_result_model_dict(compat: dict[str, Any]) -> dict[str, Any]:
    scenario_results: list[dict[str, Any]] = []
    for sc in compat.get("scenarios", []):
        turns: list[dict[str, Any]] = []
        for t in sc.get("turns", []):
            turns.append(
                {
                    "turn_index": t.get("index", 0),
                    "user_input": t.get("user_input", ""),
                    "response": t.get("response", ""),
                    "turn_type": t.get("type", "conversation"),
                    "score": t.get("score"),
                    "metadata": t.get("metadata", {}) or {},
                    "message_trace": t.get("message_trace"),
                    "depends_on_turn_indices": t.get("depends_on_turn_indices", []) or [],
                    "dependency_policy": t.get("dependency_policy", "") or "",
                }
            )
        scenario_results.append(
            {
                "scenario_id": sc.get("id", ""),
                "scenario_description": sc.get("description", ""),
                "turn_results": turns,
                "scenario_score": sc.get("scenario_score"),
                "metadata": sc.get("metadata", {}) or {},
                "preload_history": sc.get("preload_history", []) or [],
                "memory_library_id": sc.get("memory_library_id", "") or "",
            }
        )
    return {
        "benchmark_name": compat.get("benchmark_name", ""),
        "agent_identifier": compat.get("agent_identifier", ""),
        "timestamp": compat.get("timestamp", ""),
        "aggregate": compat.get("aggregate", {}),
        "metadata": compat.get("metadata", {}) or {},
        "run_config": compat.get("run_config"),
        "scenario_results": scenario_results,
    }


def _validate_output_schema(payload: dict[str, Any]) -> None:
    required_top = {
        "benchmark_name",
        "agent_identifier",
        "timestamp",
        "aggregate",
        "metadata",
        "scenarios",
    }
    missing = sorted(required_top - set(payload.keys()))
    if missing:
        raise ValueError(f"输出缺少顶层字段: {missing}")
    model_dict = _to_benchmark_result_model_dict(payload)
    BenchmarkResult.model_validate(model_dict)


def _print_summary_table(
    *,
    input_json: Path,
    output_json: Path,
    model: str,
    summary: dict[str, Any],
    aggregate_replay: dict[str, Any],
) -> None:
    original_total = int(aggregate_replay.get("original_total", 0) or 0)
    original_passed = int(aggregate_replay.get("original_passed", 0) or 0)
    original_acc = float(aggregate_replay.get("original_score_percent", 0.0) or 0.0)
    replay_scored = int(aggregate_replay.get("scored_total", 0) or 0)
    replay_passed = int(aggregate_replay.get("passed", 0) or 0)
    replay_acc = float(aggregate_replay.get("score_percent", 0.0) or 0.0)

    table = Table(title="Replay Summary", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Input", input_json.as_posix())
    table.add_row("Output", output_json.as_posix())
    table.add_row("Model", model)
    table.add_row("Total Evaluation Turns", str(summary.get("total_evaluation_turns", 0)))
    table.add_row("Replay Attempted", str(summary.get("replay_attempted", 0)))
    table.add_row("Replay Scored", str(summary.get("replay_scored", 0)))
    table.add_row("Replay Passed", str(summary.get("replay_passed", 0)))
    table.add_row("Replay Unscored", str(summary.get("replay_unscored", 0)))
    table.add_row("Replay API Errors", str(summary.get("replay_api_errors", 0)))
    table.add_row(
        "Accuracy (Original -> Replay)",
        f"{original_passed}/{original_total} ({original_acc:.2f}%) -> "
        f"{replay_passed}/{replay_scored} ({replay_acc:.2f}%)",
    )
    if original_total > 0 and replay_scored > 0:
        table.add_row("Accuracy Delta", f"{(replay_acc - original_acc):+.2f}%")

    console = get_console()
    if console is not None:
        console.print(table)
    else:
        print(f"Replay Summary | input={input_json.as_posix()} output={output_json.as_posix()}")
        print(f"Model={model}")
        print(
            f"Original={original_passed}/{original_total} ({original_acc:.2f}%), "
            f"Replay={replay_passed}/{replay_scored} ({replay_acc:.2f}%), "
            f"Delta={(replay_acc - original_acc):+.2f}%"
        )
        print(
            "Counts: "
            f"total_eval={summary.get('total_evaluation_turns', 0)}, "
            f"attempted={summary.get('replay_attempted', 0)}, "
            f"scored={summary.get('replay_scored', 0)}, "
            f"passed={summary.get('replay_passed', 0)}, "
            f"unscored={summary.get('replay_unscored', 0)}, "
            f"api_errors={summary.get('replay_api_errors', 0)}"
        )


def _run_pipeline(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    input_json = Path(args.input_json)
    if not input_json.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_json}")
    output_json = Path(args.output) if args.output else _default_output_path(input_json, args.model)

    original = json.loads(input_json.read_text(encoding="utf-8"))
    replayed = copy.deepcopy(original)
    benchmark_prompt_template = _try_recover_benchmark_prompt(original)
    if benchmark_prompt_template:
        logger.info("已恢复 benchmark eval_prompt，将优先用于 evaluator 判分")
    else:
        logger.info("未恢复 benchmark eval_prompt，使用 evaluator 内置 prompt（最佳努力回退）")

    agent_cls = Agent.compose(BufferMemory)
    agent = agent_cls(
        model=args.model,
        system_prompt="",
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    total_eval = 0
    attempted = 0
    api_errors = 0
    scored = 0
    passed = 0
    unscored = 0
    limit_hit = False
    evaluator_cache: dict[tuple[str, str, str], Any] = {}

    pg = get_progress()
    total_turns = sum(
        1
        for scenario in replayed.get("scenarios", [])
        for turn in scenario.get("turns", [])
        if turn.get("type") == "evaluation"
    )
    eval_h = pg.add_task(
        "Replay · 扫描 evaluation 回合",
        total=float(total_turns) if total_turns > 0 else 1.0,
        task_key="replay:scan_eval_turns",
    )
    replay_h = pg.add_task(
        "Replay · 可重放回合执行",
        total=None,
        task_key="replay:run_replay",
    )

    for scenario in replayed.get("scenarios", []):
        for turn in scenario.get("turns", []):
            if turn.get("type") != "evaluation":
                continue
            total_eval += 1
            pg.advance(eval_h, 1)

            if args.limit is not None and attempted >= args.limit:
                limit_hit = True
                continue

            message_trace = turn.get("message_trace", {})
            extra = message_trace.get("extra", {}) if isinstance(message_trace, dict) else {}
            snapshot = extra.get("query_ready_messages_snapshot", [])
            metadata = turn.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                turn["metadata"] = metadata

            replay_info: dict[str, Any] = {
                "model": args.model,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "judged": False,
                "judge_type": "",
                "response": "",
                "score": None,
                "error": "",
            }

            if not isinstance(snapshot, list) or not snapshot:
                replay_info["error"] = "missing_query_ready_messages_snapshot"
                metadata["replay"] = replay_info
                logger.debug("跳过回放：缺少 query_ready_messages_snapshot")
                continue

            attempted += 1
            pg.update(
                replay_h,
                description=(
                    f"Replay · attempted={attempted} scored={scored} "
                    f"passed={passed} errors={api_errors}"
                ),
            )
            messages = _with_replay_system_prompt(snapshot, args.system_prompt)
            try:
                replay_answer = str(agent.generate(messages))
                eval_mode = str(metadata.get("eval_mode", "binary") or "binary")
                question_text = str(metadata.get("question_text", turn.get("user_input", "")))
                ground_truth = metadata.get("ground_truth", "")
                max_score = metadata.get("max_score", 1.0)
                if isinstance(max_score, (int, float)):
                    max_score_f = float(max_score)
                else:
                    max_score_f = 1.0
                evidence = metadata.get("evidence")

                replay_score, judge_type = _score_with_evaluator(
                    eval_mode=eval_mode,
                    eval_model=args.eval_model,
                    benchmark_prompt_template=benchmark_prompt_template,
                    question_text=question_text,
                    ground_truth=ground_truth,
                    replay_answer=replay_answer,
                    max_score=max_score_f,
                    evidence=evidence,
                    evaluator_cache=evaluator_cache,
                )
                replay_info["response"] = replay_answer
                replay_info["judge_type"] = judge_type
                replay_info["score"] = replay_score
                replay_info["judged"] = True
                scored += 1
                if bool(replay_score.get("passed")):
                    passed += 1
            except Exception as exc:  # noqa: BLE001
                api_errors += 1
                unscored += 1
                replay_info["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning(f"回放异常: {type(exc).__name__}: {exc}")

            metadata["replay"] = replay_info
            pg.update(
                replay_h,
                description=(
                    f"Replay · attempted={attempted} scored={scored} "
                    f"passed={passed} errors={api_errors}"
                ),
            )

    if total_turns == 0:
        pg.advance(eval_h, 1)
    pg.remove_task(eval_h)
    pg.remove_task(replay_h)

    metadata_top = _ensure_dict(replayed, "metadata")
    replay_meta = {
        "is_replay_result": True,
        "source_result_file": str(input_json.as_posix()),
        "replay_model": args.model,
        "eval_model": args.eval_model,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_evaluation_turns": total_eval,
            "replay_attempted": attempted,
            "replay_scored": scored,
            "replay_passed": passed,
            "replay_unscored": unscored,
            "replay_api_errors": api_errors,
            "limit_hit": limit_hit,
            "limit": args.limit,
        },
    }
    metadata_top["replay"] = replay_meta

    aggregate = _ensure_dict(replayed, "aggregate")
    agg_extra = _ensure_dict(aggregate, "extra")
    original_total = int(aggregate.get("total", 0) or 0)
    original_passed = int(aggregate.get("passed", 0) or 0)
    replay_acc = (passed / scored) if scored > 0 else 0.0
    agg_extra["replay"] = {
        "scored_total": scored,
        "passed": passed,
        "score": replay_acc,
        "score_percent": replay_acc * 100.0,
        "attempted": attempted,
        "api_errors": api_errors,
        "original_passed": original_passed,
        "original_total": original_total,
        "original_score_percent": (original_passed / original_total * 100.0) if original_total else 0.0,
    }

    _validate_output_schema(replayed)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(replayed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "重放完成: "
        f"attempted={attempted}, scored={scored}, passed={passed}, "
        f"unscored={unscored}, api_errors={api_errors}, "
        f"output={output_json.as_posix()}"
    )
    return output_json, replay_meta["summary"], agg_extra["replay"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a benchmark result JSON using query_ready_messages_snapshot"
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="历史 benchmark 结果 JSON 文件（兼容导出格式，含 scenarios）",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="重放模型名（例如 openrouter/google/gemini-3.1-pro-preview）",
    )
    parser.add_argument(
        "--eval-model",
        default="openrouter/google/gemini-2.5-flash",
        help="重放结果评估模型（默认: openrouter/google/gemini-2.5-flash）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径（默认自动在输入同目录生成）",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="重放使用的系统提示词",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度（默认 0.0）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="LLM 调用最大重试次数（默认 1）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多重放前 N 个可重放评测回合（默认全部）",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用终端 Rich 多任务进度条（适合 CI / 重定向输出）",
    )
    args = parser.parse_args()

    tty = sys.stderr.isatty()
    use_live = tty and not args.no_progress
    if tty:
        logger.remove()
        with progress_context(live=use_live, force_console=True):
            logger.add(
                loguru_sink_message,
                format="{message}",
                level="INFO",
                colorize=False,
            )
            try:
                output_json, summary, aggregate_replay = _run_pipeline(args)
                _print_summary_table(
                    input_json=Path(args.input_json),
                    output_json=output_json,
                    model=args.model,
                    summary=summary,
                    aggregate_replay=aggregate_replay,
                )
            finally:
                logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level:<8}</level> | <level>{message}</level>"
            ),
            colorize=sys.stderr.isatty(),
        )
    else:
        output_json, summary, aggregate_replay = _run_pipeline(args)
        _print_summary_table(
            input_json=Path(args.input_json),
            output_json=output_json,
            model=args.model,
            summary=summary,
            aggregate_replay=aggregate_replay,
        )


if __name__ == "__main__":
    main()
