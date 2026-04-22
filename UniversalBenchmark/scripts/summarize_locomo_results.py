"""
按文件名正则筛选 LoCoMo / BenchmarkResult JSON，按题目数加权汇总准确率。

默认优先使用 aggregate.extra.replay（重放评测）；缺失时回退 aggregate（原始跑分）。
使用 --no-prefer-replay 仅使用 aggregate。

示例（在 UniversalBenchmark 目录下）::

    uv run python scripts/summarize_locomo_results.py outputs/locomo
    uv run python scripts/summarize_locomo_results.py outputs/locomo --no-prefer-replay
    uv run python scripts/summarize_locomo_results.py outputs/locomo --name-regex '.*conv-4[0-3].*replay.*\\.json$'

``--json-out`` 写入的汇总包含 ``schema_version``、各文件的 ``dataset_name`` / ``memory_type`` /
``agent_model`` 元数据，以及 ``dimensions`` / ``by_group``（供 ``scoreboard/`` 前端展示）。
其中当文件口径为 ``replay`` 时，``agent_model`` 优先为 **重放评测模型**
（``metadata.replay.replay_model`` / ``aggregate.extra.replay_model`` 等），否则为 ``run_config.model``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SourceKind = Literal["replay", "fallback_normal", "normal"]

_SOURCE_STYLE: dict[SourceKind, str] = {
    "replay": "bold green",
    "fallback_normal": "yellow",
    "normal": "cyan",
}


@dataclass
class FileStat:
    path: str
    basename: str
    passed: int
    total: int
    accuracy_percent: float
    source: SourceKind
    # Scoreboard / 多维展示：来自 BenchmarkResult JSON
    dataset_name: str = "unknown"
    memory_type: str = "unknown"
    agent_model: str = "unknown"
    agent_identifier: str = ""


_AGENT_IDENTIFIER_RE = re.compile(
    r"^Agent\[(?P<model>[^|]+)\|(?P<memory>[^\]]+)\]\s*$",
)


def _memory_slug_from_class_name(mem: str) -> str:
    mem = mem.strip()
    if mem.endswith("Memory") and len(mem) > len("Memory"):
        return mem[: -len("Memory")].lower()
    return mem.lower()


def _extract_replay_model_string(data: dict[str, Any]) -> str | None:
    """从重放元数据读取用于打分的模型名（若存在）。"""
    meta = data.get("metadata")
    if isinstance(meta, dict):
        rep = meta.get("replay")
        if isinstance(rep, dict):
            for key in ("replay_model", "model"):
                m = rep.get(key)
                if isinstance(m, str) and m.strip():
                    return m.strip()
    aggregate = data.get("aggregate")
    if isinstance(aggregate, dict):
        extra = aggregate.get("extra")
        if isinstance(extra, dict):
            m = extra.get("replay_model")
            if isinstance(m, str) and m.strip():
                return m.strip()
            replay = extra.get("replay")
            if isinstance(replay, dict):
                for key in ("replay_model", "model"):
                    m2 = replay.get(key)
                    if isinstance(m2, str) and m2.strip():
                        return m2.strip()
    return None


def extract_run_metadata(
    data: dict[str, Any],
    *,
    source: SourceKind,
) -> tuple[str, str, str, str]:
    """从单份 BenchmarkResult JSON 提取 dataset / memory / agent 元信息。

    当 ``source == "replay"`` 时，``agent_model`` 优先为重放评测所用模型
    （``metadata.replay.replay_model`` / ``aggregate.extra.replay_model`` 等），
    否则与原始跑分一致：优先 ``run_config.model``。

    返回 ``(dataset_name, memory_type, agent_model, agent_identifier)``。
    """
    bn = data.get("benchmark_name")
    dataset_name = (
        bn.strip()
        if isinstance(bn, str) and bn.strip()
        else "unknown"
    )

    aid = data.get("agent_identifier")
    agent_identifier = aid.strip() if isinstance(aid, str) else ""

    memory_type = "unknown"
    agent_model = "unknown"

    if source == "replay":
        rm = _extract_replay_model_string(data)
        if rm:
            agent_model = rm

    rc = data.get("run_config")
    if isinstance(rc, dict):
        if agent_model == "unknown":
            m = rc.get("model")
            if isinstance(m, str) and m.strip():
                agent_model = m.strip()
        mem = rc.get("memory_type")
        if isinstance(mem, str) and mem.strip():
            memory_type = mem.strip()

    if agent_identifier:
        m2 = _AGENT_IDENTIFIER_RE.match(agent_identifier)
        if m2:
            if agent_model == "unknown":
                agent_model = m2.group("model").strip()
            if memory_type == "unknown":
                memory_type = _memory_slug_from_class_name(m2.group("memory"))

    return dataset_name, memory_type, agent_model, agent_identifier


def build_dimensions(stats: list[FileStat]) -> dict[str, list[str]]:
    """各维度去重后的候选值（供 scoreboard 筛选）。"""
    ds = sorted({s.dataset_name for s in stats})
    mem = sorted({s.memory_type for s in stats})
    ag = sorted({s.agent_model for s in stats})
    return {"datasets": ds, "memories": mem, "agents": ag}


def build_by_group(stats: list[FileStat]) -> list[dict[str, Any]]:
    """按 dataset + memory + agent_model 聚合通过数与总题数。"""
    acc: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for s in stats:
        key = (s.dataset_name, s.memory_type, s.agent_model)
        acc[key][0] += s.passed
        acc[key][1] += s.total
    rows: list[dict[str, Any]] = []
    for (dataset_name, memory_type, agent_model), (p, t) in sorted(acc.items()):
        pct = (100.0 * float(p) / float(t)) if t > 0 else 0.0
        rows.append(
            {
                "dataset_name": dataset_name,
                "memory_type": memory_type,
                "agent_model": agent_model,
                "passed": p,
                "total": t,
                "accuracy_percent": pct,
            }
        )
    return rows


@dataclass
class CategoryStat:
    """按 question_type 聚合（跨所选文件的逐题统计）。"""

    question_type: str
    passed: int
    total: int
    accuracy_percent: float


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def extract_passed_total(
    data: dict[str, Any], *, prefer_replay: bool
) -> tuple[int | None, int | None, SourceKind | None]:
    """
    返回 (passed, total, source)。若无法读取则返回 (None, None, None)。
    """
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        return None, None, None

    if prefer_replay:
        extra = aggregate.get("extra")
        if isinstance(extra, dict):
            replay = extra.get("replay")
            if isinstance(replay, dict):
                st = _safe_int(replay.get("scored_total"), -1)
                pa = _safe_int(replay.get("passed"), -1)
                if st > 0 and pa >= 0:
                    return pa, st, "replay"
        # 回退普通 aggregate
        t = _safe_int(aggregate.get("total"), -1)
        p = _safe_int(aggregate.get("passed"), -1)
        if t > 0 and p >= 0:
            return p, t, "fallback_normal"
        return None, None, None

    t = _safe_int(aggregate.get("total"), -1)
    p = _safe_int(aggregate.get("passed"), -1)
    if t > 0 and p >= 0:
        return p, t, "normal"
    return None, None, None


def evaluation_turn_passed(turn: dict[str, Any], *, prefer_replay: bool) -> bool | None:
    """
    单题是否通过。无法从 turn 读取分数时返回 None。
    prefer_replay：优先 metadata.replay.score.passed，否则 turn.score.passed。
    """
    meta = turn.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    if prefer_replay:
        replay = meta.get("replay")
        if isinstance(replay, dict):
            rscore = replay.get("score")
            if isinstance(rscore, dict) and "passed" in rscore:
                return bool(rscore.get("passed"))
    score_info = turn.get("score")
    if isinstance(score_info, dict) and "passed" in score_info:
        return bool(score_info.get("passed"))
    return None


def question_type_from_turn(turn: dict[str, Any]) -> str:
    meta = turn.get("metadata")
    if not isinstance(meta, dict):
        return "unknown"
    ev = meta.get("evidence")
    if not isinstance(ev, dict):
        return "unknown"
    payload = ev.get("payload")
    if not isinstance(payload, dict):
        return "unknown"
    qt = payload.get("question_type")
    if isinstance(qt, str) and qt.strip():
        return qt.strip()
    return "unknown"


def aggregate_question_types(
    data: dict[str, Any], *, prefer_replay: bool
) -> tuple[dict[str, tuple[int, int]], int]:
    """
    返回 (各题型 (passed, total), 无法判分的 evaluation 条数)。
    """
    counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    bad = 0

    for sc in data.get("scenarios", []):
        if not isinstance(sc, dict):
            continue
        for turn in sc.get("turns", []):
            if not isinstance(turn, dict):
                continue
            if turn.get("type") != "evaluation":
                continue
            qt = question_type_from_turn(turn)
            p = evaluation_turn_passed(turn, prefer_replay=prefer_replay)
            if p is None:
                bad += 1
                continue
            passed_i, tot_i = counts[qt]
            counts[qt] = (passed_i + (1 if p else 0), tot_i + 1)

    return dict(counts), bad


def sort_question_type_keys(keys: list[str]) -> list[str]:
    rest = sorted(k for k in keys if k != "unknown")
    if "unknown" in keys:
        rest.append("unknown")
    return rest


def category_stats_from_merged(
    merged: dict[str, tuple[int, int]],
) -> list[CategoryStat]:
    out: list[CategoryStat] = []
    for qt in sort_question_type_keys(list(merged.keys())):
        passed_i, tot_i = merged[qt]
        acc = (100.0 * float(passed_i) / float(tot_i)) if tot_i > 0 else 0.0
        out.append(
            CategoryStat(
                question_type=qt,
                passed=passed_i,
                total=tot_i,
                accuracy_percent=acc,
            )
        )
    return out


def iter_json_files(root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.json"))
    return sorted(root.glob("*.json"))


def run_summary(
    input_dir: Path,
    *,
    name_pattern: re.Pattern[str],
    prefer_replay: bool,
    recursive: bool,
) -> tuple[list[FileStat], dict[str, Any], dict[str, tuple[int, int]]]:
    """返回各文件统计、跳过原因、按题型合并的 (passed, total)。"""
    skipped: dict[str, int] = {
        "not_file": 0,
        "name_no_match": 0,
        "json_error": 0,
        "not_object": 0,
        "no_stats": 0,
        "zero_total": 0,
        "turn_no_pass_field": 0,
    }
    stats: list[FileStat] = []
    merged_qt: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    for path in iter_json_files(input_dir, recursive=recursive):
        if not path.is_file():
            skipped["not_file"] += 1
            continue
        if not name_pattern.search(path.name):
            skipped["name_no_match"] += 1
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            skipped["json_error"] += 1
            continue
        if not isinstance(data, dict):
            skipped["not_object"] += 1
            continue

        passed, total, source = extract_passed_total(data, prefer_replay=prefer_replay)
        if passed is None or total is None or source is None:
            skipped["no_stats"] += 1
            continue
        if total == 0:
            skipped["zero_total"] += 1
            continue

        acc = 100.0 * float(passed) / float(total)
        ds_name, mem_t, ag_model, ag_id = extract_run_metadata(data, source=source)
        stats.append(
            FileStat(
                path=str(path.as_posix()),
                basename=path.name,
                passed=passed,
                total=total,
                accuracy_percent=acc,
                source=source,
                dataset_name=ds_name,
                memory_type=mem_t,
                agent_model=ag_model,
                agent_identifier=ag_id,
            )
        )

        per_file_qt, bad_turns = aggregate_question_types(data, prefer_replay=prefer_replay)
        skipped["turn_no_pass_field"] += bad_turns
        for qt, (p_i, t_i) in per_file_qt.items():
            op, ot = merged_qt[qt]
            merged_qt[qt] = (op + p_i, ot + t_i)

    return stats, skipped, dict(merged_qt)


def _category_totals(categories: list[CategoryStat]) -> tuple[int, int]:
    return (sum(c.passed for c in categories), sum(c.total for c in categories))


def _print_plain_summary(
    *,
    input_dir: Path,
    name_regex: str,
    prefer_replay: bool,
    recursive: bool,
    stats: list[FileStat],
    skipped: dict[str, Any],
    total_passed: int,
    total_questions: int,
    weighted_acc: float,
    categories: list[CategoryStat],
) -> None:
    print(f"目录: {input_dir}")
    print(f"name-regex: {name_regex!r}")
    print(f"prefer-replay: {prefer_replay}")
    print(f"recursive: {recursive}")
    print(f"有效文件数: {len(stats)}")
    if any(skipped.values()):
        print(f"跳过统计: {skipped}")
    print()
    hdr = (
        f"{'basename':<40} {'dataset':<18} {'mem':<10} {'agent':<22} "
        f"{'passed':>7} {'total':>7} {'acc%':>9} {'source':<18}"
    )
    print(hdr)
    print("-" * min(140, len(hdr) + 20))
    for s in stats:
        ag_short = (s.agent_model[:20] + "…") if len(s.agent_model) > 21 else s.agent_model
        ds_short = (s.dataset_name[:16] + "…") if len(s.dataset_name) > 17 else s.dataset_name
        bn_short = (s.basename[:38] + "…") if len(s.basename) > 39 else s.basename
        print(
            f"{bn_short:<40} {ds_short:<18} {s.memory_type:<10} {ag_short:<22} "
            f"{s.passed:>7} {s.total:>7} {s.accuracy_percent:>8.4f}% {s.source:<18}"
        )
    print("-" * min(140, len(hdr) + 20))
    print(
        f"{'WEIGHTED':<40} {'':<18} {'':<10} {'':<22} "
        f"{total_passed:>7} {total_questions:>7} {weighted_acc:>8.4f}%"
    )

    cat_p, cat_t = _category_totals(categories)
    print()
    print("分题型汇总（逐题，跨文件）")
    print(f"{'question_type':<22} {'passed':>7} {'total':>7} {'acc%':>10}")
    print("-" * 52)
    for c in categories:
        print(f"{c.question_type:<22} {c.passed:>7} {c.total:>7} {c.accuracy_percent:>9.4f}%")
    print("-" * 52)
    cat_acc = (100.0 * cat_p / cat_t) if cat_t else 0.0
    print(f"{'CATEGORY_TOTAL':<22} {cat_p:>7} {cat_t:>7} {cat_acc:>9.4f}%")
    if skipped.get("turn_no_pass_field") or cat_t != total_questions:
        print(
            "[提示] 逐题合计题数与文件级加权题数不一致时，"
            f"可能含无法读取 passed 的 evaluation 条（未判分={skipped.get('turn_no_pass_field', 0)}）。"
        )


def _print_rich_summary(
    *,
    console: Console,
    input_dir: Path,
    name_regex: str,
    prefer_replay: bool,
    recursive: bool,
    stats: list[FileStat],
    skipped: dict[str, Any],
    total_passed: int,
    total_questions: int,
    weighted_acc: float,
    categories: list[CategoryStat],
) -> None:
    cfg = Table.grid(padding=(0, 2))
    cfg.add_column(style="cyan", justify="right")
    cfg.add_column(style="white")
    cfg.add_row("目录", str(input_dir))
    cfg.add_row("name-regex", name_regex)
    cfg.add_row("prefer-replay", "开启" if prefer_replay else "关闭（仅用 aggregate）")
    cfg.add_row("recursive", "是" if recursive else "否")
    cfg.add_row("有效文件", str(len(stats)))

    skip_parts = [f"{k}={v}" for k, v in skipped.items() if v]
    if skip_parts:
        cfg.add_row("跳过", Text(", ".join(skip_parts), style="dim yellow"))

    replay_hint = (
        "aggregate.extra.replay 优先"
        if prefer_replay
        else "仅 aggregate（原始跑分）"
    )
    console.print()
    console.print(
        Panel(
            cfg,
            title="[bold bright_white]Benchmark 结果汇总[/]",
            subtitle=f"[dim]按题目数加权 · {replay_hint}[/]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    table = Table(
        title="[bold]分文件明细[/]",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold bright_cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("文件名", overflow="ellipsis", max_width=36, no_wrap=False)
    table.add_column("数据集", overflow="ellipsis", max_width=18, style="dim cyan")
    table.add_column("Memory", max_width=10, style="magenta")
    table.add_column("Agent 模型", overflow="ellipsis", max_width=26)
    table.add_column("通过", justify="right", style="green")
    table.add_column("题数", justify="right")
    table.add_column("准确率", justify="right", style="bright_white")
    table.add_column("口径", justify="center")

    for s in stats:
        acc_txt = f"{s.accuracy_percent:.4f}%"
        src = Text(s.source, style=_SOURCE_STYLE.get(s.source, "white"))
        table.add_row(
            s.basename,
            s.dataset_name,
            s.memory_type,
            s.agent_model,
            str(s.passed),
            str(s.total),
            acc_txt,
            src,
        )

    table.add_section()
    table.add_row(
        Text("加权合计", style="bold"),
        Text("—", style="dim"),
        Text("—", style="dim"),
        Text("—", style="dim"),
        Text(str(total_passed), style="bold green"),
        Text(str(total_questions), style="bold"),
        Text(f"{weighted_acc:.4f}%", style="bold bright_yellow"),
        Text("—", style="dim"),
    )

    console.print(table)

    summary_grid = Table.grid(padding=(0, 2))
    summary_grid.add_column(justify="center")
    summary_grid.add_row(
        f"[bold bright_green]总通过[/]  [bold]{total_passed}[/]  /  [bold]{total_questions}[/]  题"
    )
    summary_grid.add_row(f"[bold]加权准确率[/]  [bold bright_yellow]{weighted_acc:.4f}%[/]")
    console.print(Panel(summary_grid, border_style="green", title="[bold]汇总[/]"))

    cat_p, cat_t = _category_totals(categories)
    cat_acc = (100.0 * cat_p / cat_t) if cat_t else 0.0
    if skipped.get("turn_no_pass_field") or cat_t != total_questions:
        sub = "[dim]逐题与文件级题数不一致时见下方未判分条数[/]"
    elif prefer_replay:
        sub = "[dim]逐题合计应对齐文件级加权（replay 判分）[/]"
    else:
        sub = "[dim]逐题合计应对齐文件级加权（turn.score 原始跑分）[/]"
    qtable = Table(
        title="[bold]分题型汇总（逐题，跨文件）[/]",
        caption=sub,
        box=box.ROUNDED,
        border_style="magenta",
        header_style="bold bright_magenta",
        padding=(0, 1),
    )
    qtable.add_column("题型", style="bright_white")
    qtable.add_column("通过", justify="right", style="green")
    qtable.add_column("题数", justify="right")
    qtable.add_column("准确率", justify="right", style="bright_yellow")
    for c in categories:
        qtable.add_row(
            c.question_type,
            str(c.passed),
            str(c.total),
            f"{c.accuracy_percent:.4f}%",
        )
    qtable.add_section()
    qtable.add_row(
        Text("逐题合计", style="bold"),
        Text(str(cat_p), style="bold green"),
        Text(str(cat_t), style="bold"),
        Text(f"{cat_acc:.4f}%", style="bold bright_yellow"),
    )
    console.print(qtable)
    if skipped.get("turn_no_pass_field"):
        console.print(
            f"[dim yellow]逐题无法判分（缺 passed）的 evaluation 条数: "
            f"{skipped['turn_no_pass_field']}[/]"
        )
    console.print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按文件名正则筛选结果 JSON，按题目数加权汇总准确率（默认优先 replay）。"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="扫描的目录（例如 outputs/locomo）",
    )
    parser.add_argument(
        "--name-regex",
        type=str,
        default=r".*replay.*\.json$",
        help=r'仅处理 basename 匹配该正则的文件（默认: .*replay.*\.json$）',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归子目录扫描 *.json",
    )
    parser.add_argument(
        "--prefer-replay",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="优先使用 aggregate.extra.replay，缺失则回退 aggregate（默认：开启）",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="将汇总结果写入该 JSON 文件",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="不使用 Rich（纯文本表格，便于重定向/CI）",
    )

    args = parser.parse_args(argv)

    try:
        name_pattern = re.compile(args.name_regex)
    except re.error as e:
        print(f"无效的正则 --name-regex: {e}", file=sys.stderr)
        return 2

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"目录不存在或不是目录: {input_dir}", file=sys.stderr)
        return 2

    stats, skipped, merged_qt = run_summary(
        input_dir,
        name_pattern=name_pattern,
        prefer_replay=bool(args.prefer_replay),
        recursive=bool(args.recursive),
    )

    total_questions = sum(s.total for s in stats)
    total_passed = sum(s.passed for s in stats)
    weighted_acc = (100.0 * float(total_passed) / float(total_questions)) if total_questions > 0 else 0.0
    categories = category_stats_from_merged(merged_qt)

    console = Console(stderr=False)
    if args.plain:
        _print_plain_summary(
            input_dir=input_dir,
            name_regex=args.name_regex,
            prefer_replay=bool(args.prefer_replay),
            recursive=bool(args.recursive),
            stats=stats,
            skipped=skipped,
            total_passed=total_passed,
            total_questions=total_questions,
            weighted_acc=weighted_acc,
            categories=categories,
        )
    else:
        _print_rich_summary(
            console=console,
            input_dir=input_dir,
            name_regex=args.name_regex,
            prefer_replay=bool(args.prefer_replay),
            recursive=bool(args.recursive),
            stats=stats,
            skipped=skipped,
            total_passed=total_passed,
            total_questions=total_questions,
            weighted_acc=weighted_acc,
            categories=categories,
        )

    by_qtype_payload: dict[str, Any] = {
        c.question_type: {
            "passed": c.passed,
            "total": c.total,
            "accuracy_percent": c.accuracy_percent,
        }
        for c in categories
    }

    dimensions = build_dimensions(stats)
    by_group = build_by_group(stats)

    out_payload: dict[str, Any] = {
        "schema_version": 2,
        "input_dir": str(input_dir.as_posix()),
        "name_regex": args.name_regex,
        "prefer_replay": bool(args.prefer_replay),
        "recursive": bool(args.recursive),
        "skipped": skipped,
        "file_count": len(stats),
        "weighted": {
            "passed": total_passed,
            "total": total_questions,
            "accuracy_percent": weighted_acc,
        },
        "by_question_type": by_qtype_payload,
        "dimensions": dimensions,
        "by_group": by_group,
        "per_file": [asdict(s) for s in stats],
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(out_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if args.plain:
            print(f"\n已写入: {args.json_out}")
        else:
            console.print(f"[dim]已写入[/] [cyan]{args.json_out.as_posix()}[/]")

    if not stats:
        print("错误: 没有任何有效样本（检查目录、正则与 JSON 结构）。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
