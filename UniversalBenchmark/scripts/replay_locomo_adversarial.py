from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import Agent
from agent.memory import BufferMemory

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. The following question may contain traps. "
    "Carefully distinguish whether it is answerable from the provided context. "
    "If it is answerable, answer precisely; if it is not answerable, refuse and say it is not answerable."
)


@dataclass
class ReplayItem:
    question_id: str
    scenario_id: str
    turn_index: int
    question_text: str
    question_type: str
    ground_truth: str
    original_answer: str
    original_passed: bool
    replay_answer: str
    replay_passed: bool
    replay_error: str | None
    selected_option_index: int | None
    selected_option_text: str | None
    wrong_reason: str | None


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[\"'`.,!?;:()\[\]{}]", "", lowered)
    return lowered


def _is_dont_know(answer: str) -> bool:
    ans = answer.lower()
    patterns = (
        "不知道",
        "没有提到",
        "not mentioned",
        "not answerable",
        "need more information",
        "i need more",
        "无法确定",
        "信息不足",
    )
    return any(p in ans for p in patterns)


def _extract_options(question_text: str) -> dict[int, str]:
    options: dict[int, str] = {}
    for line in question_text.splitlines():
        stripped = line.strip()
        match = re.match(r"^([0-9])\.\s*(.+?)\s*$", stripped)
        if match:
            options[int(match.group(1))] = match.group(2).strip()
    return options


def _parse_selected_option(answer: str, options: dict[int, str]) -> tuple[int | None, str | None]:
    if not answer:
        return None, None

    first_line = answer.strip().splitlines()[0].strip()
    idx_match = re.match(r"^\s*([0-9])(?:\s*[\.\):]|$)", first_line)
    if idx_match:
        idx = int(idx_match.group(1))
        return idx, options.get(idx)

    normalized_answer = _normalize_text(answer)
    for idx, text in options.items():
        if _normalize_text(text) == normalized_answer:
            return idx, text
    for idx, text in options.items():
        if _normalize_text(text) in normalized_answer:
            return idx, text
    return None, None


def _judge_mc_pass(answer: str, ground_truth: str, question_text: str) -> tuple[bool, int | None, str | None]:
    options = _extract_options(question_text)
    selected_idx, selected_text = _parse_selected_option(answer, options)

    gt_norm = _normalize_text(ground_truth)
    if selected_text is not None:
        return _normalize_text(selected_text) == gt_norm, selected_idx, selected_text

    ans_norm = _normalize_text(answer)
    return ans_norm == gt_norm, selected_idx, selected_text


def _collect_adversarial_turns(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in data.get("scenarios", []):
        scenario_id = str(scenario.get("scenario_id", ""))
        for turn in scenario.get("turns", []):
            if turn.get("type") != "evaluation":
                continue
            metadata = turn.get("metadata", {})
            payload = metadata.get("evidence", {}).get("payload", {})
            if payload.get("question_type") != "adversarial":
                continue
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "turn": turn,
                }
            )
    return rows


def _build_report(
    output_md: Path,
    model: str,
    system_prompt: str,
    source_json: Path,
    result_json: Path,
    items: list[ReplayItem],
    original_passed: int,
) -> None:
    total = len(items)
    replay_passed = sum(1 for i in items if i.replay_passed)
    replay_wrong = total - replay_passed
    original_acc = (original_passed / total * 100) if total else 0.0
    replay_acc = (replay_passed / total * 100) if total else 0.0
    delta = replay_acc - original_acc

    dont_know_wrong = sum(1 for i in items if (not i.replay_passed and i.wrong_reason == "dont_know"))
    answered_wrong = sum(1 for i in items if (not i.replay_passed and i.wrong_reason == "answered_wrong"))
    errors = sum(1 for i in items if i.replay_error)

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Adversarial 重放报告\n\n")
        f.write(f"- 结果来源: `{source_json.as_posix()}`\n")
        f.write(f"- 模型: `{model}`\n")
        f.write(f"- System Prompt: `{system_prompt}`\n")
        f.write(f"- 明细文件: `{result_json.as_posix()}`\n")
        f.write(f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
        f.write("## 总体指标\n\n")
        f.write(f"- Adversarial 总题数: **{total}**\n")
        f.write(f"- 原始通过数: **{original_passed}** / {total} ({original_acc:.2f}%)\n")
        f.write(f"- 重放通过数: **{replay_passed}** / {total} ({replay_acc:.2f}%)\n")
        f.write(f"- 准确率变化: **{delta:+.2f}%**\n")
        f.write(f"- 重放错题: **{replay_wrong}**\n")
        f.write(f"- 其中“模型说不知道”: **{dont_know_wrong}**\n")
        f.write(f"- 其中“模型回答错误”: **{answered_wrong}**\n")
        f.write(f"- API/运行异常题数: **{errors}**\n\n")

        f.write("## 逐题明细\n\n")
        f.write("| # | question_id | original_passed | replay_passed | wrong_reason | replay_answer |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, item in enumerate(items, start=1):
            answer_preview = item.replay_answer.replace("\n", " ").strip()
            if len(answer_preview) > 120:
                answer_preview = answer_preview[:117] + "..."
            reason = item.wrong_reason or "-"
            f.write(
                f"| {i} | {item.question_id} | {item.original_passed} | {item.replay_passed} | {reason} | {answer_preview} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay adversarial LoCoMo questions with BufferMemory agent")
    parser.add_argument(
        "--input-json",
        default="outputs/locomo/memecho_conv-30_full_1.json",
        help="原始 benchmark 结果 JSON 路径",
    )
    parser.add_argument(
        "--model",
        default="openrouter/google/gemini-3-flash-preview",
        help="重放使用的模型",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/locomo/replay_adversarial_gemini3flashpreview",
        help="输出目录",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度（默认 0.0）",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="重放时使用的系统提示词",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="LLM 调用最大重试次数（默认 1，加速重放）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅重放前 N 题（默认全部）",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "replay_adversarial_results.json"
    output_md = output_dir / "replay_adversarial_report.md"

    data = json.loads(input_json.read_text(encoding="utf-8"))
    rows = _collect_adversarial_turns(data)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    agent_cls = Agent.compose(BufferMemory)
    agent = agent_cls(
        model=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    items: list[ReplayItem] = []
    original_passed = 0
    for row in rows:
        turn = row["turn"]
        scenario_id = row["scenario_id"]
        metadata = turn.get("metadata", {})
        payload = metadata.get("evidence", {}).get("payload", {})
        question_text = str(metadata.get("question_text") or turn.get("user_input") or "")
        question_id = str(payload.get("dataset_question_id") or metadata.get("question_id") or "")
        score_info = turn.get("score", {})
        original_ok = bool(score_info.get("passed", False))
        if original_ok:
            original_passed += 1

        replay_answer = ""
        replay_error: str | None = None
        replay_ok = False
        sel_idx: int | None = None
        sel_text: str | None = None

        message_trace = turn.get("message_trace", {})
        extra = message_trace.get("extra", {}) if isinstance(message_trace, dict) else {}
        messages = extra.get("query_ready_messages_snapshot", [])
        if not isinstance(messages, list) or not messages:
            replay_error = "missing_query_ready_messages_snapshot"
        else:
            try:
                replay_answer = str(agent.generate(messages))
                replay_ok, sel_idx, sel_text = _judge_mc_pass(
                    replay_answer,
                    str(metadata.get("ground_truth", "")),
                    question_text,
                )
            except Exception as exc:  # noqa: BLE001
                replay_error = f"{type(exc).__name__}: {exc}"

        wrong_reason: str | None = None
        if not replay_ok:
            wrong_reason = "dont_know" if _is_dont_know(replay_answer) else "answered_wrong"
            if replay_error:
                wrong_reason = "runtime_error"

        items.append(
            ReplayItem(
                question_id=question_id,
                scenario_id=scenario_id,
                turn_index=int(turn.get("index", -1)),
                question_text=question_text,
                question_type="adversarial",
                ground_truth=str(metadata.get("ground_truth", "")),
                original_answer=str(turn.get("response", "")),
                original_passed=original_ok,
                replay_answer=replay_answer,
                replay_passed=replay_ok,
                replay_error=replay_error,
                selected_option_index=sel_idx,
                selected_option_text=sel_text,
                wrong_reason=wrong_reason,
            )
        )
        print(
            f"[replay] {len(items)}/{len(rows)} "
            f"{question_id} passed={replay_ok} err={bool(replay_error)}",
            flush=True,
        )

    output_payload = {
        "meta": {
            "input_json": str(input_json.as_posix()),
            "output_dir": str(output_dir.as_posix()),
            "model": args.model,
            "temperature": args.temperature,
            "system_prompt": args.system_prompt,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "total_adversarial": len(items),
            "original_passed": original_passed,
            "replay_passed": sum(1 for i in items if i.replay_passed),
            "replay_wrong": sum(1 for i in items if not i.replay_passed),
            "replay_dont_know_wrong": sum(1 for i in items if i.wrong_reason == "dont_know"),
            "replay_answered_wrong": sum(1 for i in items if i.wrong_reason == "answered_wrong"),
            "replay_runtime_error": sum(1 for i in items if i.wrong_reason == "runtime_error"),
        },
        "items": [item.__dict__ for item in items],
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _build_report(
        output_md,
        args.model,
        args.system_prompt,
        input_json,
        output_json,
        items,
        original_passed,
    )

    print(f"Replay complete. JSON: {output_json.as_posix()}")
    print(f"Replay complete. MD:   {output_md.as_posix()}")


if __name__ == "__main__":
    main()
