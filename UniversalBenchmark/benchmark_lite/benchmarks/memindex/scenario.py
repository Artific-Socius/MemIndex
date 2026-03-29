"""MemIndex InteractiveScenario 实现。

将 MemIndex 的多序列交替调度 + 记忆距离管理 + 废话填充逻辑，
封装为 benchmark_lite 的 InteractiveScenario 接口。

核心调度逻辑（与原 MemIndex Runner 保持一致）：

1. **HEAD 阶段** — 发送开场提示
2. **RUN 阶段** — 交替执行多个序列的步骤，中间插入废话填充
   - mark_queue（优先级最高，解冻后的序列）
   - queue（常规队列）
   - nonsense（当两个队列均为空但冻结区非空时，插入废话填充记忆距离）
3. **DONE** — 所有序列执行完毕，返回 None

评估在 ``next_turn`` 中以行内方式完成（因为依赖检查和重试都需要
即时知道评分结果），最终在 ``evaluate`` 中汇总为 ``ScenarioScore``。
"""

from __future__ import annotations

import json
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import tiktoken
from loguru import logger

from agent.progress import TaskHandle, get_progress

from benchmark_lite.base import InteractiveScenario
from benchmark_lite.types import (
    ScenarioScore,
    Turn,
    TurnAnnotation,
    TurnResult,
    TurnScore,
    TurnType,
)

from .data import (
    DatasetConfig,
    SequenceItem,
    resolve_refs,
)
from .evaluator import (
    evaluate_binary,
    evaluate_multi_score,
    evaluate_score,
    evaluate_weighted_binary,
)


# ── 内部状态类 ─────────────────────────────────────────────────


@dataclass
class _ActuatorState:
    """单个测试序列的运行时状态。"""

    name: str
    items: list[SequenceItem]
    cursor: int = 0
    start_tokens: int = 0
    intermediate: dict[int, SequenceItem] = field(default_factory=dict)
    pending_retry: bool = False

    @property
    def has_next(self) -> bool:
        return self.cursor < len(self.items)


@dataclass
class _PendingAction:
    """记录上一个 Turn 对应的动作，用于在下次 next_turn 中处理结果。"""

    kind: str                  # "head" | "actuator" | "nonsense"
    actuator_idx: int = -1
    list_idx: int = -1         # cursor value when the turn was emitted


@dataclass
class _InlineEval:
    """行内评估记录。"""

    turn_index: int
    score: float
    max_score: float
    reason: str
    actuator_name: str
    eval_method: str = ""


# ── 废话生成 ───────────────────────────────────────────────────

_TRIVIA_CACHE: list[dict[str, str]] | None = None


def _load_trivia(memindex_root: str) -> list[dict[str, str]]:
    """从 MemIndex/data/nonsense.json 加载废话素材。"""
    global _TRIVIA_CACHE
    if _TRIVIA_CACHE is not None:
        return _TRIVIA_CACHE
    path = os.path.join(memindex_root, "data", "nonsense.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data: list[dict[str, str]] = json.load(f).get("Data", [])
        _TRIVIA_CACHE = data
        return _TRIVIA_CACHE
    return []


def _generate_nonsense(
    trivia: list[dict[str, str]],
    target_tokens: int,
    encoder: tiktoken.Encoding,
) -> str:
    """生成指定 token 数量的废话填充。"""
    if not trivia:
        return random.choice(_FALLBACK_FILLERS)

    header = (
        "Here are some trivia questions and answers for you to process. "
        'Please extract all of the answers in json form as a single '
        'message: E.g ["answer 1", "answer 2", ...]\n'
    )
    parts = [header]
    total = len(encoder.encode(header))

    while total < target_tokens:
        t = random.choice(trivia)
        line = f"Q: {t['Question']}, A: {t['AnswerValue']}\n"
        total += len(encoder.encode(line))
        parts.append(line)

    return "".join(parts)


_FALLBACK_FILLERS = [
    "What's your favorite season and why?",
    "Can you explain how rainbows form?",
    "Tell me something interesting about space.",
    "How does the internet work in simple terms?",
    "What makes the sky blue?",
    "What are some effective study techniques?",
    "How do smartphones work?",
    "What are the benefits of reading books?",
    "Can you explain what machine learning is?",
    "What's the tallest mountain in the world?",
]

_DEP_REF_PATTERN = re.compile(
    r"\{(?P<answer>[0-9]+)\}"
    r"|\{(?P<question>q:[0-9]+)\}"
)


def _extract_ref_indices(text: str) -> set[int]:
    refs: set[int] = set()
    for m in _DEP_REF_PATTERN.finditer(text):
        answer_ref = m.group("answer")
        question_ref = m.group("question")
        if answer_ref:
            refs.add(int(answer_ref))
        elif question_ref:
            refs.add(int(question_ref[2:]))
    return refs


# ── MemIndexScenario ───────────────────────────────────────────


class MemIndexScenario(InteractiveScenario):
    """MemIndex 测试场景。

    内部实现了与 MemIndex Runner + Actuator 等价的调度逻辑：
    多序列交替调度、记忆距离管理、废话填充、
    依赖检查、重试以及行内 LLM 评估。
    """

    def __init__(
        self,
        dataset: DatasetConfig,
        eval_model: str,
        eval_mode: str = "binary",
        scenario_id: str = "",
        memindex_root: str = "",
    ) -> None:
        self._dataset = dataset
        self._eval_model = eval_model
        self._eval_mode = eval_mode
        self._scenario_id = (
            scenario_id or f"memindex_{dataset.memory_distance}"
        )

        self._encoder = tiktoken.encoding_for_model("gpt-4o-mini")

        self._actuators: list[_ActuatorState] = [
            _ActuatorState(name=name, items=deepcopy(seq.items))
            for name, seq in dataset.data.items()
        ]

        # 调度状态
        self._phase = "head"
        self._head_idx = 0
        self._queue: list[int] = list(range(len(self._actuators)))
        self._frozen: list[tuple[int, int]] = []
        self._mark_queue: list[int] = []

        self._pending: Optional[_PendingAction] = None

        # 行内评估收集
        self._evals: list[_InlineEval] = []
        self._turn_labels: dict[int, str] = {}
        self._emitted_turn_by_act_item: dict[str, dict[int, int]] = {
            a.name: {} for a in self._actuators
        }

        # 废话生成
        self._trivia = (
            _load_trivia(memindex_root) if memindex_root else []
        )
        self._nonsense_tokens = max(20, int(dataset.memory_distance * 0.05))

        # 全局 Rich 进度（与 Runner 共用 Progress；不 remove_task 以保留完成态）
        self._mi_progress_handle: Optional[TaskHandle] = None
        self._mi_total_steps: int = 1

    # ── InteractiveScenario 接口 ──────────────────────────────

    @property
    def id(self) -> str:
        return self._scenario_id

    @property
    def description(self) -> str:
        names = ", ".join(a.name for a in self._actuators)
        return (
            f"MemIndex (distance={self._dataset.memory_distance}, "
            f"sequences=[{names}])"
        )

    def next_turn(self, history: list[TurnResult]) -> Optional[Turn]:
        self._ensure_mi_progress_task()

        if history and self._pending:
            self._handle_result(history)

        turn: Optional[Turn] = None
        if self._phase == "head":
            turn = self._next_head(history)
        elif self._phase == "run":
            turn = self._next_run(history)

        if turn is None:
            self._refresh_mi_progress(finished=True)
        else:
            self._refresh_mi_progress(finished=False)
        return turn

    def _ensure_mi_progress_task(self) -> None:
        if self._mi_progress_handle is not None:
            return
        raw_total = (
            len(self._dataset.head_prompts)
            + sum(len(a.items) for a in self._actuators)
        )
        self._mi_total_steps = max(1, raw_total)
        pg = get_progress()
        self._mi_progress_handle = pg.add_task(
            f"MemIndex · {self._scenario_id}",
            total=float(self._mi_total_steps),
            task_key=f"memindex:scenario:{self._scenario_id}",
        )

    def _mi_completed_steps(self) -> int:
        if self._phase == "head":
            return self._head_idx
        return len(self._dataset.head_prompts) + sum(
            a.cursor for a in self._actuators
        )

    def _refresh_mi_progress(self, *, finished: bool) -> None:
        h = self._mi_progress_handle
        if h is None:
            return
        pg = get_progress()
        if finished:
            pg.update(
                h,
                completed=float(self._mi_total_steps),
                description=(
                    f"MemIndex · {self._scenario_id} · 完成"
                ),
            )
            return
        done = min(self._mi_completed_steps(), self._mi_total_steps)
        pg.update(
            h,
            completed=float(done),
            description=(
                f"MemIndex · {self._scenario_id} "
                f"({done}/{self._mi_total_steps})"
            ),
        )

    def evaluate(self, history: list[TurnResult]) -> ScenarioScore:
        return self._compile_score(history)

    # ── HEAD 阶段 ─────────────────────────────────────────────

    def _next_head(
        self, history: list[TurnResult],
    ) -> Optional[Turn]:
        if self._head_idx < len(self._dataset.head_prompts):
            msg = self._dataset.head_prompts[self._head_idx]
            self._head_idx += 1
            self._pending = _PendingAction(kind="head")
            return Turn(msg)

        self._phase = "run"
        return self._next_run(history)

    # ── RUN 阶段调度 ──────────────────────────────────────────

    def _next_run(self, history: list[TurnResult]) -> Optional[Turn]:
        tokens = self._count_tokens(history)
        current_turn_index = len(history)
        self._update_frozen(tokens)

        turn = self._try_from_queue(
            self._mark_queue, tokens, current_turn_index,
        )
        if turn:
            return turn

        turn = self._try_from_queue(self._queue, tokens, current_turn_index)
        if turn:
            return turn

        self._refill_queue()
        turn = self._try_from_queue(self._queue, tokens, current_turn_index)
        if turn:
            return turn

        if self._frozen:
            return self._emit_nonsense()

        return None

    def _try_from_queue(
        self,
        queue: list[int],
        tokens: int,
        current_turn_index: int,
    ) -> Optional[Turn]:
        """从队列中依次尝试取出可执行的序列步骤。"""
        while queue:
            aidx = queue.pop(0)
            turn = self._try_actuator_step(aidx, tokens, current_turn_index)
            if turn is not None:
                return turn
        return None

    def _try_actuator_step(
        self,
        aidx: int,
        tokens: int,
        current_turn_index: int,
    ) -> Optional[Turn]:
        """尝试从指定序列生成下一个 Turn。

        自动跳过依赖不满足的步骤。
        """
        act = self._actuators[aidx]

        while act.has_next:
            item = act.items[act.cursor]
            if self._check_deps(act, item):
                break
            item.activate = False
            item.executed = False
            act.intermediate[item.index] = item
            act.cursor += 1

        if not act.has_next:
            return None

        item = act.items[act.cursor]
        msg = resolve_refs(
            item.ask.replace("\\", ""), act.intermediate,
        )

        if act.start_tokens == 0:
            act.start_tokens = tokens
        self._emitted_turn_by_act_item.setdefault(act.name, {})[
            item.index
        ] = current_turn_index

        is_retry = act.pending_retry
        act.pending_retry = False

        turn_type = (
            TurnType.EVALUATION if item.score else TurnType.CONVERSATION
        )
        turn_meta: dict[str, object] = {
            "actuator_name": act.name,
            "item_index": item.index,
        }
        if item.score is not None:
            sc = item.score
            ground = resolve_refs(
                sc.answer.replace("\\", ""), act.intermediate,
            )
            dep_turns, dep_policy = self._resolve_eval_dependency_info(act, item)
            turn_meta.update({
                "eval_method_hint": (
                    "weighted_binary"
                    if sc.binary_items
                    else ("multi_score" if sc.is_multiple else self._eval_mode)
                ),
                "ground_truth": ground,
                # Keep both names for downstream compatibility.
                "evidence": ground,
                "evidence_content_map": {
                    "ground_truth": ground,
                },
                "dependency_turn_indices": dep_turns,
                "dependency_policy": dep_policy,
            })

        self._pending = _PendingAction(
            kind="actuator", actuator_idx=aidx, list_idx=act.cursor,
        )

        logger.debug(
            f"  [{act.name}] step {act.cursor}/{len(act.items)} "
            f"{'(retry) ' if is_retry else ''}"
            f"{'[EVAL]' if item.score else '[CONV]'}"
        )
        return Turn(msg, turn_type=turn_type, metadata=turn_meta)

    def _resolve_eval_dependency_info(
        self,
        act: _ActuatorState,
        item: SequenceItem,
    ) -> tuple[list[int], str]:
        """解析评估回合依赖的前序 turn 索引。"""
        if item.score is None:
            return [], ""
        explicit_dep_items: set[int] = {
            dep for dep in item.depend if dep != item.index
        }
        explicit_dep_items.update(_extract_ref_indices(item.ask))
        explicit_dep_items.update(_extract_ref_indices(item.score.answer))
        for bi in item.score.binary_items:
            explicit_dep_items.update(_extract_ref_indices(bi.answer))
        emitted = self._emitted_turn_by_act_item.get(act.name, {})
        if explicit_dep_items:
            turns = [
                emitted[i] for i in sorted(explicit_dep_items) if i in emitted
            ]
            return sorted(set(turns)), "ref"

        # 无明确 ref 时，回退到该子测试（actuator）内所有前置信息语句
        # （score is None）作为依赖。
        fallback_turns: list[int] = []
        for prev in act.items:
            if prev.index == item.index:
                break
            if prev.score is None and prev.index in emitted:
                fallback_turns.append(emitted[prev.index])
        return sorted(set(fallback_turns)), "subtest_prefix_fallback"

    # ── 依赖检查 ──────────────────────────────────────────────

    @staticmethod
    def _check_deps(act: _ActuatorState, item: SequenceItem) -> bool:
        for dep in item.depend:
            if dep == item.index:
                continue
            dep_item = act.intermediate.get(dep)
            if dep_item is None or not dep_item.activate:
                return False
        return True

    # ── 废话生成 ──────────────────────────────────────────────

    def _emit_nonsense(self) -> Turn:
        msg = _generate_nonsense(
            self._trivia, self._nonsense_tokens, self._encoder,
        )
        self._pending = _PendingAction(kind="nonsense")
        return Turn(msg)

    # ── 结果处理 ──────────────────────────────────────────────

    def _handle_result(self, history: list[TurnResult]) -> None:
        """处理上一个 Turn 的 Agent 回复。"""
        last = history[-1]
        p = self._pending
        assert p is not None
        self._pending = None

        if p.kind == "head":
            self._turn_labels[last.turn_index] = "head_prompt"
            return

        if p.kind == "nonsense":
            self._turn_labels[last.turn_index] = "noise"
            return

        act = self._actuators[p.actuator_idx]
        item = act.items[p.list_idx]
        item.response = last.response
        item.executed = True

        if item.score:
            self._do_eval(act, item, last)

            if not item.activate and item.retry:
                self._setup_retry(act, item, last.turn_index)
                return
        else:
            item.activate = True
            self._turn_labels[last.turn_index] = "information"

        act.intermediate[item.index] = item
        act.cursor = p.list_idx + 1

        tokens = self._count_tokens(history)
        if act.has_next:
            self._frozen.append((p.actuator_idx, tokens))

    def _do_eval(
        self,
        act: _ActuatorState,
        item: SequenceItem,
        last: TurnResult,
    ) -> None:
        """对有 score 的步骤执行行内 LLM 评估。"""
        assert item.score is not None
        sc = item.score
        ground = resolve_refs(
            sc.answer.replace("\\", ""), act.intermediate,
        )

        if sc.binary_items:
            bi_data = [
                {
                    "key": bi.key,
                    "weight": bi.weight,
                    "answer": resolve_refs(
                        bi.answer.replace("\\", ""), act.intermediate,
                    ),
                }
                for bi in sc.binary_items
            ]
            wr = evaluate_weighted_binary(
                bi_data, last.response, sc.score, self._eval_model,
            )
            sc.result = wr.score
            sc.reason = wr.reason
            sc.eval_method = "weighted_binary"

        elif sc.is_multiple:
            r = evaluate_multi_score(
                ground, last.response, sc.score, self._eval_model,
            )
            sc.result = r.score
            sc.reason = r.reason
            sc.eval_method = "multi_score"

        elif self._eval_mode == "score":
            r = evaluate_score(
                ground, last.response, sc.score, self._eval_model,
            )
            sc.result = r.score
            sc.reason = r.reason
            sc.eval_method = "score"

        else:
            r = evaluate_binary(
                ground, last.response, sc.score, self._eval_model,
            )
            sc.result = r.score
            sc.reason = r.reason
            sc.eval_method = "binary"

        if self._eval_mode == "score":
            item.activate = sc.result > 0
        else:
            item.activate = sc.result >= sc.score

        self._turn_labels[last.turn_index] = "evaluation"
        self._evals.append(_InlineEval(
            turn_index=last.turn_index,
            score=sc.result,
            max_score=sc.score,
            reason=sc.reason,
            actuator_name=act.name,
            eval_method=sc.eval_method,
        ))

    def _setup_retry(
        self,
        act: _ActuatorState,
        item: SequenceItem,
        turn_index: int,
    ) -> None:
        """设置重试：撤销行内评估，准备重发消息。"""
        if self._evals and self._evals[-1].turn_index == turn_index:
            self._evals.pop()
        self._turn_labels[turn_index] = "evaluation_retry"

        item.ask = str(item.retry)
        item.retry = None
        item.response = None
        item.executed = False
        item.activate = True

        act.pending_retry = True
        aidx = self._actuators.index(act)
        self._mark_queue.insert(0, aidx)

    # ── 冻结区 / 队列管理 ────────────────────────────────────

    def _update_frozen(self, current_tokens: int) -> None:
        """检查冻结区，将满足记忆距离的序列移入 mark_queue。"""
        still_frozen: list[tuple[int, int]] = []
        for aidx, frozen_at in self._frozen:
            act = self._actuators[aidx]
            if not act.has_next:
                continue
            progress = (act.cursor + 1) / len(act.items)
            needed = self._dataset.memory_distance * progress
            if current_tokens - act.start_tokens >= needed:
                self._mark_queue.append(aidx)
            else:
                still_frozen.append((aidx, frozen_at))
        self._frozen = still_frozen

    def _refill_queue(self) -> None:
        """将未完成且不在冻结区/优先队列中的序列重新加入常规队列。"""
        frozen_set = {aidx for aidx, _ in self._frozen}
        mark_set = set(self._mark_queue)
        self._queue = [
            i
            for i in range(len(self._actuators))
            if self._actuators[i].has_next
            and i not in frozen_set
            and i not in mark_set
        ]

    # ── Token 计数 ────────────────────────────────────────────

    def _count_tokens(self, history: list[TurnResult]) -> int:
        if not history:
            return 0
        parts: list[str] = []
        for tr in history:
            parts.append(f"user:{tr.user_input}")
            parts.append(f"assistant:{tr.response}")
        return len(self._encoder.encode("\n".join(parts)))

    # ── 最终评分编译 ──────────────────────────────────────────

    def _compile_score(
        self, history: list[TurnResult],
    ) -> ScenarioScore:
        """将行内评估记录汇总为 ScenarioScore + TurnAnnotation。"""
        eval_map: dict[int, _InlineEval] = {
            e.turn_index: e for e in self._evals
        }

        annotations: list[TurnAnnotation] = []
        for tr in history:
            label = self._turn_labels.get(tr.turn_index, "unknown")
            ts = None
            ev = eval_map.get(tr.turn_index)
            if ev:
                norm = ev.score / ev.max_score if ev.max_score > 0 else 0.0
                ts = TurnScore(
                    score=round(norm, 4),
                    passed=ev.score >= ev.max_score,
                    detail=(
                        f"[{ev.actuator_name}|{ev.eval_method}] "
                        f"{ev.score:.4f}/{ev.max_score:.4f} "
                        f"— {ev.reason}"
                    ),
                )
            annotations.append(TurnAnnotation(
                turn_index=tr.turn_index,
                label=label,
                score=ts,
            ))

        total_score = sum(e.score for e in self._evals)
        total_max = sum(e.max_score for e in self._evals)
        normalized = total_score / total_max if total_max > 0 else 0.0
        passed_count = sum(
            1 for e in self._evals if e.score >= e.max_score
        )

        return ScenarioScore(
            score=round(normalized, 4),
            passed=normalized >= 0.5,
            turn_annotations=annotations,
            detail=(
                f"{passed_count}/{len(self._evals)} evaluations passed, "
                f"weighted score = {total_score:.4f}/{total_max:.4f}"
            ),
            metadata={
                "total_score": total_score,
                "total_max_score": total_max,
                "eval_count": len(self._evals),
                "passed_count": passed_count,
                "memory_distance": self._dataset.memory_distance,
                "sequences": [a.name for a in self._actuators],
            },
        )
