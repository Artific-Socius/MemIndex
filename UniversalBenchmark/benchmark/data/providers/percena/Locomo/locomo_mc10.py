"""
LoCoMo-MC10: Groups multiple JSONL rows (``question_id``) into one Scene by parsing `conv-*` from question_id.

Loads ``transformed/locomo_mc10_with_name.json`` when present, else ``data/locomo_mc10.json``.
Uses byte-offset indexing to limit memory, grouping rows by conversation to avoid redundant history parsing.

Context is exposed via ``conversation_history()`` (default) or ``background_text()`` when
``use_background_text=True``. Very long haystacks may exceed model context; ``benchmark_lite``
``max_bg_chars`` only truncates the background path, not conversation preload.

Each turn line is prefixed with ``haystack_session_datetimes`` / ``haystack_session_ids`` when
present in the row (e.g. ``[session_time: 2023-05-08T13:56:00] [session_id: session_1] ...``)
so relative-time questions remain grounded after flattening.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .....interfaces.benchmark import Benchmark
from .....interfaces.evidence import EvidenceBundle
from .....interfaces.question import Question
from .....interfaces.scene import ConversationTurn, Scene
from .....interfaces.scoring import ScoringConfig

POOL_NAME = "locomo_mc10"

RAW_REPO_REL = Path("UniversalBenchmark") / "benchmark" / "data" / "raw" / "percena" / "Locomo" / "locomo-mc10"

_QUESTION_ID_RE = re.compile(rb'"question_id"\s*:\s*"([^"]+)"')

# Internal keys on flattened turn dicts (copies only; not from dataset JSON).
_SESSION_DATETIME_KEY = "_session_datetime"
_SESSION_ID_KEY = "_session_id"

LOCOMO_MC10_EVAL_PROMPT = """You are judging a multiple-choice answer for LoCoMo-MC10 (long-conversation memory).

The model was given prior conversation context, a question, and 10 options (indices 0-9).
The reference is the single correct answer text (gold option).

Decide whether the model's reply selects or states the same answer as the reference.
Treat paraphrases or equivalent dates/entities as correct if they clearly match the gold option.

Respond with exactly one token: True or False (capitalized), then a short one-line rationale.

Question (with options as shown to the model):
{question}

Reference (gold) answer:
{reference}

Model answer:
{model_answer}

Judgment (True/False):"""


def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError(
        "Could not find git repository root (.git). "
        "Run from inside the MemIndex clone, or set paths explicitly."
    )


def _raw_root_from_package() -> Path:
    return _find_repo_root(Path(__file__).resolve()) / RAW_REPO_REL


def _default_jsonl_path(raw_root: Path) -> Path:
    tr = raw_root / "transformed" / "locomo_mc10_with_name.json"
    da = raw_root / "data" / "locomo_mc10.json"
    if tr.is_file():
        return tr
    return da


def _index_jsonl(path: Path) -> tuple[dict[str, list[int]], list[str]]:
    """Map scene_id -> list of start byte offsets; preserve file order for list_scenes."""
    offsets: dict[str, list[int]] = {}
    ordered: list[str] = []
    with path.open("rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            m = _QUESTION_ID_RE.search(line)
            if not m:
                continue
            qid = m.group(1).decode("utf-8", errors="replace")
            scene_id = qid.rsplit("_q", 1)[0] if "_q" in qid else qid
            if scene_id not in offsets:
                offsets[scene_id] = []
                ordered.append(scene_id)
            offsets[scene_id].append(off)
    return offsets, ordered


def _turn_line(turn: dict[str, Any]) -> str:
    name = turn.get("name") or turn.get("speaker") or turn.get("role") or "?"
    content = turn.get("content") or turn.get("text") or ""
    body = f"{name}: {content}"
    dt = turn.get(_SESSION_DATETIME_KEY)
    sid = turn.get(_SESSION_ID_KEY)
    prefixes: list[str] = []
    if dt is not None and str(dt).strip():
        prefixes.append(f"[session_time: {dt}]")
    if sid is not None and str(sid).strip():
        prefixes.append(f"[session_id: {sid}]")
    if prefixes:
        return f"{' '.join(prefixes)} {body}"
    return body


def _row_haystack_to_flat_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten haystack_sessions and attach per-session datetime / id from the row."""
    haystack = row.get("haystack_sessions")
    if not isinstance(haystack, list):
        return []
    datetimes = row.get("haystack_session_datetimes") or []
    session_ids = row.get("haystack_session_ids") or []
    if not isinstance(datetimes, list):
        datetimes = []
    if not isinstance(session_ids, list):
        session_ids = []
    flat: list[dict[str, Any]] = []
    for si, session in enumerate(haystack):
        if not isinstance(session, list):
            continue
        dt = datetimes[si] if si < len(datetimes) else None
        sid = session_ids[si] if si < len(session_ids) else None
        for t in session:
            if not isinstance(t, dict):
                continue
            turn = dict(t)
            if dt is not None:
                turn[_SESSION_DATETIME_KEY] = str(dt)
            if sid is not None:
                turn[_SESSION_ID_KEY] = str(sid)
            flat.append(turn)
    return flat


def _flat_turns_to_conversation_history(flat: list[dict[str, Any]]) -> list[ConversationTurn]:
    """
    Pair messages into user/assistant turns. Consecutive ``user`` lines merge into one user_message;
    ``assistant`` completes the pair. Orphan assistant pairs with a placeholder user; trailing user
    pairs with empty assistant.
    """
    out: list[ConversationTurn] = []
    pending_user: list[str] = []

    for t in flat:
        role = str(t.get("role") or "").lower()
        line = _turn_line(t)
        if role == "user":
            pending_user.append(line)
            continue
        if pending_user:
            u = "\n".join(pending_user)
            out.append(ConversationTurn(user_message=u, assistant_response=line))
            pending_user = []
        else:
            out.append(
                ConversationTurn(
                    user_message="[prior context]",
                    assistant_response=line,
                )
            )

    if pending_user:
        out.append(
            ConversationTurn(
                user_message="\n".join(pending_user),
                assistant_response="",
            )
        )
    return out


def _haystack_to_background_text(row: dict[str, Any], *, join_sep: str = "\n\n") -> str:
    flat = _row_haystack_to_flat_turns(row)
    if not flat:
        return ""
    parts = [_turn_line(t) for t in flat]
    return join_sep.join(parts)


def _format_mc_question(
    question: str,
    choices: list[str],
) -> str:
    lines = [question.strip(), "", "Options (choose exactly one; reply with the index 0-9 or the full option text):"]
    for i, c in enumerate(choices):
        lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("Your answer:")
    return "\n".join(lines)


class LocomoMc10Scene(Scene):
    """Multiple MC items for the same conversation grouped by scene_id."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        scene_id: str,
        use_background_text: bool = False,
    ) -> None:
        self._scene_id = scene_id
        self._primary_row = rows[0]
        self._use_background_text = use_background_text
        
        self._questions: list[Question] = []
        for i, row in enumerate(rows):
            qid = str(row.get("question_id") or f"{scene_id}_q{i}")
            choices = list(row.get("choices") or [])
            if len(choices) != 10:
                raise ValueError(f"{qid}: expected 10 choices, got {len(choices)}")
            qtext = _format_mc_question(str(row.get("question", "")), choices)
            answer = str(row.get("answer", ""))
            idx = row.get("correct_choice_index")
            qtype = str(row.get("question_type", ""))
            raw_ev = row.get("evidence")
            if isinstance(raw_ev, list):
                ref_strings = [str(x) for x in raw_ev]
            else:
                ref_strings = []
            allow_miss = len(ref_strings) == 0

            self._questions.append(
                Question(
                    question_id=qid,
                    question_text=qtext,
                    ground_truth=answer,
                    evidence=EvidenceBundle(
                        evidence_type="locomo_mc10.mc",
                        payload={
                            "dataset_question_id": qid,
                            "choices": choices,
                            "correct_choice_index": idx,
                            "question_type": qtype,
                        },
                        references=ref_strings if ref_strings else None,
                        allow_missing_references=allow_miss,
                    ),
                    scoring=ScoringConfig(
                        eval_mode="score",
                        eval_prompt_key="locomo_mc10_mc",
                        max_score=1.0,
                    ),
                )
            )

    @property
    def scene_id(self) -> str:
        return self._scene_id

    @property
    def scene_name(self) -> str | None:
        qt = self._primary_row.get("question_type")
        return str(qt) if qt is not None else None

    @property
    def task_type(self) -> str | None:
        return "locomo_mc10_multiple_choice"

    def conversation_history(self) -> list[ConversationTurn]:
        if self._use_background_text:
            return []
        flat = _row_haystack_to_flat_turns(self._primary_row)
        return _flat_turns_to_conversation_history(flat)

    def background_text(self, *, max_chars: int | None = None, join_sep: str = "\n\n") -> str:
        if not self._use_background_text:
            return ""
        text = _haystack_to_background_text(self._primary_row, join_sep=join_sep)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    def questions(self) -> Iterable[Question]:
        return self._questions


class LocomoMc10Benchmark(Benchmark):
    """Percena LoCoMo-MC10: scene groups multiple queries by parsing `conv-*` from question_id."""

    def __init__(
        self,
        raw_root: Path | None = None,
        *,
        use_background_text: bool = False,
        jsonl_path: Path | None = None,
    ) -> None:
        self._raw_root = raw_root if raw_root is not None else _raw_root_from_package()
        self._use_background_text = use_background_text
        self._jsonl_path = jsonl_path if jsonl_path is not None else _default_jsonl_path(self._raw_root)
        self._offsets: dict[str, list[int]] = {}
        self._scene_order: list[str] = []
        if self._jsonl_path.is_file():
            self._offsets, self._scene_order = _index_jsonl(self._jsonl_path)

    @property
    def benchmark_name(self) -> str:
        return "Percena/LoCoMo-MC10"

    @property
    def eval_prompt(self) -> str:
        return LOCOMO_MC10_EVAL_PROMPT

    @property
    def raw_root(self) -> Path:
        return self._raw_root

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def row_count(self) -> int:
        return len(self._scene_order)

    def scene_index_table(self) -> list[dict[str, str]]:
        return [{"scene_id": sid, "scene_name": sid} for sid in self._scene_order]

    def list_scenes(self) -> Iterable[str]:
        return list(self._scene_order)

    def _load_rows(self, scene_id: str) -> list[dict[str, Any]]:
        offs = self._offsets.get(scene_id)
        if not offs:
            raise KeyError(
                f"Unknown scene_id {scene_id!r} for {self.benchmark_name!r}. "
                f"Indexed scenes: {len(self._scene_order)}"
            )
        rows: list[dict[str, Any]] = []
        with self._jsonl_path.open("rb") as f:
            for off in offs:
                f.seek(off)
                line = f.readline()
                rows.append(json.loads(line.decode("utf-8")))
        return rows

    def get_scene(self, scene_id: str) -> Scene:
        rows = self._load_rows(scene_id)
        return LocomoMc10Scene(
            rows,
            scene_id=scene_id,
            use_background_text=self._use_background_text,
        )
